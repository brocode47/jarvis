"""Urdu Text-to-Speech adapter using MMS-TTS (VITS) and MultilingualTTS router.

Implements TextToSpeech (audio/protocols.py) and Loadable (core/lifecycle.py).

Architecture:
- UrduMmsTTS wraps hamza-amin/mms-tts-urd-fine-tuned (VITS architecture).
  Native rate: 16,000 Hz (matches Jarvis's internal frame format).
  Runs in asyncio.to_thread to keep the event loop free.
- MultilingualTTS composites English PiperTTS and UrduMmsTTS.
  Inspects incoming text for Arabic/Urdu Unicode blocks:
  - Urdu script (\\u0600-\\u06FF, \\u0750-\\u077F, \\uFB50-\\uFDFF, \\uFE70-\\uFEFF)
    routes to UrduMmsTTS.
  - English / Latin script routes to PiperTTS.
- Supports streaming sentence-segmentation with both Western ([.!?]) and
  Urdu ([۔؟]) sentence terminators.
- Cancellation and amplitude envelope hooks for the UI overlay orb are
  fully preserved across both engines.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np
import sounddevice as sd

from jarvis.audio.tts import (
    SPEAK_STREAM_MAX_WAIT_SECONDS,
    PiperTTS,
)
from jarvis.core.events import EventBus

log = logging.getLogger(__name__)

# Unicode ranges covering Urdu Nastaliq / Perso-Arabic scripts:
# 0600-06FF (Arabic), 0750-077F (Arabic Supplement),
# FB50-FDFF (Arabic Presentation Forms-A), FE70-FEFF (Forms-B).
_URDU_CHAR_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
)

# Sentence boundary supporting English (. ! ?) and Urdu (۔ ؟) terminators
_MULTILINGUAL_SENTENCE_BOUNDARY = re.compile(
    r"(.+?)([.!?۔؟]+)(\s+|$)", re.DOTALL
)

DEFAULT_URDU_MODEL_ID = "hamza-amin/mms-tts-urd-fine-tuned"


def is_urdu_text(text: str) -> bool:
    """Return True if text contains Urdu / Arabic script characters."""
    if not text:
        return False
    return bool(_URDU_CHAR_RE.search(text))


class UrduMmsTTS:
    """Local Urdu neural TTS using fine-tuned VITS MMS model."""

    name: str = "urdu_tts"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_URDU_MODEL_ID,
        volume: float = 1.0,
        speed: float = 1.0,
        output_device: str | int | None = None,
        on_amplitude: Callable[[float], None] | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.model_id = model_id
        self.volume = max(0.0, min(1.0, volume))
        self.speed = speed
        self._output_device = output_device
        self._on_amplitude = on_amplitude
        self._bus = bus
        self._model: Any = None
        self._tokenizer: Any = None
        self._sample_rate: int = 16_000
        self._output_rate: int = 16_000
        self._cancel_event = threading.Event()
        self.is_loaded: bool = False

    async def load(self) -> None:
        if self.is_loaded:
            return
        log.info("[boot] loading Urdu MMS-TTS model (%s)...", self.model_id)
        try:
            from transformers import AutoTokenizer, VitsModel
        except ImportError as e:
            raise RuntimeError(
                "transformers is required for UrduMmsTTS"
            ) from e

        def _sync_load():
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, local_files_only=True
                )
                model = VitsModel.from_pretrained(
                    self.model_id, local_files_only=True
                )
            except Exception:
                # First run fallback if not pre-cached
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                model = VitsModel.from_pretrained(self.model_id)
            return tokenizer, model

        self._tokenizer, self._model = await asyncio.to_thread(_sync_load)
        if hasattr(self._model, "config") and hasattr(
            self._model.config, "sampling_rate"
        ):
            self._sample_rate = int(self._model.config.sampling_rate)
        self.is_loaded = True
        log.info("[boot] Urdu MMS-TTS model loaded (%d Hz)", self._sample_rate)

    async def unload(self) -> None:
        if not self.is_loaded:
            return
        await self.cancel()
        self._model = None
        self._tokenizer = None
        self.is_loaded = False
        log.info("Urdu MMS-TTS unloaded")

    async def speak(self, text: str) -> None:
        if not self.is_loaded or self._model is None or self._tokenizer is None:
            log.warning("UrduMmsTTS.speak() called before load; ignoring")
            return
        clean_text = text.strip()
        if not clean_text:
            return

        self._cancel_event.clear()
        try:
            await asyncio.to_thread(self._sync_synthesize_and_play, clean_text)
        except Exception:
            log.exception("UrduMmsTTS.speak failed for text %r", clean_text[:30])

    async def cancel(self) -> None:
        self._cancel_event.set()
        try:
            await asyncio.to_thread(sd.stop)
        except Exception:
            pass

    stop = cancel

    async def rewire_output(self, device: str | int | None) -> None:
        self._output_device = device

    async def speak_stream(
        self,
        text_chunks: AsyncIterator[str],
        *,
        max_wait_seconds: float = SPEAK_STREAM_MAX_WAIT_SECONDS,
    ) -> None:
        buffer = ""
        aiter = text_chunks.__aiter__()
        producer_done = False

        while True:
            while True:
                match = _MULTILINGUAL_SENTENCE_BOUNDARY.match(buffer)
                if not match:
                    break
                sentence = (match.group(1) + match.group(2)).strip()
                buffer = buffer[match.end():]
                if sentence:
                    await self.speak(sentence)

            if producer_done:
                tail = buffer.strip()
                if tail:
                    await self.speak(tail)
                return

            try:
                if buffer:
                    chunk = await asyncio.wait_for(
                        aiter.__anext__(), timeout=max_wait_seconds
                    )
                else:
                    chunk = await aiter.__anext__()
                buffer += chunk
            except StopAsyncIteration:
                producer_done = True
            except TimeoutError:
                tail = buffer.strip()
                if tail:
                    await self.speak(tail)
                    buffer = ""

    def _sync_synthesize_and_play(self, text: str) -> None:
        import torch

        if self._cancel_event.is_set():
            return

        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self._model(**inputs).waveform

        if self._cancel_event.is_set():
            return

        waveform = output.squeeze().cpu().numpy()
        if self.volume != 1.0:
            waveform = waveform * self.volume

        audio_int16 = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(
            np.int16
        )

        # Emit amplitude envelopes if hook provided
        if self._on_amplitude is not None:
            chunk_size = int(self._sample_rate * 0.033)  # ~30 Hz
            for i in range(0, len(audio_int16), chunk_size):
                if self._cancel_event.is_set():
                    return
                win = audio_int16[i : i + chunk_size]
                if len(win) > 0:
                    amp = float(np.max(np.abs(win)) / 32768.0)
                    try:
                        self._on_amplitude(min(1.0, amp * 2.0))
                    except Exception:
                        pass

        if self._cancel_event.is_set():
            return

        # Play through sounddevice
        try:
            sd.play(
                audio_int16,
                samplerate=self._sample_rate,
                device=self._output_device,
                blocking=True,
            )
        except Exception as e:
            log.warning("sd.play failed, retrying on system default: %s", e)
            try:
                sd.play(audio_int16, samplerate=self._sample_rate, blocking=True)
            except Exception:
                log.exception("Urdu audio playback failed")


class MultilingualTTS:
    """Composite TTS router that routes Urdu to UrduMmsTTS and English to PiperTTS.

    Implements TextToSpeech (audio/protocols.py) and Loadable (core/lifecycle.py).
    """

    name: str = "multilingual_tts"

    def __init__(
        self,
        *,
        english_tts: PiperTTS,
        urdu_tts: UrduMmsTTS | None = None,
        urdu_enabled: bool = True,
    ) -> None:
        self.english_tts = english_tts
        self.urdu_tts = urdu_tts
        self.urdu_enabled = urdu_enabled
        self.is_loaded: bool = False

    @property
    def speed(self) -> float:
        return self.english_tts.speed

    @speed.setter
    def speed(self, val: float) -> None:
        self.english_tts.speed = val
        if self.urdu_tts is not None:
            self.urdu_tts.speed = val

    @property
    def volume(self) -> float:
        return self.english_tts.volume

    @volume.setter
    def volume(self, val: float) -> None:
        self.english_tts.volume = val
        if self.urdu_tts is not None:
            self.urdu_tts.volume = val

    @property
    def voice_name(self) -> str:
        return getattr(self.english_tts, "voice_name", "")

    @voice_name.setter
    def voice_name(self, val: str) -> None:
        if hasattr(self.english_tts, "voice_name"):
            self.english_tts.voice_name = val

    async def load(self) -> None:
        if self.is_loaded:
            return
        await self.english_tts.load()
        if self.urdu_enabled and self.urdu_tts is not None:
            try:
                await self.urdu_tts.load()
            except Exception:
                log.exception("Failed to load Urdu TTS; English will be used as fallback")
        self.is_loaded = True

    async def unload(self) -> None:
        if not self.is_loaded:
            return
        await self.english_tts.unload()
        if self.urdu_tts is not None:
            await self.urdu_tts.unload()
        self.is_loaded = False

    async def speak(self, text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        if (
            self.urdu_enabled
            and self.urdu_tts is not None
            and self.urdu_tts.is_loaded
            and is_urdu_text(clean)
        ):
            await self.urdu_tts.speak(clean)
        else:
            await self.english_tts.speak(clean)

    async def cancel(self) -> None:
        await self.english_tts.cancel()
        if self.urdu_tts is not None:
            await self.urdu_tts.cancel()

    stop = cancel

    async def rewire_output(self, device: str | int | None) -> None:
        await self.english_tts.rewire_output(device)
        if self.urdu_tts is not None:
            await self.urdu_tts.rewire_output(device)

    async def speak_stream(
        self,
        text_chunks: AsyncIterator[str],
        *,
        max_wait_seconds: float = SPEAK_STREAM_MAX_WAIT_SECONDS,
    ) -> None:
        buffer = ""
        aiter = text_chunks.__aiter__()
        producer_done = False

        while True:
            while True:
                match = _MULTILINGUAL_SENTENCE_BOUNDARY.match(buffer)
                if not match:
                    break
                sentence = (match.group(1) + match.group(2)).strip()
                buffer = buffer[match.end():]
                if sentence:
                    await self.speak(sentence)

            if producer_done:
                tail = buffer.strip()
                if tail:
                    await self.speak(tail)
                return

            try:
                if buffer:
                    chunk = await asyncio.wait_for(
                        aiter.__anext__(), timeout=max_wait_seconds
                    )
                else:
                    chunk = await aiter.__anext__()
                buffer += chunk
            except StopAsyncIteration:
                producer_done = True
            except TimeoutError:
                tail = buffer.strip()
                if tail:
                    await self.speak(tail)
                    buffer = ""
