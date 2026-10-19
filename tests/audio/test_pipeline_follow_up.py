"""Tests for follow-up listening and voice-controlled standby.

Covers Phase 5 requirements:
 1. Wake word activates command capture.
 2. Command executes successfully.
 3. Jarvis returns to follow-up listening (LISTENING) after the spoken response.
 4. A second command executes without a second wake word.
 5. Multiple consecutive commands work correctly.
 6. Idle timeout returns the assistant to IDLE.
 7. "Mute the mic" enters standby and prevents ordinary speech from triggering commands.
 8. Wake word restores normal command handling from MUTED.
 9. Standby transition does not disable wake-word detection (wake word still fed).
10. TTS playback is not mistaken for user speech (blackout + follow-up flag).
11. Commands are not duplicated (no double STT/LLM calls).
12. Cancellation and error handling remain functional.
13. Config schema v23 migration is correct.
"""

from __future__ import annotations

import asyncio

import pytest

from jarvis.audio.pipeline import _STANDBY_PHRASES, FOLLOW_UP_IDLE_TIMEOUT_DEFAULT, AudioPipeline
from jarvis.audio.protocols import (
    FRAME_BYTES,
    FRAME_DURATION_MS,
    AudioFrame,
    VADEvent,
    WakeWordResult,
)
from jarvis.core.events import EventBus, LLMResponseComplete
from jarvis.core.state_machine import ConversationalState, StateMachine

SILENCE_FRAME: AudioFrame = b"\x00" * FRAME_BYTES


class FakeSource:
    def __init__(self) -> None:
        self._on_frame = None

    async def start(self, on_frame):
        self._on_frame = on_frame

    async def stop(self):
        pass

    def push(self, frame: AudioFrame = SILENCE_FRAME) -> None:
        if self._on_frame is not None:
            self._on_frame(frame)


class FakeWakeWord:
    def __init__(self, detect_at=None) -> None:
        self.feed_count = 0
        if detect_at is None:
            self._detect_counts: set[int] = set()
        elif isinstance(detect_at, int):
            self._detect_counts = {detect_at}
        else:
            self._detect_counts = set(detect_at)
        self.reset_count = 0

    async def feed(self, frame):
        self.feed_count += 1
        if self.feed_count in self._detect_counts:
            return WakeWordResult(confidence=0.95)
        return None

    def reset(self):
        self.reset_count += 1


class FakeVAD:
    def __init__(self, script=None) -> None:
        self.feed_count = 0
        self.script = script or {}
        self.reset_count = 0

    async def feed(self, frame):
        self.feed_count += 1
        return self.script.get(self.feed_count)

    def reset(self):
        self.reset_count += 1


class FakeSTT:
    def __init__(self, transcriptions=None) -> None:
        self._transcriptions = list(transcriptions or ["hello world"])
        self.calls = 0

    async def transcribe(self, audio):
        self.calls += 1
        idx = min(self.calls - 1, len(self._transcriptions) - 1)
        return self._transcriptions[idx]


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.cancel_count = 0

    async def speak(self, text):
        self.spoken.append(text)

    async def speak_stream(self, text_chunks):
        async for chunk in text_chunks:
            self.spoken.append(chunk)

    async def cancel(self):
        self.cancel_count += 1


class GatedTTS(FakeTTS):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    async def speak_stream(self, text_chunks):
        async for chunk in text_chunks:
            self.spoken.append(chunk)
            self.started.set()
            await self.gate.wait()


def make_response_producer(*chunks: str):
    async def producer(text: str):
        for c in chunks:
            yield c
    return producer


def make_pipeline(
    *,
    detect_at=None,
    vad_script=None,
    transcriptions=None,
    response_chunks=("ok",),
    listening_timeout=6.0,
    follow_up_listening_enabled=True,
    follow_up_idle_timeout_seconds=60.0,
    on_standby_request=None,
    on_wake=None,
    barge_in_enabled=False,
    tts=None,
):
    bus = EventBus()
    sm = StateMachine(bus=bus)
    source = FakeSource()
    ww = FakeWakeWord(detect_at=detect_at)
    vad = FakeVAD(script=vad_script or {})
    stt = FakeSTT(transcriptions=transcriptions or ["hello world"])
    if tts is None:
        tts = FakeTTS()
    pipeline = AudioPipeline(
        source=source, wake_word=ww, vad=vad, stt=stt, tts=tts,
        response_producer=make_response_producer(*response_chunks),
        bus=bus, sm=sm,
        listening_timeout_seconds=listening_timeout,
        post_wake_blackout_ms=0,
        min_utterance_ms=0,
        barge_in_enabled=barge_in_enabled,
        on_wake=on_wake,
        follow_up_listening_enabled=follow_up_listening_enabled,
        follow_up_idle_timeout_seconds=follow_up_idle_timeout_seconds,
        on_standby_request=on_standby_request,
    )
    return pipeline, bus, sm, source, ww, vad, stt, tts


async def yield_loop(times: int = 30) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


async def wait_for_cs(sm, target, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while sm.conversational_state is not target:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(
                f"timed out waiting for CS={target.name}, "
                f"current={sm.conversational_state.name}"
            )
        await asyncio.sleep(0.005)
    await yield_loop()


# --- Req 1+2 ---

async def test_wake_word_activates_command_capture():
    pipeline, bus, sm, source, *_ = make_pipeline(detect_at=1)
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert sm.conversational_state is ConversationalState.LISTENING
    finally:
        await pipeline.stop()


async def test_command_executes_after_wake_word():
    completions = []
    pipeline, bus, sm, source, *_, stt, tts = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=["what time is it"],
        response_chunks=("it is noon",),
    )
    bus.subscribe(LLMResponseComplete, lambda e: completions.append(e))
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert stt.calls == 1
        assert tts.spoken == ["it is noon"]
        assert len(completions) == 1
    finally:
        await pipeline.stop()


# --- Req 3 ---

async def test_returns_to_listening_after_response():
    pipeline, bus, sm, source, *_ = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=["hello"],
        response_chunks=("hello back",),
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert pipeline._in_follow_up_listening is True
    finally:
        await pipeline.stop()


async def test_follow_up_disabled_returns_to_idle():
    pipeline, bus, sm, source, *_ = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=["hello"],
        response_chunks=("hello back",),
        follow_up_listening_enabled=False,
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await wait_for_cs(sm, ConversationalState.IDLE)
        assert pipeline._in_follow_up_listening is False
    finally:
        await pipeline.stop()


# --- Req 4 ---

async def test_second_command_without_wake_word():
    completions = []
    pipeline, bus, sm, source, ww, vad, stt, tts = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT, 3: VADEvent.ENDPOINT},
        transcriptions=["first command", "second command"],
        response_chunks=("ok",),
    )
    bus.subscribe(LLMResponseComplete, lambda e: completions.append(e))
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert stt.calls == 1
        source.push()
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert stt.calls == 2
        assert len(completions) == 2
    finally:
        await pipeline.stop()


# --- Req 5 ---

async def test_three_consecutive_commands_without_wake_word():
    completions = []
    pipeline, bus, sm, source, ww, vad, stt, tts = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT, 3: VADEvent.ENDPOINT, 5: VADEvent.ENDPOINT},
        transcriptions=["one", "two", "three"],
        response_chunks=("ok",),
    )
    bus.subscribe(LLMResponseComplete, lambda e: completions.append(e))
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        for _ in range(5):
            source.push()
            await asyncio.sleep(0)
        deadline = asyncio.get_running_loop().time() + 3.0
        while len(completions) < 3:
            if asyncio.get_running_loop().time() > deadline:
                break
            await asyncio.sleep(0.01)
        assert stt.calls == 3, f"expected 3 STT calls, got {stt.calls}"
        assert len(completions) == 3
    finally:
        await pipeline.stop()


# --- Req 6 ---

async def test_follow_up_idle_timeout_returns_to_idle():
    pipeline, bus, sm, source, *_ = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=["hello"],
        response_chunks=("hi",),
        follow_up_idle_timeout_seconds=0.05,
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert pipeline._in_follow_up_listening is True
        await asyncio.sleep(0.15)
        await yield_loop()
        assert sm.conversational_state is ConversationalState.IDLE
        assert pipeline._in_follow_up_listening is False
    finally:
        await pipeline.stop()


# --- Req 7 ---

async def test_mute_the_mic_calls_standby_callback():
    standby_calls = []

    pipeline, bus, sm, source, *_, stt, tts = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=["mute the mic"],
        response_chunks=("ok",),
        on_standby_request=lambda: standby_calls.append(1),
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await asyncio.sleep(0.1)
        await yield_loop()
        assert len(standby_calls) == 1
        assert stt.calls == 1
        assert tts.spoken == []
        assert sm.conversational_state is ConversationalState.IDLE
        assert pipeline._in_follow_up_listening is False
    finally:
        await pipeline.stop()


async def test_mute_the_mic_does_not_produce_response():
    pipeline, bus, sm, source, *_, stt, tts = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=["mute the mic"],
        response_chunks=("I will mute now",),
    )
    completions = []
    bus.subscribe(LLMResponseComplete, lambda e: completions.append(e))
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await asyncio.sleep(0.1)
        await yield_loop()
        assert len(completions) == 0
        assert tts.spoken == []
    finally:
        await pipeline.stop()


@pytest.mark.parametrize("phrase", sorted(_STANDBY_PHRASES))
async def test_all_standby_phrases_are_detected(phrase):
    standby_calls = []
    pipeline, bus, sm, source, *_, stt, tts = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=[phrase],
        response_chunks=("nope",),
        on_standby_request=lambda: standby_calls.append(1),
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await asyncio.sleep(0.1)
        await yield_loop()
        assert len(standby_calls) == 1, f"phrase {phrase!r} not detected"
        assert tts.spoken == []
    finally:
        await pipeline.stop()


async def test_standby_phrase_with_punctuation_detected():
    standby_calls = []
    pipeline, bus, sm, source, *_ = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=["Mute the mic."],
        on_standby_request=lambda: standby_calls.append(1),
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await asyncio.sleep(0.1)
        await yield_loop()
        assert len(standby_calls) == 1
    finally:
        await pipeline.stop()


# --- Req 8 ---

async def test_wake_word_reactivates_after_standby():
    standby_calls = []
    pipeline, bus, sm, source, ww, vad, stt, tts = make_pipeline(
        detect_at=[1, 5],
        vad_script={1: VADEvent.ENDPOINT, 2: VADEvent.ENDPOINT},
        transcriptions=["mute the mic", "what is the time"],
        response_chunks=("twelve o clock",),
        on_standby_request=lambda: standby_calls.append(1),
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await asyncio.sleep(0.1)
        await yield_loop()
        assert len(standby_calls) == 1
        assert sm.conversational_state is ConversationalState.IDLE
        for _ in range(3):
            source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert pipeline._in_follow_up_listening is False
    finally:
        await pipeline.stop()


# --- Req 9 ---

async def test_wake_word_still_fed_in_idle():
    pipeline, bus, sm, source, ww, *_ = make_pipeline(detect_at=None)
    await pipeline.start()
    try:
        assert sm.conversational_state is ConversationalState.IDLE
        for _ in range(5):
            source.push()
        await yield_loop()
        assert ww.feed_count == 5
    finally:
        await pipeline.stop()


# --- Req 10 ---

async def test_follow_up_flag_prevents_on_wake_during_followup():
    wake_calls = []
    pipeline, bus, sm, source, ww, vad, stt, tts = make_pipeline(
        detect_at=[1, 3],
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=["hello", "ignored"],
        response_chunks=("hi",),
        on_wake=lambda: wake_calls.append(1),
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert len(wake_calls) == 1
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert pipeline._in_follow_up_listening is True
        source.push()
        await yield_loop(50)
        assert len(wake_calls) == 1, "on_wake must not fire during follow-up listening"
    finally:
        await pipeline.stop()


async def test_blackout_prevents_frames_from_reaching_vad():
    blackout_ms = 90
    blackout_frames = -(-blackout_ms // FRAME_DURATION_MS)
    pipeline, bus, sm, source, ww, vad, stt, tts = make_pipeline(
        detect_at=1,
        vad_script={},
        transcriptions=["ok"],
    )
    pipeline._post_wake_blackout_ms = blackout_ms
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        vad_count_before = vad.feed_count
        for _ in range(blackout_frames):
            source.push()
        await yield_loop()
        vad_count_after = vad.feed_count
        assert vad_count_after == vad_count_before, (
            f"VAD was fed {vad_count_after - vad_count_before} times during blackout"
        )
    finally:
        await pipeline.stop()


# --- Req 11 ---

async def test_single_endpoint_produces_single_stt_call():
    pipeline, bus, sm, source, ww, vad, stt, tts = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=["hello"],
        response_chunks=("hi",),
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert stt.calls == 1
    finally:
        await pipeline.stop()


async def test_no_double_response_on_follow_up():
    completions = []
    pipeline, bus, sm, source, ww, vad, stt, tts = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=["hello"],
        response_chunks=("hi",),
    )
    bus.subscribe(LLMResponseComplete, lambda e: completions.append(e))
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        await asyncio.sleep(0.05)
        await yield_loop()
        assert len(completions) == 1
        assert stt.calls == 1
    finally:
        await pipeline.stop()


# --- Req 12 ---

async def test_response_chain_cancellation_resets_follow_up_flag():
    gated_tts = GatedTTS()
    pipeline, bus, sm, source, ww, vad, stt, tts = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=["hello"],
        response_chunks=("hi",),
        tts=gated_tts,
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await asyncio.wait_for(gated_tts.started.wait(), timeout=2.0)
        await wait_for_cs(sm, ConversationalState.SPEAKING)
        assert pipeline._in_follow_up_listening is False
        await pipeline._cancel_response_chain()
        await yield_loop()
        assert pipeline._in_follow_up_listening is False
    finally:
        gated_tts.gate.set()
        await pipeline.stop()


async def test_empty_transcription_does_not_enter_follow_up():
    pipeline, bus, sm, source, ww, vad, stt, tts = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=["   "],
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await wait_for_cs(sm, ConversationalState.IDLE)
        assert pipeline._in_follow_up_listening is False
    finally:
        await pipeline.stop()


async def test_no_producer_output_does_not_enter_follow_up():
    pipeline, bus, sm, source, ww, vad, stt, tts = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.ENDPOINT},
        transcriptions=["hello"],
        response_chunks=(),
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        await wait_for_cs(sm, ConversationalState.IDLE)
        assert pipeline._in_follow_up_listening is False
    finally:
        await pipeline.stop()


# --- Req 13: config schema v23 ---

def test_config_schema_v23_default_follow_up_enabled():
    from jarvis.core.config import JarvisConfig
    cfg = JarvisConfig()
    assert cfg.llm.follow_up_listening_enabled is True
    assert cfg.llm.follow_up_idle_timeout_seconds == 30


def test_config_migration_v22_to_v23():
    from jarvis.core.config import _migrate_v22_to_v23
    data = {"schema_version": 22, "llm": {"model": "qwen2.5:7b-instruct"}}
    result = _migrate_v22_to_v23(data)
    assert result["schema_version"] == 23
    assert result["llm"]["follow_up_listening_enabled"] is True
    assert result["llm"]["follow_up_idle_timeout_seconds"] == 30
    assert result["llm"]["model"] == "qwen2.5:7b-instruct"


def test_config_migration_v22_to_v23_preserves_existing():
    from jarvis.core.config import _migrate_v22_to_v23
    data = {
        "schema_version": 22,
        "llm": {"follow_up_listening_enabled": False, "follow_up_idle_timeout_seconds": 10},
    }
    result = _migrate_v22_to_v23(data)
    assert result["llm"]["follow_up_listening_enabled"] is False
    assert result["llm"]["follow_up_idle_timeout_seconds"] == 10


def test_standby_phrases_contains_mute_the_mic():
    from jarvis.audio.pipeline import _STANDBY_PHRASES
    assert "mute the mic" in _STANDBY_PHRASES


def test_pipeline_follow_up_timeout_default():
    assert FOLLOW_UP_IDLE_TIMEOUT_DEFAULT == 30.0
