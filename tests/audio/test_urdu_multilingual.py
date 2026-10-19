"""Tests for Urdu multilingual speech recognition, Urdu TTS routing, and Urdu intent handling."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.audio.stt import DEFAULT_INITIAL_PROMPT, _resolve_model_id
from jarvis.audio.urdu_tts import (
    _MULTILINGUAL_SENTENCE_BOUNDARY,
    MultilingualTTS,
    is_urdu_text,
)
from jarvis.core.config import (
    _SYSTEM_PROMPT_V7,
    DEFAULT_SYSTEM_PROMPT,
    migrate,
)
from jarvis.platform.app_discovery import normalize_open_query


class TestUrduTextDetection:
    def test_pure_urdu_detected(self):
        assert is_urdu_text("جی سر، میں آپ کی بات سن رہا ہوں۔")
        assert is_urdu_text("آپ کا کیا حال ہے؟")
        assert is_urdu_text("نوٹ پیڈ کھول دیا گیا ہے۔")

    def test_english_not_detected_as_urdu(self):
        assert not is_urdu_text("Hello, how are you doing today?")
        assert not is_urdu_text("Opening Chrome now, sir.")
        assert not is_urdu_text("Calculator has been launched.")

    def test_urdu_with_english_app_names_detected(self):
        # Urdu sentence mentioning English app names
        assert is_urdu_text("سر، میں نے Notepad کھول دیا ہے۔")
        assert is_urdu_text("Calculator تیار ہے۔")

    def test_empty_and_whitespace(self):
        assert not is_urdu_text("")
        assert not is_urdu_text("   \n\t  ")


class TestMultilingualSentenceBoundary:
    def test_urdu_full_stop_and_question_mark(self):
        text = "جی سر میں سن رہا ہوں۔ بتائیے کیا حکم ہے؟"
        matches = [m.group(0).strip() for m in _MULTILINGUAL_SENTENCE_BOUNDARY.finditer(text)]
        assert len(matches) >= 1
        assert "جی سر میں سن رہا ہوں۔" in matches

    def test_english_punctuation(self):
        text = "Yes, sir. Opening Chrome now! Anything else?"
        matches = [m.group(0).strip() for m in _MULTILINGUAL_SENTENCE_BOUNDARY.finditer(text)]
        assert "Yes, sir." in matches
        assert "Opening Chrome now!" in matches


class TestMultilingualTTSRouting:
    @pytest.mark.asyncio
    async def test_routes_urdu_to_urdu_tts(self):
        mock_english = AsyncMock()
        mock_urdu = AsyncMock()
        mock_urdu.is_loaded = True
        mock_urdu.speak = AsyncMock()

        tts = MultilingualTTS(english_tts=mock_english, urdu_tts=mock_urdu, urdu_enabled=True)
        tts.is_loaded = True

        await tts.speak("جی سر، بتائیے۔")
        mock_urdu.speak.assert_awaited_once_with("جی سر، بتائیے۔")
        mock_english.speak.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_routes_english_to_piper(self):
        mock_english = AsyncMock()
        mock_urdu = AsyncMock()
        mock_urdu.is_loaded = True
        mock_english.speak = AsyncMock()

        tts = MultilingualTTS(english_tts=mock_english, urdu_tts=mock_urdu, urdu_enabled=True)
        tts.is_loaded = True

        await tts.speak("Opening Chrome for you now, sir.")
        mock_english.speak.assert_awaited_once_with("Opening Chrome for you now, sir.")
        mock_urdu.speak.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_to_english_when_urdu_disabled(self):
        mock_english = AsyncMock()
        mock_urdu = AsyncMock()
        mock_urdu.is_loaded = True
        mock_english.speak = AsyncMock()

        tts = MultilingualTTS(english_tts=mock_english, urdu_tts=mock_urdu, urdu_enabled=False)
        tts.is_loaded = True

        await tts.speak("جی سر")
        mock_english.speak.assert_awaited_once_with("جی سر")
        mock_urdu.speak.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_to_english_when_urdu_not_loaded(self):
        mock_english = AsyncMock()
        mock_urdu = AsyncMock()
        mock_urdu.is_loaded = False
        mock_english.speak = AsyncMock()

        tts = MultilingualTTS(english_tts=mock_english, urdu_tts=mock_urdu, urdu_enabled=True)
        tts.is_loaded = True

        await tts.speak("جی سر")
        mock_english.speak.assert_awaited_once_with("جی سر")

    @pytest.mark.asyncio
    async def test_lifecycle_delegates_to_both(self):
        mock_english = AsyncMock()
        mock_urdu = AsyncMock()
        tts = MultilingualTTS(english_tts=mock_english, urdu_tts=mock_urdu, urdu_enabled=True)

        await tts.load()
        mock_english.load.assert_awaited_once()
        mock_urdu.load.assert_awaited_once()
        assert tts.is_loaded

        await tts.cancel()
        mock_english.cancel.assert_awaited_once()
        mock_urdu.cancel.assert_awaited_once()

        await tts.rewire_output("New Speakers")
        mock_english.rewire_output.assert_awaited_once_with("New Speakers")
        mock_urdu.rewire_output.assert_awaited_once_with("New Speakers")

        await tts.unload()
        mock_english.unload.assert_awaited_once()
        mock_urdu.unload.assert_awaited_once()
        assert not tts.is_loaded

    def test_property_delegation(self):
        mock_english = MagicMock()
        mock_english.speed = 1.0
        mock_english.volume = 0.8
        mock_english.voice_name = "alan"

        mock_urdu = MagicMock()
        mock_urdu.speed = 1.0
        mock_urdu.volume = 0.8

        tts = MultilingualTTS(english_tts=mock_english, urdu_tts=mock_urdu)
        assert tts.speed == 1.0
        assert tts.volume == 0.8
        assert tts.voice_name == "alan"

        tts.speed = 1.2
        assert mock_english.speed == 1.2
        assert mock_urdu.speed == 1.2

        tts.volume = 0.9
        assert mock_english.volume == 0.9
        assert mock_urdu.volume == 0.9


class TestSTTResolutionAndBiasing:
    def test_resolve_model_id_multilingual(self):
        # Auto language with base resolves to multilingual base
        assert _resolve_model_id("base", "auto") == "base"
        assert _resolve_model_id("base.en", "auto") == "base"
        assert _resolve_model_id("small.en", "ur") == "small"
        assert _resolve_model_id("tiny.en", "auto") == "tiny"

    def test_resolve_model_id_english_only(self):
        # English language retains English-only variants for max speed
        assert _resolve_model_id("base.en", "en") == "base.en"
        assert _resolve_model_id("tiny.en", "en") == "tiny.en"

    def test_default_urdu_prompt_biasing(self):
        assert "meri baat suno" in DEFAULT_INITIAL_PROMPT
        assert "Notepad" in DEFAULT_INITIAL_PROMPT
        assert "Calculator" in DEFAULT_INITIAL_PROMPT


class TestUrduCommandNormalization:
    def test_normalize_open_query_urdu_phrases(self):
        assert normalize_open_query("mere liye Notepad kholo") == "Notepad"
        assert normalize_open_query("Ab Calculator bhi kholo") == "Calculator"
        assert normalize_open_query("Chrome kholo") == "Chrome"
        assert normalize_open_query("نوٹ پیڈ کھولو") == "نوٹ پیڈ"
        assert normalize_open_query("کیلکولیٹر کھولو") == "کیلکولیٹر"

    def test_open_patterns_match_urdu(self):
        from jarvis.tools.local.open_app import _resolve_name
        assert _resolve_name("mere liye Notepad kholo") == "Notepad"
        assert _resolve_name("Ab Calculator bhi kholo") == "Calculator"
        assert _resolve_name("Chrome kholo") == "Chrome"
        assert _resolve_name("نوٹ پیڈ کھولو") == "نوٹ پیڈ"
        assert _resolve_name("کیلکولیٹر کھولو") == "کیلکولیٹر"


class TestSchemaV24Migration:
    def test_migration_v23_to_v24(self):
        old_data = {
            "schema_version": 23,
            "stt": {"model_size": "tiny.en", "language": "en"},
            "tts": {"voice": "en_GB-alan-medium"},
            "llm": {"system_prompt": _SYSTEM_PROMPT_V7},
        }
        migrated = migrate(old_data)
        assert migrated["schema_version"] == 24
        assert migrated["stt"]["model_size"] == "tiny.en"
        assert migrated["tts"]["urdu_enabled"] is True
        assert migrated["tts"]["urdu_voice"] == "hamza-amin/mms-tts-urd-fine-tuned"
        assert migrated["llm"]["system_prompt"] == DEFAULT_SYSTEM_PROMPT
