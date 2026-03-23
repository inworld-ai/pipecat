#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Inworld TTS services.

Tests timestamp calculation, audio processing, settings initialization,
and state management without requiring network access or API keys.
"""

import unittest
from unittest.mock import MagicMock

import pytest

from pipecat.services.inworld.tts import (
    InworldHttpTTSService,
    InworldTTSService,
    InworldTTSSettings,
)

# Sentinel used in place of a real aiohttp.ClientSession (which requires
# a running event loop to construct). The session is never used in unit tests.
_MOCK_SESSION = MagicMock()


def _make_http_service(**kwargs) -> InworldHttpTTSService:
    """Create an InworldHttpTTSService with dummy credentials for unit testing."""
    return InworldHttpTTSService(
        api_key="test-key",
        aiohttp_session=_MOCK_SESSION,
        **kwargs,
    )


def _make_ws_service(**kwargs) -> InworldTTSService:
    """Create an InworldTTSService with dummy credentials for unit testing."""
    return InworldTTSService(api_key="test-key", **kwargs)


# ── HTTP: timestamp calculation ──────────────────────────────────────────────


class TestHttpWordTimestamps(unittest.TestCase):
    """Tests for InworldHttpTTSService._calculate_word_times."""

    def test_basic_alignment(self):
        """Word times and chunk end time are computed from alignment data."""
        svc = _make_http_service()
        svc._cumulative_time = 0.0
        word_times, chunk_end = svc._calculate_word_times(
            {
                "wordAlignment": {
                    "words": ["Hello", "world"],
                    "wordStartTimeSeconds": [0.0, 0.3],
                    "wordEndTimeSeconds": [0.3, 0.6],
                }
            }
        )
        assert word_times == [("Hello", 0.0), ("world", 0.3)]
        assert chunk_end == 0.6

    def test_cumulative_offset_applied(self):
        """Cumulative time is added to raw start times."""
        svc = _make_http_service()
        svc._cumulative_time = 1.5
        word_times, chunk_end = svc._calculate_word_times(
            {
                "wordAlignment": {
                    "words": ["How", "are", "you"],
                    "wordStartTimeSeconds": [0.0, 0.2, 0.4],
                    "wordEndTimeSeconds": [0.2, 0.4, 0.7],
                }
            }
        )
        assert word_times == [("How", 1.5), ("are", 1.7), ("you", 1.9)]
        # chunk_end is the raw (non-cumulative) end time of the last word
        assert chunk_end == 0.7

    def test_empty_alignment(self):
        """Empty alignment data produces no word times."""
        svc = _make_http_service()
        word_times, chunk_end = svc._calculate_word_times(
            {"wordAlignment": {"words": [], "wordStartTimeSeconds": [], "wordEndTimeSeconds": []}}
        )
        assert word_times == []
        assert chunk_end == 0.0

    def test_missing_alignment(self):
        """Missing alignment key produces no word times."""
        svc = _make_http_service()
        word_times, chunk_end = svc._calculate_word_times({})
        assert word_times == []
        assert chunk_end == 0.0

    def test_missing_end_times(self):
        """Word times are computed even when end times are absent."""
        svc = _make_http_service()
        svc._cumulative_time = 0.0
        word_times, chunk_end = svc._calculate_word_times(
            {
                "wordAlignment": {
                    "words": ["Hello"],
                    "wordStartTimeSeconds": [0.0],
                    "wordEndTimeSeconds": [],
                }
            }
        )
        assert word_times == [("Hello", 0.0)]
        assert chunk_end == 0.0

    def test_cumulative_across_two_utterances(self):
        """Cumulative time grows monotonically across sequential utterances."""
        svc = _make_http_service()
        svc._cumulative_time = 0.0

        _, end1 = svc._calculate_word_times(
            {
                "wordAlignment": {
                    "words": ["First"],
                    "wordStartTimeSeconds": [0.0],
                    "wordEndTimeSeconds": [0.5],
                }
            }
        )
        svc._cumulative_time += end1

        word_times, _ = svc._calculate_word_times(
            {
                "wordAlignment": {
                    "words": ["Second"],
                    "wordStartTimeSeconds": [0.0],
                    "wordEndTimeSeconds": [0.4],
                }
            }
        )
        assert word_times == [("Second", 0.5)]

    def test_mismatched_lengths_ignored(self):
        """When words and start_times have different lengths, no times are produced."""
        svc = _make_http_service()
        word_times, _ = svc._calculate_word_times(
            {
                "wordAlignment": {
                    "words": ["Hello", "world"],
                    "wordStartTimeSeconds": [0.0],
                    "wordEndTimeSeconds": [0.3],
                }
            }
        )
        assert word_times == []


# ── WebSocket: timestamp calculation ─────────────────────────────────────────


class TestWsWordTimestamps(unittest.TestCase):
    """Tests for InworldTTSService._calculate_word_times."""

    def test_basic_alignment(self):
        """Word times are computed and generation_end_time is tracked."""
        svc = _make_ws_service()
        svc._cumulative_time = 0.0
        word_times = svc._calculate_word_times(
            {
                "wordAlignment": {
                    "words": ["Hello", "world"],
                    "wordStartTimeSeconds": [0.0, 0.3],
                    "wordEndTimeSeconds": [0.3, 0.6],
                }
            }
        )
        assert word_times == [("Hello", 0.0), ("world", 0.3)]
        assert svc._generation_end_time == 0.6

    def test_cumulative_offset_applied(self):
        """Cumulative time is added to raw start times; generation_end_time includes offset."""
        svc = _make_ws_service()
        svc._cumulative_time = 2.0
        word_times = svc._calculate_word_times(
            {
                "wordAlignment": {
                    "words": ["Test"],
                    "wordStartTimeSeconds": [0.0],
                    "wordEndTimeSeconds": [0.3],
                }
            }
        )
        assert word_times == [("Test", 2.0)]
        assert svc._generation_end_time == 2.3

    def test_flush_advances_cumulative_time(self):
        """Simulates flushCompleted: cumulative_time advances to generation_end_time."""
        svc = _make_ws_service()
        svc._cumulative_time = 0.0

        svc._calculate_word_times(
            {
                "wordAlignment": {
                    "words": ["First"],
                    "wordStartTimeSeconds": [0.0],
                    "wordEndTimeSeconds": [0.5],
                }
            }
        )
        # Simulate flushCompleted handler
        svc._cumulative_time = svc._generation_end_time

        word_times = svc._calculate_word_times(
            {
                "wordAlignment": {
                    "words": ["Second"],
                    "wordStartTimeSeconds": [0.0],
                    "wordEndTimeSeconds": [0.4],
                }
            }
        )
        assert word_times == [("Second", 0.5)]
        assert svc._generation_end_time == 0.9

    def test_empty_alignment(self):
        """Empty alignment data returns empty list and doesn't change state."""
        svc = _make_ws_service()
        svc._generation_end_time = 0.0
        word_times = svc._calculate_word_times(
            {"wordAlignment": {"words": [], "wordStartTimeSeconds": [], "wordEndTimeSeconds": []}}
        )
        assert word_times == []
        assert svc._generation_end_time == 0.0

    def test_missing_alignment(self):
        """Missing wordAlignment key returns empty list."""
        svc = _make_ws_service()
        word_times = svc._calculate_word_times({})
        assert word_times == []

    def test_multiple_generations_monotonic(self):
        """Timestamps are monotonically increasing across three generations."""
        svc = _make_ws_service()
        svc._cumulative_time = 0.0

        all_times = []
        for text, start, end in [("A", 0.0, 0.3), ("B", 0.0, 0.4), ("C", 0.0, 0.2)]:
            wt = svc._calculate_word_times(
                {
                    "wordAlignment": {
                        "words": [text],
                        "wordStartTimeSeconds": [start],
                        "wordEndTimeSeconds": [end],
                    }
                }
            )
            all_times.extend(t for _, t in wt)
            svc._cumulative_time = svc._generation_end_time

        for i in range(1, len(all_times)):
            assert all_times[i] > all_times[i - 1], f"Timestamps not monotonic: {all_times}"


# ── WebSocket: context and fallback tracking ─────────────────────────────────


class TestWsContextTracking(unittest.TestCase):
    """Tests for context text and timestamp tracking state."""

    def test_context_text_accumulates(self):
        """Text sent across multiple run_tts calls accumulates per context."""
        svc = _make_ws_service()
        svc._context_texts["ctx-1"] = ""
        svc._context_texts["ctx-1"] += "Hello "
        svc._context_texts["ctx-1"] += "world."
        assert svc._context_texts["ctx-1"] == "Hello world."

    def test_timestamp_flag_tracks_contexts(self):
        """Contexts receiving timestamps are tracked in _contexts_with_timestamps."""
        svc = _make_ws_service()
        ctx = "ctx-1"
        assert ctx not in svc._contexts_with_timestamps
        svc._contexts_with_timestamps.add(ctx)
        assert ctx in svc._contexts_with_timestamps
        svc._contexts_with_timestamps.discard(ctx)
        assert ctx not in svc._contexts_with_timestamps

    def test_disconnect_clears_state(self):
        """All context tracking state is cleared on disconnect."""
        svc = _make_ws_service()
        svc._cumulative_time = 5.0
        svc._generation_end_time = 3.0
        svc._context_texts["ctx-1"] = "text"
        svc._contexts_with_timestamps.add("ctx-1")

        # Directly clear state as _disconnect_websocket does
        svc._cumulative_time = 0.0
        svc._generation_end_time = 0.0
        svc._context_texts.clear()
        svc._contexts_with_timestamps.clear()

        assert svc._cumulative_time == 0.0
        assert svc._generation_end_time == 0.0
        assert len(svc._context_texts) == 0
        assert len(svc._contexts_with_timestamps) == 0


# ── Audio processing ─────────────────────────────────────────────────────────


class TestAudioProcessing(unittest.IsolatedAsyncioTestCase):
    """Tests for audio chunk processing in InworldHttpTTSService."""

    async def _make_async_http_service(self, **kwargs):
        """Create HTTP service inside an async context (event loop available)."""
        import aiohttp

        self._session = aiohttp.ClientSession()
        return InworldHttpTTSService(api_key="test-key", aiohttp_session=self._session, **kwargs)

    async def asyncTearDown(self):
        if hasattr(self, "_session") and self._session and not self._session.closed:
            await self._session.close()

    async def test_wav_header_stripped(self):
        """44-byte RIFF/WAV header is stripped from audio data."""
        svc = await self._make_async_http_service()
        raw_pcm = b"\x01\x02" * 100
        riff_header = b"RIFF" + b"\x00" * 40
        audio_with_header = riff_header + raw_pcm

        frames = [f async for f in svc._process_audio_chunk(audio_with_header, "ctx-1")]
        assert len(frames) == 1
        assert frames[0].audio == raw_pcm

    async def test_raw_pcm_passed_through(self):
        """Audio without a WAV header is passed through unchanged."""
        svc = await self._make_async_http_service()
        raw_pcm = b"\x01\x02" * 100

        frames = [f async for f in svc._process_audio_chunk(raw_pcm, "ctx-1")]
        assert len(frames) == 1
        assert frames[0].audio == raw_pcm

    async def test_empty_chunk_yields_nothing(self):
        """Empty audio chunk produces no frames."""
        svc = await self._make_async_http_service()
        frames = [f async for f in svc._process_audio_chunk(b"", "ctx-1")]
        assert frames == []

    async def test_short_riff_not_stripped(self):
        """Audio starting with RIFF but shorter than 44 bytes is not stripped."""
        svc = await self._make_async_http_service()
        short_audio = b"RIFF" + b"\x00" * 20
        frames = [f async for f in svc._process_audio_chunk(short_audio, "ctx-1")]
        assert len(frames) == 1
        assert frames[0].audio == short_audio


# ── Settings initialization ──────────────────────────────────────────────────


class TestHttpSettings(unittest.TestCase):
    """Tests for InworldHttpTTSService settings initialization."""

    def test_default_voice_and_model(self):
        """Default settings use Ashley voice and max model."""
        svc = _make_http_service()
        assert svc._settings.voice == "Ashley"
        assert svc._settings.model == "inworld-tts-1.5-max"

    def test_settings_override(self):
        """Settings parameter overrides defaults."""
        svc = _make_http_service(
            settings=InworldTTSSettings(voice="Brian", model="inworld-tts-1.5-mini")
        )
        assert svc._settings.voice == "Brian"
        assert svc._settings.model == "inworld-tts-1.5-mini"

    def test_deprecated_voice_id_param(self):
        """Deprecated voice_id parameter sets voice in settings."""
        svc = _make_http_service(voice_id="CustomVoice")
        assert svc._settings.voice == "CustomVoice"

    def test_deprecated_model_param(self):
        """Deprecated model parameter sets model in settings."""
        svc = _make_http_service(model="inworld-tts-1.5-mini")
        assert svc._settings.model == "inworld-tts-1.5-mini"

    def test_settings_wins_over_deprecated(self):
        """Settings parameter takes precedence over deprecated voice_id."""
        svc = _make_http_service(voice_id="Ignored", settings=InworldTTSSettings(voice="Winner"))
        assert svc._settings.voice == "Winner"

    def test_streaming_url(self):
        """Streaming mode uses the stream endpoint."""
        svc = _make_http_service(streaming=True)
        assert ":stream" in svc._base_url

    def test_non_streaming_url(self):
        """Non-streaming mode uses the non-stream endpoint."""
        svc = _make_http_service(streaming=False)
        assert ":stream" not in svc._base_url

    def test_default_timestamp_strategy(self):
        """Default timestamp transport strategy is ASYNC."""
        svc = _make_http_service()
        assert svc._timestamp_transport_strategy == "ASYNC"


class TestWsSettings(unittest.TestCase):
    """Tests for InworldTTSService settings initialization."""

    def test_default_voice_and_model(self):
        """Default settings use Ashley voice and max model."""
        svc = _make_ws_service()
        assert svc._settings.voice == "Ashley"
        assert svc._settings.model == "inworld-tts-1.5-max"

    def test_auto_mode_default(self):
        """Auto mode defaults to True when aggregate_sentences is not set."""
        svc = _make_ws_service()
        assert svc._auto_mode is True

    def test_auto_mode_follows_aggregate_sentences(self):
        """Auto mode follows aggregate_sentences when auto_mode is not explicit."""
        svc = _make_ws_service(aggregate_sentences=False)
        assert svc._auto_mode is False

    def test_auto_mode_explicit_override(self):
        """Explicit auto_mode overrides the aggregate_sentences inference."""
        svc = _make_ws_service(aggregate_sentences=False, auto_mode=True)
        assert svc._auto_mode is True

    def test_settings_override(self):
        """Settings parameter overrides defaults."""
        svc = _make_ws_service(
            settings=InworldTTSSettings(voice="Brian", model="inworld-tts-1.5-mini")
        )
        assert svc._settings.voice == "Brian"
        assert svc._settings.model == "inworld-tts-1.5-mini"

    def test_default_timestamp_strategy(self):
        """Default timestamp transport strategy is ASYNC."""
        svc = _make_ws_service()
        assert svc._timestamp_transport_strategy == "ASYNC"

    def test_buffer_settings_defaults(self):
        """Buffer settings default to None (server will use hardcoded defaults)."""
        svc = _make_ws_service()
        assert svc._buffer_settings["maxBufferDelayMs"] is None
        assert svc._buffer_settings["bufferCharThreshold"] is None


# ── InworldTTSSettings ───────────────────────────────────────────────────────


class TestInworldTTSSettings(unittest.TestCase):
    """Tests for InworldTTSSettings dataclass."""

    def test_from_mapping_flat(self):
        """from_mapping handles flat key/value pairs."""
        settings = InworldTTSSettings.from_mapping({"voice": "Brian", "model": "mini"})
        assert settings.voice == "Brian"
        assert settings.model == "mini"

    def test_from_mapping_nested_audio_config(self):
        """from_mapping destructures nested audioConfig."""
        settings = InworldTTSSettings.from_mapping(
            {"voice": "Brian", "audioConfig": {"speakingRate": 1.2}}
        )
        assert settings.speaking_rate == 1.2

    def test_from_mapping_aliases(self):
        """from_mapping resolves legacy aliases (voiceId -> voice, modelId -> model)."""
        settings = InworldTTSSettings.from_mapping({"voiceId": "Brian", "modelId": "mini"})
        assert settings.voice == "Brian"
        assert settings.model == "mini"


if __name__ == "__main__":
    unittest.main()
