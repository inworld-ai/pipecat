#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Integration tests for Inworld TTS services.

These tests hit the real Inworld API and require INWORLD_API_KEY to be set.
They are automatically skipped in CI when the key is not available.
"""

import asyncio
import os

import aiohttp
import pytest
from dotenv import load_dotenv

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.inworld.tts import InworldHttpTTSService, InworldTTSService
from pipecat.tests.utils import run_test

load_dotenv(override=True)

INWORLD_API_KEY = os.getenv("INWORLD_API_KEY")
pytestmark = pytest.mark.skipif(INWORLD_API_KEY is None, reason="INWORLD_API_KEY is not set")


class FrameCollector(FrameProcessor):
    """Captures downstream frames for test assertions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frames: list[Frame] = []
        self.audio_bytes = 0
        self.ttfb: float | None = None
        self.done = asyncio.Event()

    def reset_turn(self):
        self.done.clear()
        self.audio_bytes = 0
        self.frames.clear()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.frames.append(frame)

        if isinstance(frame, MetricsFrame):
            for data in frame.data:
                if isinstance(data, TTFBMetricsData) and self.ttfb is None:
                    self.ttfb = data.value

        if isinstance(frame, TTSAudioRawFrame):
            self.audio_bytes += len(frame.audio)

        if isinstance(frame, TTSStoppedFrame):
            self.done.set()

        await self.push_frame(frame, direction)


def _api_key() -> str:
    assert INWORLD_API_KEY is not None
    return INWORLD_API_KEY


# ── HTTP TTS ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_http_streaming_produces_audio():
    """HTTP streaming TTS produces audio frames with correct sample rate."""
    async with aiohttp.ClientSession() as session:
        tts = InworldHttpTTSService(
            api_key=_api_key(),
            aiohttp_session=session,
            streaming=True,
            sample_rate=24000,
        )
        (down, up) = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Hello, how are you today?")]
        )

        errors = [f for f in up if isinstance(f, ErrorFrame)]
        assert len(errors) == 0, f"Unexpected errors: {[e.error for e in errors]}"

        audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
        assert len(audio) > 0, "Expected at least one audio frame"
        assert all(f.sample_rate == 24000 for f in audio)


@pytest.mark.asyncio
async def test_http_non_streaming_produces_audio():
    """HTTP non-streaming TTS produces audio frames."""
    async with aiohttp.ClientSession() as session:
        tts = InworldHttpTTSService(
            api_key=_api_key(),
            aiohttp_session=session,
            streaming=False,
            sample_rate=24000,
        )
        (down, up) = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Testing non-streaming mode.")]
        )

        errors = [f for f in up if isinstance(f, ErrorFrame)]
        assert len(errors) == 0, f"Unexpected errors: {[e.error for e in errors]}"

        audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
        assert len(audio) > 0, "Expected at least one audio frame"


@pytest.mark.asyncio
async def test_http_timestamps_received():
    """HTTP streaming TTS produces word-level timestamp text frames."""
    async with aiohttp.ClientSession() as session:
        tts = InworldHttpTTSService(
            api_key=_api_key(),
            aiohttp_session=session,
            streaming=True,
            sample_rate=24000,
        )
        (down, _) = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello world, this is a timestamp test.")],
        )

        text_frames = [f for f in down if isinstance(f, TTSTextFrame)]
        assert len(text_frames) > 0, "Expected word timestamp text frames"


@pytest.mark.asyncio
async def test_http_frame_ordering():
    """HTTP TTS produces frames in correct order: Started -> Audio/Text -> Stopped."""
    async with aiohttp.ClientSession() as session:
        tts = InworldHttpTTSService(
            api_key=_api_key(),
            aiohttp_session=session,
            streaming=True,
            sample_rate=24000,
        )
        (down, _) = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Frame ordering test.")])

        types = [type(f) for f in down]
        assert TTSStartedFrame in types
        assert TTSStoppedFrame in types

        started_idx = types.index(TTSStartedFrame)
        stopped_idx = types.index(TTSStoppedFrame)
        assert started_idx < stopped_idx, "TTSStartedFrame must come before TTSStoppedFrame"

        for i in range(started_idx + 1, stopped_idx):
            assert types[i] in (TTSAudioRawFrame, TTSTextFrame), (
                f"Unexpected frame between Started and Stopped: {types[i].__name__}"
            )


# ── WebSocket TTS ────────────────────────────────────────────────────────────


async def _run_ws_pipeline(tts, collector, coro, timeout=15.0):
    """Helper to run a WebSocket TTS pipeline with a coroutine that drives it."""
    pipeline = Pipeline([tts, collector])
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))
    runner = PipelineRunner(handle_sigint=False)
    run_task = asyncio.create_task(runner.run(task))

    try:
        await asyncio.sleep(0.5)
        await coro(task)
        await asyncio.wait_for(collector.done.wait(), timeout=timeout)
    finally:
        await task.queue_frame(EndFrame())
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass


async def _emit_sentence(task: PipelineTask, text: str, token_delay: float = 0.03):
    """Simulate LLM token-by-token output for a single sentence."""
    await task.queue_frame(LLMFullResponseStartFrame())
    for i, word in enumerate(text.split()):
        token = word if i == 0 else " " + word
        await task.queue_frame(TextFrame(text=token))
        await asyncio.sleep(token_delay)
    await task.queue_frame(LLMFullResponseEndFrame())


@pytest.mark.asyncio
async def test_ws_produces_audio():
    """WebSocket TTS produces audio through a full pipeline lifecycle."""
    tts = InworldTTSService(api_key=_api_key(), sample_rate=24000)
    collector = FrameCollector()

    async def drive(task):
        await _emit_sentence(task, "Hello world, how are you doing today?")

    await _run_ws_pipeline(tts, collector, drive)

    assert collector.audio_bytes > 0, "Expected audio data from WebSocket TTS"
    audio = [f for f in collector.frames if isinstance(f, TTSAudioRawFrame)]
    assert all(f.sample_rate == 24000 for f in audio)


@pytest.mark.asyncio
async def test_ws_timestamps_received():
    """WebSocket TTS produces word-level timestamp text frames."""
    tts = InworldTTSService(api_key=_api_key(), sample_rate=24000)
    collector = FrameCollector()

    async def drive(task):
        await _emit_sentence(task, "This sentence should produce word timestamps.")

    await _run_ws_pipeline(tts, collector, drive)

    text_frames = [f for f in collector.frames if isinstance(f, TTSTextFrame)]
    assert len(text_frames) > 0, "Expected word timestamp text frames from WebSocket TTS"


@pytest.mark.asyncio
async def test_ws_interruption_recovery():
    """Pipeline recovers cleanly and produces audio after an interruption."""
    tts = InworldTTSService(api_key=_api_key(), sample_rate=24000)
    collector = FrameCollector()

    pipeline = Pipeline([tts, collector])
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))
    runner = PipelineRunner(handle_sigint=False)
    run_task = asyncio.create_task(runner.run(task))

    try:
        await asyncio.sleep(0.5)

        # First turn
        await _emit_sentence(task, "First sentence before interruption.")
        await asyncio.wait_for(collector.done.wait(), timeout=15.0)
        first_audio = collector.audio_bytes
        assert first_audio > 0, "Expected audio from first turn"

        # Interrupt
        await task.queue_frame(InterruptionFrame())
        await asyncio.sleep(1.0)

        # Second turn after interruption
        collector.reset_turn()
        await _emit_sentence(task, "Second sentence after recovery.")
        await asyncio.wait_for(collector.done.wait(), timeout=15.0)
        assert collector.audio_bytes > 0, "Expected audio after interruption recovery"
    finally:
        await task.queue_frame(EndFrame())
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass


@pytest.mark.asyncio
async def test_ws_audio_completeness():
    """WebSocket TTS produces enough audio data for the input text duration."""
    tts = InworldTTSService(api_key=_api_key(), sample_rate=24000)
    collector = FrameCollector()

    async def drive(task):
        await _emit_sentence(
            task, "The quick brown fox jumps over the lazy dog near the river bank."
        )

    await _run_ws_pipeline(tts, collector, drive)

    # 16-bit mono at 24kHz = 48000 bytes/sec. A ~3-4 second sentence
    # should produce at least 1 second worth of audio.
    min_expected_bytes = 24000 * 2 * 1  # 1 second at 24kHz 16-bit mono
    assert collector.audio_bytes >= min_expected_bytes, (
        f"Audio too short: {collector.audio_bytes} bytes "
        f"(expected at least {min_expected_bytes} for ~1s)"
    )
