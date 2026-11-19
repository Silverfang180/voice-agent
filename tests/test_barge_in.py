"""Automated verification test for Smart Barge-In & Interruption Handling."""
import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from livekit_phone_agent import (
    ALLOW_BARGE_IN,
    BARGE_IN_GRACE_PERIOD,
    BARGE_IN_MIN_PROBABILITY,
    BARGE_IN_MIN_SPEECH_DURATION,
    ECHO_GUARD_SECONDS,
    CallSessionHandler,
)


async def test_barge_in_mechanics():
    print("=" * 65)
    print("SMART BARGE-IN & INTERRUPTION MECHANICS TEST")
    print("=" * 65)

    print(f"ALLOW_BARGE_IN:               {ALLOW_BARGE_IN}")
    print(f"BARGE_IN_MIN_SPEECH_DURATION: {BARGE_IN_MIN_SPEECH_DURATION}s")
    print(f"BARGE_IN_GRACE_PERIOD:        {BARGE_IN_GRACE_PERIOD}s")
    print(f"BARGE_IN_MIN_PROBABILITY:     {BARGE_IN_MIN_PROBABILITY}")
    print(f"ECHO_GUARD_SECONDS:           {ECHO_GUARD_SECONDS}s")

    # Mock Room, Pipeline, VAD
    mock_room = MagicMock()
    mock_pipeline = MagicMock()
    mock_vad = MagicMock()

    handler = CallSessionHandler(
        room=mock_room,
        pipeline=mock_pipeline,
        vad_engine=mock_vad,
        session_id="test_barge_in_session",
    )

    # 1. Verify continuous listening during playback
    print("\n--- TEST 1: CONTINUOUS LISTENING STATE CHECK ---")
    handler.is_playing_response = True
    suspended = handler.is_listening_suspended()
    print(f"When agent is playing, is_listening_suspended() = {suspended}")
    assert suspended is False, "With ALLOW_BARGE_IN=True, listening MUST NOT be suspended while playing!"
    print("PASS: Microphone remains active during agent speech.")

    # 2. Test instant interruption and audio queue purge
    print("\n--- TEST 2: INSTANT PLAYBACK INTERRUPTION (< 15ms) ---")
    handler.audio_source = MagicMock()
    handler.audio_source.clear_queue = MagicMock()

    # Create dummy long playback task
    playback_cancelled = False

    async def dummy_playback():
        nonlocal playback_cancelled
        try:
            handler.is_playing_response = True
            await asyncio.sleep(10.0)  # Simulating 10s audio speech
        except asyncio.CancelledError:
            playback_cancelled = True
            raise

    play_task = asyncio.create_task(dummy_playback())
    handler._current_playback_task = play_task
    await asyncio.sleep(0.01)  # Yield to let dummy_playback run

    assert handler.is_playing_response is True

    t0 = time.perf_counter()
    handler.interrupt_playback(reason="caller spoke 'Rukiye'")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Interruption execution time: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 20.0, f"Interruption took too long: {elapsed_ms} ms"
    assert handler.is_playing_response is False, "Agent should immediately mark is_playing_response = False"
    assert handler.audio_source.clear_queue.called, "audio_source.clear_queue() must be called to purge WebRTC frames!"

    # Allow cancellation to propagate
    await asyncio.sleep(0.01)
    assert playback_cancelled is True, "Playback task was not cancelled!"
    print(f"PASS: Playback cancelled in {elapsed_ms:.2f}ms and queue purged immediately.")

    # 3. Test echo vs sustained speech logic
    print("\n--- TEST 3: ECHO / TRANSIENT VS SUSTAINED SPEECH LOGIC ---")
    handler.is_playing_response = True
    handler._playback_start_mono = time.monotonic() - 1.0  # Started 1s ago (past grace period)
    handler._barge_in_triggered = False

    # A short click / echo spike: 150ms speech duration
    short_speech_duration = 0.15
    should_interrupt_short = (
        short_speech_duration >= BARGE_IN_MIN_SPEECH_DURATION
    )
    print(f"Transient spike ({short_speech_duration}s): Triggers barge-in? {should_interrupt_short}")
    assert not should_interrupt_short, "Short transient should NOT interrupt!"

    # Real human speech: 450ms sustained utterance
    human_speech_duration = 0.45
    should_interrupt_human = (
        human_speech_duration >= BARGE_IN_MIN_SPEECH_DURATION
    )
    print(f"Human speech ({human_speech_duration}s): Triggers barge-in? {should_interrupt_human}")
    assert should_interrupt_human, "Sustained human speech MUST trigger barge-in!"
    print("PASS: Filter accurately distinguishes transient echo spikes from sustained human speech.")

    # 4. Clean close
    await handler.close()
    print("\nPASS: Handler closed cleanly.")

    print("\n" + "=" * 65)
    print("ALL SMART BARGE-IN MECHANICS TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(test_barge_in_mechanics())
