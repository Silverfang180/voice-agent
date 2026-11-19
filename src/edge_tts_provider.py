"""Modular Microsoft Edge Neural TTS Provider using edge-tts + in-memory PyAV MP3->WAV decoding.

IMPORTANT LIMITATIONS:
- edge-tts is an unofficial client of Microsoft's online Edge TTS service.
- Availability, latency, and rate limits are external and not SLA-guaranteed.
- Intended strictly for ₹0-cost prototypes and demonstration use only; do not send
  confidential customer PII through an unapproved external TTS service.
"""
import asyncio
import io
import os
import sys
import time
import wave
from typing import Dict, List, Optional, Tuple

import av
import edge_tts
import numpy as np

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1


def decode_mp3_to_wav24k(mp3_bytes: bytes) -> Tuple[bytes, float, float]:
    """Decode MP3 bytes in memory using PyAV into 24 kHz mono 16-bit PCM WAV bytes.

    Returns:
        Tuple[bytes, float, float]: (wav_bytes, decode_time_seconds, audio_duration_seconds)
    """
    t0 = time.perf_counter()
    if not mp3_bytes:
        raise ValueError("Empty MP3 buffer received from Edge TTS.")

    pcm_chunks: List[bytes] = []
    resampler = av.AudioResampler(
        format="s16",
        layout="mono",
        rate=TARGET_SAMPLE_RATE,
    )

    with av.open(io.BytesIO(mp3_bytes), mode="r", format="mp3") as container:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise RuntimeError("No audio stream found in MP3 output from Edge TTS.")

        for frame in container.decode(audio_stream):
            resampled_frames = resampler.resample(frame)
            if resampled_frames is None:
                continue
            if not isinstance(resampled_frames, list):
                resampled_frames = [resampled_frames]
            for rf in resampled_frames:
                arr = rf.to_ndarray()
                # Ensure int16 mono 1D contiguous bytes
                arr = np.asarray(arr, dtype=np.int16).reshape(-1)
                pcm_chunks.append(arr.tobytes())

        # Flush resampler
        flushed = resampler.resample(None)
        if flushed:
            if not isinstance(flushed, list):
                flushed = [flushed]
            for rf in flushed:
                arr = np.asarray(rf.to_ndarray(), dtype=np.int16).reshape(-1)
                pcm_chunks.append(arr.tobytes())

    pcm_bytes = b"".join(pcm_chunks)
    if not pcm_bytes:
        raise RuntimeError("Decoded MP3 contained zero PCM samples.")

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(TARGET_CHANNELS)
        wf.setsampwidth(2)  # 16-bit signed PCM
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

    wav_bytes = wav_buf.getvalue()
    duration = (len(pcm_bytes) // 2) / float(TARGET_SAMPLE_RATE)
    t_decode = time.perf_counter() - t0
    return wav_bytes, t_decode, duration


class EdgeTTSProvider:
    """TTS provider wrapping edge-tts with native Hindi and Marathi neural voices."""

    DEFAULT_VOICES = {
        "hi": "hi-IN-SwaraNeural",
        "mr": "mr-IN-AarohiNeural",
    }

    SUPPORTED_VOICES = {
        "hi-in-swaraneural": "hi-IN-SwaraNeural",
        "hi-in-madhurneural": "hi-IN-MadhurNeural",
        "mr-in-aarohineural": "mr-IN-AarohiNeural",
        "mr-in-manoharneural": "mr-IN-ManoharNeural",
        # Short / persona aliases mapped cleanly by language
        "swara": "hi-IN-SwaraNeural",
        "madhur": "hi-IN-MadhurNeural",
        "aarohi": "mr-IN-AarohiNeural",
        "manohar": "mr-IN-ManoharNeural",
    }

    def __init__(
        self,
        hi_voice: Optional[str] = None,
        mr_voice: Optional[str] = None,
        rate: str = "+0%",
    ):
        self.hi_voice = hi_voice or os.environ.get("EDGE_TTS_VOICE_HI", self.DEFAULT_VOICES["hi"]).strip()
        self.mr_voice = mr_voice or os.environ.get("EDGE_TTS_VOICE_MR", self.DEFAULT_VOICES["mr"]).strip()
        self.rate = rate
        self.last_decode_time = 0.0
        self.last_synth_network_time = 0.0

    def resolve_voice(self, voice: Optional[str], language: str) -> str:
        """Select the exact language-appropriate Microsoft Neural voice."""
        lang = (language or "hi").strip().lower()
        if voice:
            key = voice.strip().lower()
            if key in self.SUPPORTED_VOICES:
                chosen = self.SUPPORTED_VOICES[key]
                # Guard against language mismatch if a Hindi voice was requested for Marathi or vice versa
                if lang == "mr" and chosen.startswith("hi-IN-"):
                    return self.mr_voice
                if lang == "hi" and chosen.startswith("mr-IN-"):
                    return self.hi_voice
                return chosen
            # Male persona aliases from Gemini/Mithi
            if key in ("puck", "satvik", "charon", "fenrir", "male"):
                return "mr-IN-ManoharNeural" if lang == "mr" else "hi-IN-MadhurNeural"
            # Female persona aliases from Gemini/Mithi
            if key in ("kore", "charvi", "aoede", "female"):
                return "mr-IN-AarohiNeural" if lang == "mr" else "hi-IN-SwaraNeural"
        return self.mr_voice if lang == "mr" else self.hi_voice

    async def list_indian_voices(self) -> List[Dict]:
        """Query available Hindi and Marathi voices from Edge TTS."""
        voices = await edge_tts.list_voices()
        return [v for v in voices if v.get("Locale") in ("hi-IN", "mr-IN")]

    async def synthesize_async(
        self,
        text: str,
        language: str = "hi",
        voice: Optional[str] = None,
    ) -> Tuple[bytes, float, float]:
        """Asynchronously synthesize text -> MP3 -> 24 kHz mono WAV bytes.

        Returns:
            Tuple[bytes, float, float]: (wav_bytes, total_tts_seconds, audio_duration_seconds)
        """
        if not text or not text.strip():
            raise ValueError("Input text for Edge TTS synthesis cannot be empty.")

        resolved_voice = self.resolve_voice(voice, language)

        t_net_start = time.perf_counter()
        communicate = edge_tts.Communicate(text.strip(), voice=resolved_voice, rate=self.rate)
        mp3_chunks: List[bytes] = []
        try:
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    mp3_chunks.append(chunk["data"])
        except Exception as e:
            raise RuntimeError(f"Edge TTS streaming failed (voice={resolved_voice}): {e}") from None

        t_net = time.perf_counter() - t_net_start
        self.last_synth_network_time = t_net

        mp3_bytes = b"".join(mp3_chunks)
        if not mp3_bytes:
            raise RuntimeError(f"Edge TTS returned zero audio bytes (voice={resolved_voice}).")

        # Decode MP3 to 24 kHz mono 16-bit PCM WAV in worker thread
        wav_bytes, t_decode, duration = await asyncio.to_thread(decode_mp3_to_wav24k, mp3_bytes)
        self.last_decode_time = t_decode
        total_tts_time = t_net + t_decode
        return wav_bytes, total_tts_time, duration

    def synthesize(
        self,
        text: str,
        language: str = "hi",
        voice: Optional[str] = None,
    ) -> Tuple[bytes, float, float]:
        """Synchronous wrapper compatible with VoicePipeline.synthesize_speech()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Running inside an active event loop (e.g. FastAPI async or thread); spawn isolated loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, self.synthesize_async(text, language, voice)).result()
        return asyncio.run(self.synthesize_async(text, language, voice))

