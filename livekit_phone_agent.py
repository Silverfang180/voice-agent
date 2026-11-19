"""Integrated LiveKit SIP Phone Agent for ABC Business Services (Hindi & Marathi).

Architecture:
  LiveKit SIP -> Python LiveKit Agent (abc-receptionist)
    -> Silero Neural VAD (Half-duplex with echo guard)
    -> Groq Whisper STT (Hindi & Marathi auto-detection + quality validation)
    -> Groq ReceptionistBrain LLM + business_faq.json (factual knowledge base)
    -> Edge TTS primary / Gemini TTS / Mithi CPU fallback (24 kHz mono PCM)
    -> LiveKit AudioSource publication back into the SIP call.
"""
import asyncio
import io
import logging
import os
import re
import sys
import time
import wave
from typing import Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Load .env securely without printing credentials
load_dotenv(override=True)

from livekit import agents, rtc
from livekit.agents import AgentServer, AutoSubscribe, JobContext, JobProcess, vad as lk_vad
from livekit.plugins import silero

from src.voice_pipeline import OUTPUT_DIR, VoicePipeline

logger = logging.getLogger("abc-receptionist")

# Agent & Audio Constants
AGENT_NAME = "abc-receptionist"
INPUT_SAMPLE_RATE = 16000       # 16 kHz mono for Silero VAD & Whisper STT
INPUT_CHANNELS = 1
INPUT_FRAME_MS = 20             # 20 ms per audio frame (320 samples)

OUTPUT_SAMPLE_RATE = 24000      # 24 kHz mono matching Edge / Gemini / Mithi TTS
OUTPUT_CHANNELS = 1
OUTPUT_FRAME_MS = 20            # 20 ms per output frame (480 samples)

# Barge-In & Echo-Guard Configuration
ALLOW_BARGE_IN = os.environ.get("ALLOW_BARGE_IN", "true").strip().lower() in ("1", "true", "yes")
BARGE_IN_MIN_SPEECH_DURATION = float(os.environ.get("BARGE_IN_MIN_SPEECH_DURATION", "0.35"))
BARGE_IN_GRACE_PERIOD = float(os.environ.get("BARGE_IN_GRACE_PERIOD", "0.45"))
BARGE_IN_MIN_PROBABILITY = float(os.environ.get("BARGE_IN_MIN_PROBABILITY", "0.65"))
ECHO_GUARD_SECONDS = float(os.environ.get("ECHO_GUARD_SECONDS", "0.35"))

# Silero VAD & Endpointing Configuration (Priority 3)
VAD_MIN_SPEECH_SEC = float(os.environ.get("VAD_MIN_SPEECH_SEC", "0.20"))
VAD_MIN_SILENCE_SEC = float(os.environ.get("VAD_MIN_SILENCE_SEC", "0.80"))
VAD_PREFIX_PADDING_SEC = float(os.environ.get("VAD_PREFIX_PADDING_SEC", "0.35"))
VAD_MAX_UTTERANCE_SEC = float(os.environ.get("VAD_MAX_UTTERANCE_SEC", "18.0"))
VAD_ACTIVATION_THRESHOLD = float(os.environ.get("VAD_ACTIVATION_THRESHOLD", "0.55"))

# Marathi vs Hindi Devanagari lexical indicators
MARATHI_WORDS = {
    "आहे", "आहेत", "नाही", "तुम्ही", "तुमी", "तुमचं", "तुमचा", "तुमची", "तुमचे",
    "तुमच्या", "मला", "आमचं", "आमचा", "आमची", "आमचे", "कधी", "किती", "कसे", "कशी",
    "काय", "सांगा", "माहिती", "सकाळी", "संध्याकाळी", "दुपारी", "वाजता",
    "शुल्क", "पत्ता", "उघडे", "उघडते", "उघडता", "बंद", "हवी", "हवे", "पाहिजे",
    "नमस्कार", "धन्यवाद", "कोणत्या", "कोणती", "सेवा", "व्यवसाय", "कार्यालय",
    "येऊ", "शकतो", "शकते", "शकता", "येईल", "मिळतील", "कुठे", "माझा", "माझी",
    "माझे", "माझ्या", "तुम्हाला", "मध्ये", "बोलू", "बोला", "का",
    "हो", "होय", "शनिवारी", "रविवारी", "सोमवारी",
}
HINDI_WORDS = {
    "है", "हैं", "नहीं", "आप", "आपका", "आपकी", "आपके", "मुझे", "हमारा",
    "कब", "कितना", "कितने", "कैसे", "कैसी", "क्या", "बताइए", "बताओ",
    "जानकारी", "सुबह", "शाम", "दोपहर", "बजे", "फीस", "पता", "खुला",
    "खुलता", "खुलती", "चाहिए", "नमस्ते", "शुक्रिया", "कौन", "सेवाएं",
    "दुकान", "ऑफिस", "सकता", "सकती", "कहां", "कहाँ", "आएगा", "हाँ", "जी",
    "शनिवार", "रविवार", "सोमवार",
}

# Valid short greetings / confirmations that must NOT be rejected
VALID_SHORT_UTTERANCES = {
    "नमस्ते", "नमस्कार", "हाँ", "हा", "जी", "नहीं", "हो", "होय", "नाही",
    "ठीक", "अच्छा", "धन्यवाद", "शुक्रिया", "हेलो", "हॅलो",
    "hello", "hi", "yes", "no", "ok", "okay", "thanks", "thank you", "bye",
}

# Known Whisper silence/noise hallucinations
WHISPER_HALLUCINATIONS = {
    "thank you.", "thanks for watching", "thanks for watching.",
    "thank you for watching", "thank you for watching.",
    "subtitles by", "you", "you.", "mbc", "amara.org",
    "...", ".", "।", ",", "?", "!",
}

# Unclear speech polite retry messages (only used when STT output is clearly garbled/unintelligible)
CLARIFY_PROMPTS = {
    "hi": "क्षमा करें, आपकी आवाज़ स्पष्ट नहीं आ रही है। कृपया अपना प्रश्न फिर से बोलिए।",
    "mr": "क्षमस्व, तुमचा आवाज स्पष्ट येत नाहीये. कृपया तुमचा प्रश्न पुन्हा सांगा.",
}

DEFAULT_GREETING_TEXT = (
    "नमस्ते! ABC Business Services में आपका स्वागत है। "
    "आप हिंदी या मराठी में अपना सवाल पूछ सकते हैं।"
)

# Shared instances per worker process (pre-warmed in setup_fnc)
_shared_pipeline: Optional[VoicePipeline] = None
_shared_vad: Optional[lk_vad.VAD] = None


def get_pipeline() -> VoicePipeline:
    """Return shared VoicePipeline per worker process."""
    global _shared_pipeline
    if _shared_pipeline is None:
        t0 = time.perf_counter()
        _shared_pipeline = VoicePipeline()
        logger.info(
            "VoicePipeline initialized in %.2fs (TTS_PROVIDER=%s)",
            time.perf_counter() - t0,
            _shared_pipeline.tts_provider,
        )
    return _shared_pipeline


def get_silero_vad() -> lk_vad.VAD:
    """Return shared pre-loaded Silero Neural VAD instance."""
    global _shared_vad
    if _shared_vad is None:
        t0 = time.perf_counter()
        _shared_vad = silero.VAD.load(
            min_speech_duration=VAD_MIN_SPEECH_SEC,
            min_silence_duration=VAD_MIN_SILENCE_SEC,
            prefix_padding_duration=VAD_PREFIX_PADDING_SEC,
            max_buffered_speech=VAD_MAX_UTTERANCE_SEC,
            activation_threshold=VAD_ACTIVATION_THRESHOLD,
            sample_rate=INPUT_SAMPLE_RATE,
            force_cpu=True,
        )
        logger.info("Silero Neural VAD loaded in %.2fs", time.perf_counter() - t0)
    return _shared_vad


def pcm16_to_wav_bytes(
    pcm_bytes: bytes,
    sample_rate: int = INPUT_SAMPLE_RATE,
    channels: int = INPUT_CHANNELS,
) -> bytes:
    """Wrap raw 16-bit little-endian PCM bytes in a standard RIFF WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def decode_wav_to_pcm24k(wav_bytes: bytes) -> Tuple[bytes, float, float]:
    """Decode RIFF WAV bytes into 24 kHz mono int16 PCM, resampling if necessary.

    Returns:
        Tuple[bytes, float, float]: (pcm24k_bytes, audio_duration_s, decode_time_s)
    """
    t0 = time.perf_counter()
    if not wav_bytes or not wav_bytes.startswith(b"RIFF"):
        raise ValueError("Expected valid RIFF WAV bytes for PCM decoding (do not pass raw MP3).")

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw_pcm = wf.readframes(nframes)

    if sampwidth != 2:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes (expected 2 bytes / 16-bit PCM)")

    duration = nframes / float(framerate) if framerate > 0 else 0.0

    # Downmix stereo to mono if needed
    if channels > 1:
        samples = np.frombuffer(raw_pcm, dtype=np.int16).reshape(-1, channels)
        raw_pcm = samples.mean(axis=1).astype(np.int16).tobytes()

    # Resample to 24 kHz using LiveKit AudioResampler if needed
    if framerate != OUTPUT_SAMPLE_RATE:
        resampler = rtc.AudioResampler(
            input_rate=framerate,
            output_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
        )
        in_frame = rtc.AudioFrame(
            data=raw_pcm,
            sample_rate=framerate,
            num_channels=1,
            samples_per_channel=len(raw_pcm) // 2,
        )
        out_frames = resampler.push(in_frame) + resampler.flush()
        raw_pcm = b"".join(bytes(f.data) for f in out_frames)

    t_decode = time.perf_counter() - t0
    return raw_pcm, duration, t_decode


def detect_customer_language(
    transcript: str,
    whisper_lang: Optional[str],
    current_lang: str = "hi",
) -> Tuple[str, str]:
    """Detect whether the customer spoke Hindi ('hi') or Marathi ('mr').

    Returns:
        Tuple[str, str]: (language_code, detection_reason)
    """
    text = (transcript or "").strip()
    if not text:
        return current_lang, "empty_fallback"

    # Explicit language switch requests take highest priority
    lower_text = text.lower()
    if any(k in lower_text for k in ("मराठी", "महराठी", "marathi")):
        return "mr", "explicit_marathi_request"
    if any(k in lower_text for k in ("हिंदी", "हिन्दी", "hindi")):
        return "hi", "explicit_hindi_request"

    # Character 'ळ' is uniquely Marathi in Devanagari
    if "ळ" in text:
        return "mr", "marathi_char_la"

    tokens = set(re.findall(r"[\u0900-\u097F]+", text))
    mr_score = len(tokens & MARATHI_WORDS)
    hi_score = len(tokens & HINDI_WORDS)

    wl = (whisper_lang or "").strip().lower()
    if wl in ("mr", "marathi"):
        mr_score += 2

    if mr_score > hi_score:
        return "mr", f"lexical_mr({mr_score}>{hi_score})"
    if hi_score > mr_score:
        return "hi", f"lexical_hi({hi_score}>{mr_score})"

    if wl in ("mr", "marathi"):
        return "mr", "whisper_mr"
    if wl in ("hi", "hindi"):
        return "hi", "whisper_hi"

    return current_lang, f"keep_active({current_lang})"


def validate_transcript_quality(
    transcript: str,
    whisper_lang: Optional[str],
    audio_duration_s: float,
) -> Tuple[str, str]:
    """Classify transcript quality before sending to the LLM.

    Returns:
        Tuple[str, str]: (status, reason) where status is:
          - 'valid': safe to pass to LLM
          - 'ignore': silence/hallucination; ignore silently
          - 'unclear': garbled speech; politely ask customer to repeat
    """
    clean = (transcript or "").strip()
    if not clean:
        return "ignore", "empty"

    lower_stripped = clean.lower().strip(" .!?,।")
    if clean.lower() in WHISPER_HALLUCINATIONS or (
        lower_stripped in WHISPER_HALLUCINATIONS and lower_stripped not in VALID_SHORT_UTTERANCES
    ):
        return "ignore", f"whisper_hallucination({clean!r})"

    # Explicitly accept valid short greetings / yes-no answers
    if lower_stripped in VALID_SHORT_UTTERANCES:
        return "valid", "short_valid_utterance"

    # Reject non-Latin/non-Devanagari scripts (Korean/CJK/Arabic/Cyrillic noise artifacts)
    foreign_chars = re.findall(r"[\u0600-\u06FF\u0400-\u04FF\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]", clean)
    if len(foreign_chars) >= 2:
        return "unclear", f"unexpected_script(lang={whisper_lang})"

    # Check meaningful Devanagari or Latin letters
    valid_chars = re.findall(r"[A-Za-z\u0900-\u097F]", clean)
    if len(valid_chars) < 2:
        return "ignore", "no_letters"

    # If Whisper guessed an unrelated language (French/Spanish/Korean/etc.) AND text has no Devanagari
    wl = (whisper_lang or "").strip().lower()
    devanagari_chars = re.findall(r"[\u0900-\u097F]", clean)
    if wl and wl not in ("hindi", "marathi", "english", "hi", "mr", "en") and len(devanagari_chars) == 0:
        return "unclear", f"unexpected_language({wl})"

    # Very long captured audio (> 8s) yielding only 1 tiny word is usually severe line noise
    words = clean.split()
    if audio_duration_s > 8.0 and len(words) <= 1 and lower_stripped not in VALID_SHORT_UTTERANCES:
        return "unclear", f"low_density_long_audio({audio_duration_s:.1f}s/{len(words)}w)"

    return "valid", "ok"


def transcribe_customer_audio(
    pipeline: VoicePipeline,
    wav_bytes: bytes,
    language_hint: Optional[str] = None,
) -> Tuple[str, str, float]:
    """Transcribe customer WAV via Groq Whisper (NO generic hallucination-inducing prompt)."""
    t_start = time.perf_counter()
    kwargs = {
        "file": ("caller_turn.wav", wav_bytes),
        "model": pipeline.stt_model,
        "response_format": "verbose_json",
    }
    if language_hint in ("hi", "mr"):
        kwargs["language"] = language_hint

    response = pipeline.groq_client.audio.transcriptions.create(**kwargs)
    t_stt = time.perf_counter() - t_start
    transcript = getattr(response, "text", str(response)).strip()
    detected_lang = getattr(response, "language", "") or ""
    return transcript, detected_lang, t_stt


def load_or_generate_greeting(pipeline: VoicePipeline) -> bytes:
    """Load cached provider-matched greeting WAV or generate once via active TTS provider."""
    provider_tag = pipeline.tts_provider
    greeting_cache = os.path.join(OUTPUT_DIR, f"receptionist_greeting_{provider_tag}.wav")
    if os.path.isfile(greeting_cache) and os.path.getsize(greeting_cache) > 1000:
        with open(greeting_cache, "rb") as f:
            data = f.read()
        if data.startswith(b"RIFF"):
            logger.info("Loaded cached greeting WAV: %s", os.path.basename(greeting_cache))
            return data

    # Fallback to existing Gemini sample ONLY if active provider is Gemini (avoids burning quota!)
    if provider_tag == "gemini":
        legacy_hi = os.path.join(OUTPUT_DIR, "gemini_sample_hi.wav")
        if os.path.isfile(legacy_hi) and os.path.getsize(legacy_hi) > 1000:
            with open(legacy_hi, "rb") as f:
                return f.read()

    logger.info("Synthesizing initial greeting via %s TTS...", provider_tag)
    wav_bytes, t_tts, dur = pipeline.synthesize_speech(DEFAULT_GREETING_TEXT, language="hi")
    with open(greeting_cache, "wb") as f:
        f.write(wav_bytes)
    logger.info("Greeting synthesized (%.2fs audio in %.2fs) and cached.", dur, t_tts)
    return wav_bytes


class CallSessionHandler:
    """Half-duplex, echo-safe multilingual receptionist session for one SIP call."""

    def __init__(
        self,
        room: rtc.Room,
        pipeline: VoicePipeline,
        vad_engine: lk_vad.VAD,
        session_id: str,
    ):
        self.room = room
        self.pipeline = pipeline
        self.vad_engine = vad_engine
        self.session_id = session_id
        self.active_language = "hi"
        self.language_confidence_turns = 0

        self.audio_source = rtc.AudioSource(
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=OUTPUT_CHANNELS,
            queue_size_ms=2000,
        )

        # Barge-In, Playback & Echo-Guard State
        self.is_playing_response: bool = False
        self.is_processing: bool = False
        self._playback_start_mono: float = 0.0
        self._ignore_input_until: float = 0.0
        self._playback_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._closed = False

        # Active async tasks for cancellation during barge-in
        self._current_playback_task: Optional[asyncio.Task] = None
        self._current_turn_task: Optional[asyncio.Task] = None

        # Barge-in event flags
        self._speech_in_progress: bool = False
        self._speech_started_during_playback: bool = False
        self._barge_in_triggered: bool = False

    def is_listening_suspended(self) -> bool:
        """Check whether incoming audio processing is currently suspended.
        
        When ALLOW_BARGE_IN is True, we keep listening during playback so the
        caller can interrupt naturally. We only suspend during the post-playback echo guard.
        """
        if self._closed:
            return True
        if not ALLOW_BARGE_IN and (self.is_playing_response or self.is_processing):
            return True
        if time.monotonic() < self._ignore_input_until:
            return True
        return False

    def interrupt_playback(self, reason: str = "caller_barge_in"):
        """Immediately stop agent audio output and flush queued WebRTC frames."""
        if not self.is_playing_response and not self.is_processing:
            return

        logger.info("⚡ [Barge-in] Interrupting agent playback immediately (%s).", reason)
        self._barge_in_triggered = True

        # 1. Immediately clear audio buffer so remote participant hears silence in < 15ms
        try:
            self.audio_source.clear_queue()
        except Exception as e:
            logger.debug("Error clearing audio queue on interrupt: %s", e)

        # 2. Cancel active playback task if running
        if self._current_playback_task and not self._current_playback_task.done():
            self._current_playback_task.cancel()

        # 3. Cancel active turn task if agent was synthesizing/thinking
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()

        self.is_playing_response = False
        self.is_processing = False
        self._ignore_input_until = 0.0

    async def publish_audio_track(self):
        """Publish the receptionist's 24 kHz mono audio track into the LiveKit room."""
        track = rtc.LocalAudioTrack.create_audio_track("receptionist-voice", self.audio_source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        publication = await self.room.local_participant.publish_track(track, options)
        logger.info("Published receptionist audio track (sid=%s)", publication.sid)

    async def play_wav(
        self,
        wav_bytes: bytes,
        label: str = "response",
        turn_end_mono: Optional[float] = None,
    ) -> float:
        """Play WAV audio with instant barge-in cancellation and echo protection.

        Returns:
            float: decode_resample_seconds
        """
        async with self._playback_lock:
            if self._closed:
                return 0.0

            self.is_playing_response = True
            self._playback_start_mono = time.monotonic()
            self._current_playback_task = asyncio.current_task()
            self._barge_in_triggered = False
            t_decode = 0.0
            try:
                pcm24k, duration, t_decode = await asyncio.to_thread(decode_wav_to_pcm24k, wav_bytes)
                samples_per_frame = (OUTPUT_SAMPLE_RATE * OUTPUT_FRAME_MS) // 1000  # 480 samples
                bytes_per_frame = samples_per_frame * 2  # 960 bytes
                total_len = len(pcm24k)

                first_frame_logged = False
                logger.info(
                    "Playback START (%s): duration=%.2fs | decode=%.3fs | barge_in=%s",
                    label,
                    duration,
                    t_decode,
                    ALLOW_BARGE_IN,
                )

                offset = 0
                while offset < total_len and not self._closed:
                    if self._barge_in_triggered:
                        logger.info("Playback loop aborted mid-stream due to barge-in (%s).", label)
                        break

                    chunk = pcm24k[offset : offset + bytes_per_frame]
                    if len(chunk) < bytes_per_frame:
                        chunk = chunk + b"\x00" * (bytes_per_frame - len(chunk))

                    frame = rtc.AudioFrame(
                        data=chunk,
                        sample_rate=OUTPUT_SAMPLE_RATE,
                        num_channels=OUTPUT_CHANNELS,
                        samples_per_channel=samples_per_frame,
                    )
                    await self.audio_source.capture_frame(frame)

                    if not first_frame_logged:
                        first_frame_logged = True
                        if turn_end_mono is not None:
                            ttfa = time.monotonic() - turn_end_mono
                            logger.info("First response audio frame published (TTFA=%.3fs)", ttfa)

                    offset += bytes_per_frame

                if not self._closed and not self._barge_in_triggered:
                    await self.audio_source.wait_for_playout()

                if not self._barge_in_triggered:
                    logger.info(
                        "Playback COMPLETE (%s). Applying %.2fs echo guard before resuming listening.",
                        label,
                        ECHO_GUARD_SECONDS,
                    )
            except asyncio.CancelledError:
                self.audio_source.clear_queue()
                self._barge_in_triggered = True
                logger.info("Playback CANCELLED via task cancellation (%s).", label)
                raise
            except Exception as e:
                self.audio_source.clear_queue()
                logger.error("Audio playback error (%s): %s", label, e)
            finally:
                if not self._barge_in_triggered:
                    self._ignore_input_until = time.monotonic() + ECHO_GUARD_SECONDS
                else:
                    self._ignore_input_until = 0.0
                self.is_playing_response = False
                self._current_playback_task = None

            return t_decode

    async def _synthesize_reply(self, text: str, lang: str) -> Tuple[bytes, float, float]:
        """Synthesize reply asynchronously (non-blocking on event loop)."""
        if self.pipeline.tts_provider == "edge" and self.pipeline.edge_tts is not None:
            try:
                return await self.pipeline.edge_tts.synthesize_async(text=text, language=lang)
            except Exception as e:
                logger.warning("Async Edge TTS failed (%s); delegating to pipeline fallback.", e)
        return await asyncio.to_thread(self.pipeline.synthesize_speech, text, lang)

    async def handle_customer_utterance(
        self,
        pcm16k_bytes: bytes,
        speech_duration_s: float,
        silence_duration_s: float,
        end_reason: str,
    ):
        """Process one finalized customer speech segment through STT -> LLM -> TTS -> Playback."""
        if self._closed or not pcm16k_bytes:
            return

        async with self._turn_lock:
            self.is_processing = True
            turn_end_mono = time.monotonic()
            t_proc_start = time.perf_counter()
            try:
                captured_s = len(pcm16k_bytes) / (INPUT_SAMPLE_RATE * 2.0)
                wav_in = pcm16_to_wav_bytes(pcm16k_bytes, INPUT_SAMPLE_RATE, INPUT_CHANNELS)

                # Step 1: Groq Whisper STT (auto-detect language; no hallucination-inducing prompt)
                transcript, whisper_lang, t_stt = await asyncio.to_thread(
                    transcribe_customer_audio, self.pipeline, wav_in, None
                )

                # Step 2: Validate transcript quality
                status, reason = validate_transcript_quality(transcript, whisper_lang, captured_s)
                if status == "ignore":
                    logger.info(
                        "Ignored non-speech/hallucination (%s | stt=%.2fs | audio=%.2fs): %r",
                        reason,
                        t_stt,
                        captured_s,
                        transcript,
                    )
                    return

                if status == "unclear":
                    logger.info(
                        "Unclear speech detected (%s | stt=%.2fs | raw=%r); asking customer to repeat in '%s'.",
                        reason,
                        t_stt,
                        transcript,
                        self.active_language,
                    )
                    clarify_text = CLARIFY_PROMPTS[self.active_language]
                    try:
                        wav_out, t_tts, _ = await self._synthesize_reply(clarify_text, self.active_language)
                        await self.play_wav(wav_out, label=f"clarify_{self.active_language}", turn_end_mono=turn_end_mono)
                    except Exception as e:
                        logger.error("Failed to synthesize clarify prompt: %s", e)
                    return

                # Step 3: Detect Hindi vs Marathi dynamically
                lang, lang_reason = detect_customer_language(transcript, whisper_lang, self.active_language)
                if lang != self.active_language:
                    logger.info("Active caller language switched: %s -> %s (%s)", self.active_language, lang, lang_reason)
                self.active_language = lang

                logger.info(
                    "[Caller | lang=%s (%s) | whisper=%s | stt=%.3fs] %s",
                    lang,
                    lang_reason,
                    whisper_lang,
                    t_stt,
                    transcript,
                )

                # Step 4: Groq LLM ReceptionistBrain (grounded in business_faq.json + call memory)
                reply_text, t_llm = await asyncio.to_thread(
                    self.pipeline.generate_response, transcript, self.session_id, lang
                )
                if not reply_text:
                    logger.warning("LLM returned empty response.")
                    return

                logger.info("[Receptionist | lang=%s | llm=%.3fs] %s", lang, t_llm, reply_text)

                # Step 5: TTS (Edge primary / Gemini / Mithi fallback)
                try:
                    audio_bytes, t_tts, audio_dur = await self._synthesize_reply(reply_text, lang)
                except Exception as e:
                    logger.error("TTS synthesis failed gracefully (session stays alive): %s", e)
                    return

                t_total_before_play = time.perf_counter() - t_proc_start

                # Step 6: Half-duplex playback into LiveKit SIP room
                t_decode = 0.0
                if not self._closed:
                    t_decode = await self.play_wav(
                        audio_bytes,
                        label=f"reply_{lang}",
                        turn_end_mono=turn_end_mono,
                    )

                logger.info(
                    "TURN LATENCY BREAKDOWN | capture=%.2fs (speech=%.2fs, silence=%.2fs, end=%s) | "
                    "stt=%.3fs | llm=%.3fs | tts=%.3fs (%s) | decode=%.3fs | total_proc=%.3fs | reply_audio=%.2fs",
                    captured_s,
                    speech_duration_s,
                    silence_duration_s,
                    end_reason,
                    t_stt,
                    t_llm,
                    t_tts,
                    self.pipeline.tts_provider,
                    t_decode,
                    t_total_before_play + t_decode,
                    audio_dur,
                )
            except asyncio.CancelledError:
                logger.info("Customer turn task cancelled (barge-in or session end).")
            except Exception as e:
                logger.error("Unhandled error processing customer turn (recovering listener): %s", e)
            finally:
                self.is_processing = False

    async def listen_to_track(self, track: rtc.Track, participant_identity: str):
        """Consume caller audio through Silero Neural VAD with half-duplex suspension during playback."""
        logger.info(
            "Subscribed to remote caller audio track from '%s' (16kHz mono, Silero Neural VAD, half-duplex)",
            participant_identity,
        )
        audio_stream = rtc.AudioStream(
            track,
            sample_rate=INPUT_SAMPLE_RATE,
            num_channels=INPUT_CHANNELS,
            frame_size_ms=INPUT_FRAME_MS,
        )
        vad_stream = self.vad_engine.stream()

        speech_start_wall: Optional[float] = None

        async def _feed_vad():
            """Push caller frames to Silero VAD (always active during barge-in; paused only during echo guard)."""
            was_suspended = False
            try:
                async for event in audio_stream:
                    if self._closed:
                        break

                    if self.is_listening_suspended():
                        if not was_suspended:
                            was_suspended = True
                        continue

                    if was_suspended:
                        was_suspended = False
                        logger.info("Listening RESUMED after echo guard.")

                    vad_stream.push_frame(event.frame)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("Audio feed error (%s): %s", participant_identity, e)
            finally:
                vad_stream.end_input()

        async def _consume_vad():
            """Handle Silero START_OF_SPEECH, INFERENCE_DONE, and END_OF_SPEECH with smart barge-in."""
            nonlocal speech_start_wall
            try:
                async for ev in vad_stream:
                    if self._closed:
                        break

                    if ev.type == lk_vad.VADEventType.START_OF_SPEECH:
                        speech_start_wall = time.time()
                        self._speech_in_progress = True
                        self._speech_started_during_playback = self.is_playing_response
                        logger.info(
                            "VAD: Speech STARTED (ts=%.2f | during_playback=%s)",
                            speech_start_wall,
                            self._speech_started_during_playback,
                        )

                    elif ev.type == lk_vad.VADEventType.INFERENCE_DONE:
                        # Monitor for confirmed caller interruption while agent is actively speaking
                        if ALLOW_BARGE_IN and self.is_playing_response and not self._barge_in_triggered:
                            playback_elapsed = time.monotonic() - self._playback_start_mono
                            # Evaluate after initial grace period to prevent initial speaker pop echo
                            if playback_elapsed >= BARGE_IN_GRACE_PERIOD:
                                if ev.speaking and ev.raw_accumulated_speech >= BARGE_IN_MIN_SPEECH_DURATION:
                                    if ev.probability >= BARGE_IN_MIN_PROBABILITY:
                                        self.interrupt_playback(
                                            f"sustained speech={ev.raw_accumulated_speech:.2f}s, prob={ev.probability:.2f}"
                                        )

                    elif ev.type == lk_vad.VADEventType.END_OF_SPEECH:
                        self._speech_in_progress = False
                        was_during_playback = self._speech_started_during_playback
                        self._speech_started_during_playback = False

                        speech_end_wall = time.time()
                        pcm_chunks = [bytes(f.data) for f in ev.frames if f.data]
                        captured_pcm = b"".join(pcm_chunks)
                        captured_dur = len(captured_pcm) / (INPUT_SAMPLE_RATE * 2.0)

                        # If speech happened during playback but was too brief to be an intentional interruption:
                        if was_during_playback and not self._barge_in_triggered:
                            if ev.speech_duration < BARGE_IN_MIN_SPEECH_DURATION:
                                logger.debug(
                                    "Ignored brief transient sound during playback (speech=%.2fs < %.2fs)",
                                    ev.speech_duration,
                                    BARGE_IN_MIN_SPEECH_DURATION,
                                )
                                continue

                        end_reason = (
                            "max_utterance_cap"
                            if ev.speech_duration >= (VAD_MAX_UTTERANCE_SEC - 0.5)
                            else "silence_endpoint"
                        )
                        logger.info(
                            "VAD: Speech ENDED (speech=%.2fs | silence=%.2fs | captured=%.2fs | was_barge_in=%s)",
                            ev.speech_duration,
                            ev.silence_duration,
                            captured_dur,
                            self._barge_in_triggered,
                        )

                        if captured_pcm and ev.speech_duration >= VAD_MIN_SPEECH_SEC:
                            # Cancel any previous in-flight turn task
                            if self._current_turn_task and not self._current_turn_task.done():
                                self._current_turn_task.cancel()

                            # Dispatch turn task asynchronously without blocking VAD event loop!
                            self._current_turn_task = asyncio.create_task(
                                self.handle_customer_utterance(
                                    pcm16k_bytes=captured_pcm,
                                    speech_duration_s=ev.speech_duration,
                                    silence_duration_s=ev.silence_duration,
                                    end_reason=end_reason,
                                )
                            )
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("VAD consumer error (%s): %s", participant_identity, e)

        feed_task = asyncio.create_task(_feed_vad())
        consume_task = asyncio.create_task(_consume_vad())

        try:
            await asyncio.gather(feed_task, consume_task)
        except asyncio.CancelledError:
            feed_task.cancel()
            consume_task.cancel()
            await asyncio.gather(feed_task, consume_task, return_exceptions=True)
            logger.info("Audio listener cancelled for '%s'.", participant_identity)
        finally:
            await vad_stream.aclose()
            await audio_stream.aclose()

    async def close(self):
        """Clean up playback queue and conversation memory for this call."""
        self._closed = True
        self.interrupt_playback("session_closed")
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()
        self.audio_source.clear_queue()
        self.pipeline.brain.reset_session(self.session_id)
        logger.info("Call session '%s' closed and memory cleared.", self.session_id)


def prewarm(proc: JobProcess):
    """Priority 5: Pre-initialize Silero VAD, Groq client, TTS provider, and greeting before calls arrive."""
    t0 = time.perf_counter()
    vad_engine = get_silero_vad()
    pipeline = get_pipeline()
    greeting_wav = None
    try:
        greeting_wav = load_or_generate_greeting(pipeline)
    except Exception as e:
        logger.warning("Prewarm greeting generation skipped (%s); will retry on call.", e)
    proc.userdata["vad"] = vad_engine
    proc.userdata["pipeline"] = pipeline
    proc.userdata["greeting_wav"] = greeting_wav
    logger.info("Worker prewarm complete in %.2fs (zero blocking on first call).", time.perf_counter() - t0)


# Official LiveKit AgentServer with prewarm (2 warm runners, ephemeral HTTP port to avoid 8081 clashes)
server = AgentServer(
    setup_fnc=prewarm,
    num_idle_processes=int(os.environ.get("AGENT_IDLE_PROCESSES", "2")),
    initialize_process_timeout=20.0,
    port=int(os.environ.get("AGENT_HTTP_PORT", "0")),
)


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):
    """LiveKit Agent entrypoint invoked when SIP dispatches 'abc-receptionist'."""
    t_entry = time.perf_counter()
    logger.info("Job received! Connecting to room '%s'...", ctx.room.name)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info("Connected to LiveKit room '%s' as '%s' (%.3fs).", ctx.room.name, AGENT_NAME, time.perf_counter() - t_entry)

    # Retrieve pre-warmed components without blocking the asyncio event loop
    pipeline: VoicePipeline = ctx.proc.userdata.get("pipeline") or await asyncio.to_thread(get_pipeline)
    vad_engine: lk_vad.VAD = ctx.proc.userdata.get("vad") or await asyncio.to_thread(get_silero_vad)
    greeting_wav: Optional[bytes] = ctx.proc.userdata.get("greeting_wav")

    session_id = f"sip_{ctx.room.name}_{int(time.time())}"
    handler = CallSessionHandler(ctx.room, pipeline, vad_engine, session_id)
    listener_tasks: dict[str, asyncio.Task] = {}
    done_event = asyncio.Event()

    await handler.publish_audio_track()

    def _start_listening(track: rtc.Track, participant: rtc.RemoteParticipant):
        # Strictly ignore non-audio tracks and never listen to our own local participant
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if participant.identity == ctx.room.local_participant.identity:
            return
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            return

        key = f"{participant.identity}:{track.sid}"
        if key in listener_tasks:
            return
        logger.info("Attaching listener to caller '%s' (kind=%s)", participant.identity, participant.kind)
        listener_tasks[key] = asyncio.create_task(handler.listen_to_track(track, participant.identity))

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        _start_listening(track, participant)

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info("Participant joined: identity='%s', kind=%s", participant.identity, participant.kind)

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info("Participant disconnected: identity='%s'", participant.identity)
        if len(ctx.room.remote_participants) == 0:
            logger.info("All callers left room '%s'. Closing session.", ctx.room.name)
            done_event.set()

    @ctx.room.on("disconnected")
    def on_room_disconnected(*_args):
        logger.info("Disconnected from room '%s'.", ctx.room.name)
        done_event.set()

    # Wait for incoming SIP caller
    participant = await ctx.wait_for_participant()
    logger.info("Active caller detected: identity='%s', kind=%s", participant.identity, participant.kind)

    # Attach listener to already-subscribed audio tracks
    for p in ctx.room.remote_participants.values():
        for pub in p.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                _start_listening(pub.track, p)

    # Play greeting completely (half-duplex: listening is suspended until greeting + echo guard finish)
    await asyncio.sleep(0.30)
    t_greet_start = time.perf_counter()
    try:
        if greeting_wav is None:
            greeting_wav = await asyncio.to_thread(load_or_generate_greeting, pipeline)
        await handler.play_wav(greeting_wav, label="greeting")
        logger.info("Greeting startup + playout complete (%.2fs from connect).", time.perf_counter() - t_entry)
    except Exception as e:
        logger.error("Failed to play greeting audio: %s", e)

    try:
        await done_event.wait()
    finally:
        for task in listener_tasks.values():
            task.cancel()
        if listener_tasks:
            await asyncio.gather(*listener_tasks.values(), return_exceptions=True)
        await handler.close()
        logger.info("Agent session finished for room '%s'.", ctx.room.name)


if __name__ == "__main__":
    agents.cli.run_app(server)
