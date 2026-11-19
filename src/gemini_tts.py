"""Modular Gemini Text-to-Speech (TTS) Provider using official Google GenAI SDK."""
import io
import os
import sys
import time
import wave
from typing import Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class GeminiTTSProvider:
    """TTS Provider utilizing Google Gemini dedicated speech synthesis models."""

    SUPPORTED_VOICES = {
        "kore": "Kore",
        "puck": "Puck",
        "charon": "Charon",
        "fenrir": "Fenrir",
        "aoede": "Aoede",
        # Friendly aliases mapping from existing Mithi voice personas
        "charvi": "Kore",
        "satvik": "Puck",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.model_name = model_name or os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")
        
        # Securely read GEMINI_API_KEY
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from environment. Please add GEMINI_API_KEY to your .env file."
            )

        # Lazy import and client setup
        from google import genai
        self.client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1alpha"}
        )

    def resolve_voice(self, voice: Optional[str], language: str) -> str:
        """Resolve voice name to an official Gemini prebuilt voice."""
        if not voice:
            return "Kore" if language == "hi" else "Puck"
        clean_voice = voice.strip().lower()
        return self.SUPPORTED_VOICES.get(clean_voice, "Kore")

    def synthesize(
        self,
        text: str,
        language: str = "hi",
        voice: Optional[str] = None
    ) -> Tuple[bytes, float, float]:
        """Synthesize speech using Gemini TTS.
        
        Returns:
            Tuple[bytes, float, float]: (audio_bytes_wav, api_latency_seconds, audio_duration_seconds)
        """
        if not text or not text.strip():
            raise ValueError("Input text for synthesis cannot be empty.")

        resolved_voice = self.resolve_voice(voice, language)

        from google.genai import types

        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=resolved_voice
                )
            )
        )

        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config
        )

        t_start = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=text.strip(),
                config=config
            )
        except Exception as e:
            # Mask any accidental secrets in error message
            err_msg = str(e)
            if self.api_key and self.api_key in err_msg:
                err_msg = err_msg.replace(self.api_key, "[REDACTED_API_KEY]")
            raise RuntimeError(f"Gemini TTS generation failed: {err_msg}") from None

        t_latency = time.perf_counter() - t_start

        # Validate response structure
        if not response.candidates or not response.candidates[0].content.parts:
            raise RuntimeError("Gemini TTS returned an empty response candidate or part.")

        part = response.candidates[0].content.parts[0]
        if not hasattr(part, "inline_data") or not part.inline_data:
            raise RuntimeError("Gemini TTS response did not contain inline audio data.")

        raw_data = part.inline_data.data
        if not raw_data:
            raise RuntimeError("Gemini TTS returned zero audio bytes.")

        # Ensure containerized WAV format
        audio_bytes = raw_data
        if not audio_bytes.startswith(b"RIFF"):
            # If headerless raw PCM (24kHz, 16-bit mono), wrap in standard WAV header
            pcm_buffer = io.BytesIO()
            with wave.open(pcm_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_bytes)
            audio_bytes = pcm_buffer.getvalue()

        # Verify WAV metadata and calculate duration
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                channels = wf.getnchannels()
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                duration = nframes / float(framerate) if framerate > 0 else 0.0
        except Exception as e:
            raise RuntimeError(f"Failed to parse output WAV audio metadata: {e}")

        return audio_bytes, t_latency, duration

