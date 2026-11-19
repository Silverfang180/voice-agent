"""Unified Voice Pipeline orchestrating STT (Groq Whisper), LLM (Groq Receptionist Brain), and TTS (Edge / Gemini / Mithi)."""
import io
import os
import subprocess
import sys
import time
import wave
from typing import Dict, Optional, Tuple

import httpx

try:
    from src.groq_llm import ReceptionistBrain
except ImportError:
    from groq_llm import ReceptionistBrain

# Load environment configuration
MITHI_SERVICE_URL = os.environ.get("MITHI_SERVICE_URL", "http://127.0.0.1:5002")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.abspath(os.environ.get("VOICE_OUTPUT_DIR", os.path.join(_PROJECT_ROOT, "output")))
DEFAULT_TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "edge").strip().lower()
TTS_ENABLE_FALLBACK = os.environ.get("TTS_ENABLE_FALLBACK", "true").strip().lower() in ("1", "true", "yes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MITHI_VALID_VOICES = {
    "satvik", "anirban", "nakul", "vasant", "siddhant",
    "charvi", "ahana", "vaidehi", "malini", "jahnavi",
}


def resolve_mithi_voice(voice: Optional[str], language: str) -> str:
    """Map any provider voice name (Edge/Gemini/Mithi) safely to a valid Mithi voice."""
    if voice:
        v = voice.strip().lower()
        if v in MITHI_VALID_VOICES:
            return v
        if any(tag in v for tag in ("madhur", "manohar", "puck", "charon", "fenrir", "male")):
            return "satvik"
        if any(tag in v for tag in ("swara", "aarohi", "kore", "aoede", "female")):
            return "charvi"
    return "charvi" if (language or "hi").lower() == "hi" else "satvik"


class VoicePipeline:
    def __init__(
        self,
        stt_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        tts_provider: Optional[str] = None,
    ):
        self.stt_model = stt_model or os.environ.get("GROQ_STT_MODEL", "whisper-large-v3").strip()
        self.brain = ReceptionistBrain(model_name=llm_model)
        self.groq_client = self.brain.client
        self.http_client = httpx.Client(trust_env=False)
        self.tts_provider = (tts_provider or os.environ.get("TTS_PROVIDER", DEFAULT_TTS_PROVIDER)).strip().lower()
        if self.tts_provider not in ("edge", "gemini", "mithi"):
            print(f"[VoicePipeline] Unknown TTS_PROVIDER '{self.tts_provider}'; defaulting to 'edge'.", file=sys.stderr)
            self.tts_provider = "edge"

        # Negative caches to avoid repeatedly attempting unavailable fallbacks
        self._mithi_http_backoff_until: float = 0.0
        self._docker_check_cache: Tuple[float, bool] = (0.0, False)
        self._gemini_backoff_until: float = 0.0

        # Initialize modular TTS providers
        self.edge_tts = None
        self.gemini_tts = None

        if self.tts_provider == "edge":
            try:
                try:
                    from src.edge_tts_provider import EdgeTTSProvider
                except ImportError:
                    from edge_tts_provider import EdgeTTSProvider
                self.edge_tts = EdgeTTSProvider()
                print(
                    f"[VoicePipeline] Edge TTS Provider initialized "
                    f"(hi={self.edge_tts.hi_voice}, mr={self.edge_tts.mr_voice})."
                )
            except Exception as e:
                print(f"[VoicePipeline] Failed to initialize Edge TTS: {e}.", file=sys.stderr)

        if self.tts_provider == "gemini":
            try:
                try:
                    from src.gemini_tts import GeminiTTSProvider
                except ImportError:
                    from gemini_tts import GeminiTTSProvider
                self.gemini_tts = GeminiTTSProvider()
                print(f"[VoicePipeline] Gemini TTS Provider initialized ({self.gemini_tts.model_name}).")
            except Exception as e:
                print(f"[VoicePipeline] Failed to initialize Gemini TTS: {e}.", file=sys.stderr)

    def check_mithi_health(self) -> Dict[str, str]:
        """Check availability of persistent Mithi TTS HTTP service."""
        try:
            r = self.http_client.get(f"{MITHI_SERVICE_URL}/health", timeout=2.0)
            if r.status_code == 200:
                data = r.json()
                self._mithi_http_backoff_until = 0.0
                return {
                    "status": "online",
                    "model": data.get("model", "mithi-v2-flash"),
                    "device": data.get("device", "cpu"),
                }
            return {"status": "error", "error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"status": "offline", "error": str(e)}

    def _is_docker_available(self) -> bool:
        """Check whether Docker Desktop daemon is reachable (cached 60s)."""
        now = time.time()
        cached_at, cached_ok = self._docker_check_cache
        if now - cached_at < 60.0:
            return cached_ok
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.5,
            )
            ok = res.returncode == 0
        except Exception:
            ok = False
        self._docker_check_cache = (now, ok)
        return ok

    def transcribe_audio(
        self,
        audio_file_bytes: bytes,
        filename: str,
        language: Optional[str] = "hi",
        prompt: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Transcribe audio bytes using Groq Whisper STT API."""
        t_start = time.perf_counter()
        kwargs = {
            "file": (filename, audio_file_bytes),
            "model": self.stt_model,
            "response_format": "verbose_json",
        }
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        response = self.groq_client.audio.transcriptions.create(**kwargs)
        t_stt = time.perf_counter() - t_start
        transcript = getattr(response, "text", str(response)).strip()
        return transcript, t_stt

    def generate_response(self, transcript: str, session_id: str, language: str) -> Tuple[str, float]:
        """Generate conversational answer via ReceptionistBrain."""
        res = self.brain.ask(user_question=transcript, session_id=session_id, language=language)
        reply = res.get("reply", "")
        t_llm = res.get("response_time", 0.0)
        return reply, t_llm

    def _synthesize_mithi(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        threads: int = 4,
    ) -> Tuple[bytes, float, float]:
        """Synthesize speech using persistent Mithi HTTP service, with safe Docker CLI fallback."""
        mithi_voice = resolve_mithi_voice(voice, language)
        mithi_lang = "mr" if (language or "hi").lower() == "mr" else "hi"
        t_start = time.perf_counter()
        now = time.time()

        # 1. Primary Mithi path: Persistent HTTP Microservice (skip if in backoff)
        if now >= self._mithi_http_backoff_until:
            try:
                payload = {
                    "text": text,
                    "language": mithi_lang,
                    "voice": mithi_voice,
                    "threads": threads,
                }
                r = self.http_client.post(f"{MITHI_SERVICE_URL}/synthesize", json=payload, timeout=90.0)
                if r.status_code == 200 and r.content:
                    t_tts = time.perf_counter() - t_start
                    duration = float(r.headers.get("X-Audio-Duration", len(r.content) / (24000 * 2)))
                    return r.content, t_tts, duration
            except Exception as e:
                self._mithi_http_backoff_until = now + 30.0
                print(
                    f"[VoicePipeline] Mithi HTTP service unavailable ({e}). Backing off 30s.",
                    file=sys.stderr,
                )
        else:
            print("[VoicePipeline] Skipping Mithi HTTP (currently in offline backoff).", file=sys.stderr)

        # 2. Docker CLI fallback: only attempt if Docker Desktop daemon is actually running
        if not self._is_docker_available():
            raise RuntimeError(
                "Mithi TTS fallback unavailable: HTTP service is offline and Docker Desktop daemon is not running."
            )

        try:
            project_dir = os.path.abspath(os.path.dirname(__file__))
            temp_out = os.path.join(OUTPUT_DIR, f"temp_{int(time.time() * 1000)}.wav")
            docker_cmd = [
                "docker", "run", "--rm",
                "-v", f"{os.path.join(project_dir, 'mithi-tts', 'model')}:/app/model",
                "-v", f"{OUTPUT_DIR}:/app/output",
                "-v", f"{os.path.join(project_dir, 'mithi-tts', 'test_mithi_tts.py')}:/app/test_mithi_tts.py",
                "mithi-v2-flash:latest",
                "python", "/app/test_mithi_tts.py",
                "--language", mithi_lang,
                "--voice", mithi_voice,
                "--device", "cpu",
                "--threads", str(threads),
                "--text", text,
                "--output", f"/app/output/{os.path.basename(temp_out)}",
            ]
            subprocess.run(docker_cmd, check=True)
            if os.path.isfile(temp_out):
                with open(temp_out, "rb") as f:
                    audio_bytes = f.read()
                os.remove(temp_out)
                t_tts = time.perf_counter() - t_start
                duration = len(audio_bytes) / (24000.0 * 2.0)
                return audio_bytes, t_tts, duration
        except Exception as e:
            print(f"[VoicePipeline] Docker Mithi TTS fallback failed: {e}", file=sys.stderr)

        raise RuntimeError("Failed to synthesize speech via Mithi HTTP service and Docker fallback.")

    def synthesize_speech(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        threads: int = 4,
    ) -> Tuple[bytes, float, float]:
        """Synthesize speech using selected TTS provider (edge | gemini | mithi) with safe fallback."""
        errors = []

        # 1. Edge TTS Provider (default for ₹0 prototype)
        if self.tts_provider == "edge":
            if self.edge_tts is None:
                from edge_tts_provider import EdgeTTSProvider
                self.edge_tts = EdgeTTSProvider()
            try:
                return self.edge_tts.synthesize(text=text, language=language, voice=voice)
            except Exception as e:
                errors.append(f"Edge TTS: {e}")
                print(f"[VoicePipeline] Edge TTS failed: {e}", file=sys.stderr)
                if not TTS_ENABLE_FALLBACK:
                    raise RuntimeError(f"Edge TTS failed and fallback is disabled: {e}")

        # 2. Gemini TTS Provider (optional; respects 429 backoff without wasting quota)
        elif self.tts_provider == "gemini":
            now = time.time()
            if now < self._gemini_backoff_until:
                errors.append("Gemini TTS: in 429 quota backoff")
            else:
                if self.gemini_tts is None:
                    from gemini_tts import GeminiTTSProvider
                    self.gemini_tts = GeminiTTSProvider()
                try:
                    return self.gemini_tts.synthesize(text=text, language=language, voice=voice)
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        self._gemini_backoff_until = now + 300.0
                    errors.append(f"Gemini TTS: {err_str}")
                    print(f"[VoicePipeline] Gemini TTS failed: {err_str}", file=sys.stderr)
                    if not TTS_ENABLE_FALLBACK:
                        raise RuntimeError(f"Gemini TTS failed and fallback is disabled: {err_str}")

        # 3. Mithi TTS Provider (selected explicitly or fallback if enabled)
        if self.tts_provider == "mithi" or TTS_ENABLE_FALLBACK:
            try:
                return self._synthesize_mithi(text, language, voice=voice, threads=threads)
            except Exception as e:
                errors.append(f"Mithi TTS: {e}")

        raise RuntimeError("TTS synthesis failed (" + " | ".join(errors) + ")")

    def process_voice_turn(
        self,
        audio_file_bytes: bytes,
        filename: str,
        language: str = "hi",
        session_id: str = "default",
        voice: Optional[str] = None,
    ) -> Dict:
        """Execute full end-to-end voice turn: STT -> LLM -> TTS."""
        t_turn_start = time.perf_counter()

        # Step 1: STT
        transcript, t_stt = self.transcribe_audio(audio_file_bytes, filename, language)

        # Step 2: LLM
        response_text, t_llm = self.generate_response(transcript, session_id, language)

        # Step 3: TTS
        audio_bytes, t_tts, audio_duration = self.synthesize_speech(response_text, language, voice=voice)

        t_total = time.perf_counter() - t_turn_start

        # Validate generated WAV before saving
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                wf_channels = wf.getnchannels()
                wf_rate = wf.getframerate()
        except Exception as e:
            raise RuntimeError(f"Generated audio is not a valid WAV file: {e}")

        # Save output audio file for HTTP retrieval
        timestamp = int(time.time() * 1000)
        provider_tag = self.tts_provider
        out_filename = f"reply_{session_id}_{language}_{provider_tag}_{timestamp}.wav"
        out_filepath = os.path.join(OUTPUT_DIR, out_filename)
        with open(out_filepath, "wb") as f:
            f.write(audio_bytes)

        resolved_voice = voice
        if not resolved_voice:
            if self.tts_provider == "edge" and self.edge_tts:
                resolved_voice = self.edge_tts.resolve_voice(None, language)
            elif self.tts_provider == "gemini":
                resolved_voice = "Kore" if language == "hi" else "Puck"
            else:
                resolved_voice = resolve_mithi_voice(None, language)

        return {
            "session_id": session_id,
            "language": language,
            "voice": resolved_voice,
            "tts_provider": provider_tag,
            "transcript": transcript,
            "response_text": response_text,
            "audio_filename": out_filename,
            "audio_filepath": out_filepath,
            "audio_duration_seconds": round(audio_duration, 2),
            "audio_metadata": {
                "channels": wf_channels,
                "sample_rate": wf_rate,
                "format": "WAV_16BIT_PCM",
            },
            "timings": {
                "stt_time_seconds": round(t_stt, 3),
                "llm_time_seconds": round(t_llm, 3),
                "tts_time_seconds": round(t_tts, 3),
                "total_time_seconds": round(t_total, 3),
            },
        }
