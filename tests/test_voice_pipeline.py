"""End-to-End Test for Unified AI Voice Pipeline."""
import os
import sys
import time
import wave

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from main import app


def test_pipeline():
    print("=" * 65)
    print("UNIFIED AI VOICE PIPELINE END-TO-END TEST")
    print("=" * 65)

    with TestClient(app) as client:
        # 1. Health check
        print("\n--- 1. HEALTH CHECK ---")
        health = client.get("/health").json()
        print("Backend Status:", health.get("status"))
        print("Groq STT / LLM:", health.get("groq_status"))
        print("Mithi TTS Service:", health.get("mithi_service"))
        assert health.get("status") == "healthy"
        assert health.get("groq_status", {}).get("api_key_configured") is True
        assert health.get("mithi_service", {}).get("status") in ("online", "offline")
        print("Health Check: PASSED")

        # 2. Hindi Voice Turn
        print("\n--- 2. HINDI VOICE TURN TEST ---")
        hi_audio_path = os.path.join(PROJECT_ROOT, "samples", "sample_hi.wav")
        assert os.path.isfile(hi_audio_path), f"Missing {hi_audio_path}"

        with open(hi_audio_path, "rb") as f:
            files = {"audio": ("sample_hi.wav", f, "audio/wav")}
            data = {"language": "hi", "session_id": "session_hi_test", "voice": "charvi"}
            t0 = time.perf_counter()
            resp_hi = client.post("/api/voice/turn", files=files, data=data)
            t_wall_hi = time.perf_counter() - t0

        assert resp_hi.status_code == 200, f"Error: {resp_hi.text}"
        res_hi = resp_hi.json()
        print(f"Status Code:       {resp_hi.status_code}")
        print(f"Session ID:        {res_hi.get('session_id')}")
        print(f"Input Audio File:  sample_hi.wav")
        print(f"STT Transcript:    {res_hi.get('transcript')}")
        print(f"Receptionist Text: {res_hi.get('response_text')}")
        print(f"Generated Audio:   {res_hi.get('audio_filename')}")
        print(f"Audio Duration:    {res_hi.get('audio_duration_seconds')} s")
        print(f"STT Time:          {res_hi['timings']['stt_time_seconds']} s")
        print(f"LLM Time:          {res_hi['timings']['llm_time_seconds']} s")
        print(f"TTS Time:          {res_hi['timings']['tts_time_seconds']} s")
        print(f"Total Pipeline:    {res_hi['timings']['total_time_seconds']} s (Wall: {t_wall_hi:.2f} s)")

        # Verify generated audio file validity
        gen_hi_path = res_hi.get("audio_filepath")
        assert os.path.isfile(gen_hi_path), f"Audio file not saved at {gen_hi_path}"
        with wave.open(gen_hi_path, "rb") as wf:
            print(f"WAV Check:         Channels={wf.getnchannels()}, Rate={wf.getframerate()} Hz, Frames={wf.getnframes()}")
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 24000

        # 3. Marathi Voice Turn
        print("\n--- 3. MARATHI VOICE TURN TEST ---")
        mr_audio_path = os.path.join(PROJECT_ROOT, "samples", "sample_mr.wav")
        assert os.path.isfile(mr_audio_path), f"Missing {mr_audio_path}"

        with open(mr_audio_path, "rb") as f:
            files = {"audio": ("sample_mr.wav", f, "audio/wav")}
            data = {"language": "mr", "session_id": "session_mr_test", "voice": "satvik"}
            t0 = time.perf_counter()
            resp_mr = client.post("/api/voice/turn", files=files, data=data)
            t_wall_mr = time.perf_counter() - t0

        assert resp_mr.status_code == 200, f"Error: {resp_mr.text}"
        res_mr = resp_mr.json()
        print(f"Status Code:       {resp_mr.status_code}")
        print(f"Session ID:        {res_mr.get('session_id')}")
        print(f"Input Audio File:  sample_mr.wav")
        print(f"STT Transcript:    {res_mr.get('transcript')}")
        print(f"Receptionist Text: {res_mr.get('response_text')}")
        print(f"Generated Audio:   {res_mr.get('audio_filename')}")
        print(f"Audio Duration:    {res_mr.get('audio_duration_seconds')} s")
        print(f"STT Time:          {res_mr['timings']['stt_time_seconds']} s")
        print(f"LLM Time:          {res_mr['timings']['llm_time_seconds']} s")
        print(f"TTS Time:          {res_mr['timings']['tts_time_seconds']} s")
        print(f"Total Pipeline:    {res_mr['timings']['total_time_seconds']} s (Wall: {t_wall_mr:.2f} s)")

        gen_mr_path = res_mr.get("audio_filepath")
        assert os.path.isfile(gen_mr_path), f"Audio file not saved at {gen_mr_path}"
        with wave.open(gen_mr_path, "rb") as wf:
            print(f"WAV Check:         Channels={wf.getnchannels()}, Rate={wf.getframerate()} Hz, Frames={wf.getnframes()}")
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 24000

        # 4. Audio Download Endpoint Verification
        print("\n--- 4. AUDIO RETRIEVAL ENDPOINT TEST ---")
        audio_dl_resp = client.get(res_hi.get("audio_url"))
        assert audio_dl_resp.status_code == 200
        assert audio_dl_resp.headers.get("content-type") == "audio/wav"
        print(f"Retrieved: {res_hi.get('audio_url')} -> {len(audio_dl_resp.content)} bytes (audio/wav)")

        print("\n" + "=" * 65)
        print("ALL PIPELINE TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 65)


if __name__ == "__main__":
    test_pipeline()

