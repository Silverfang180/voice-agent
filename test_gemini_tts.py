"""Benchmark and validation test for Gemini TTS provider in Hindi and Marathi."""
import os
import sys
import time
import wave

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from gemini_tts import GeminiTTSProvider

OUTPUT_DIR = os.path.abspath("output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_gemini_tts():
    print("=" * 65)
    print("GEMINI TTS BENCHMARK & METADATA VERIFICATION")
    print("=" * 65)

    provider = GeminiTTSProvider(model_name="gemini-3.8-flash-lite-tts")
    print(f"Model: {provider.model_name}")
    print(f"API Key Configured: {bool(provider.api_key)}")
    print("-" * 65)

    test_cases = [
        {
            "name": "Hindi Test",
            "lang": "hi",
            "voice": "Kore",
            "text": "नमस्ते! ABC Business Services में आपका स्वागत है। हम आपकी कैसे सहायता कर सकते हैं?",
            "output_file": os.path.join(OUTPUT_DIR, "gemini_sample_hi.wav")
        },
        {
            "name": "Marathi Test",
            "lang": "mr",
            "voice": "Puck",
            "text": "नमस्कार! ABC Business Services मध्ये तुमचं स्वागत आहे. आम्ही तुमची कशी मदत करू शकतो?",
            "output_file": os.path.join(OUTPUT_DIR, "gemini_sample_mr.wav")
        }
    ]

    results = []

    for tc in test_cases:
        print(f"\nRunning {tc['name']} (Language: {tc['lang']}, Voice: {tc['voice']})...")
        print(f"Text: '{tc['text']}'")

        t0 = time.perf_counter()
        audio_bytes, t_api, duration = provider.synthesize(
            text=tc["text"],
            language=tc["lang"],
            voice=tc["voice"]
        )
        t_total = time.perf_counter() - t0

        # Save audio file
        with open(tc["output_file"], "wb") as f:
            f.write(audio_bytes)

        # Inspect WAV metadata with wave module
        with wave.open(tc["output_file"], "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            calc_dur = nframes / float(framerate)

        rtf = t_api / duration if duration > 0 else 0.0

        res = {
            "name": tc["name"],
            "lang": tc["lang"],
            "voice": tc["voice"],
            "api_latency_s": t_api,
            "total_time_s": t_total,
            "audio_duration_s": duration,
            "rtf": rtf,
            "file_size_bytes": len(audio_bytes),
            "channels": channels,
            "framerate": framerate,
            "sampwidth": sampwidth,
            "output_file": tc["output_file"]
        }
        results.append(res)

        print(f"  API Latency:        {t_api:.3f} s")
        print(f"  Total Time:         {t_total:.3f} s")
        print(f"  Audio Duration:     {duration:.2f} s")
        print(f"  Real-Time Factor:   {rtf:.3f}x ({'Faster' if rtf < 1.0 else 'Slower'} than real-time)")
        print(f"  WAV Metadata:       Channels={channels}, Rate={framerate} Hz, Bit Depth={sampwidth * 8}-bit")
        print(f"  File Size:          {len(audio_bytes):,} bytes")
        print(f"  Saved To:           {tc['output_file']}")
        print(f"  Status:             PASSED (Valid WAV)")

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY (gemini-3.8-flash-lite-tts)")
    print("=" * 65)
    print(f"{'Language':<10} | {'Voice':<6} | {'Audio Dur':<10} | {'API Latency':<12} | {'RTF':<8} | {'WAV Spec':<15}")
    print("-" * 65)
    for r in results:
        print(f"{r['lang']:<10} | {r['voice']:<6} | {r['audio_duration_s']:<9.2f}s | {r['api_latency_s']:<11.3f}s | {r['rtf']:<7.3f}x | {r['framerate']}Hz/{r['channels']}ch")
    print("=" * 65)


if __name__ == "__main__":
    test_gemini_tts()

