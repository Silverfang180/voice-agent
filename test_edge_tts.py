"""Verification and latency benchmark for Edge TTS Hindi and Marathi voices."""
import asyncio
import os
import sys
import time
import wave

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from edge_tts_provider import EdgeTTSProvider

OUTPUT_DIR = os.path.abspath("output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


async def run_tests():
    print("=" * 72)
    print("EDGE TTS VOICE AVAILABILITY, MP3->WAV DECODING & LATENCY BENCHMARK")
    print("=" * 72)

    provider = EdgeTTSProvider()

    # 1. Verify target voices via official edge_tts.list_voices()
    voices = await provider.list_indian_voices()
    target_names = {
        "hi-IN-SwaraNeural",
        "hi-IN-MadhurNeural",
        "mr-IN-AarohiNeural",
        "mr-IN-ManoharNeural",
    }
    found_names = {v["ShortName"] for v in voices}
    print(f"Indian (hi-IN / mr-IN) voices reported by Edge TTS: {len(voices)}")
    for v in voices:
        mark = "[TARGET]" if v["ShortName"] in target_names else "        "
        print(f"  {mark} {v['ShortName']:<22} | {v['Locale']} | {v['Gender']}")

    missing = target_names - found_names
    assert not missing, f"Missing target voices: {missing}"
    print("All 4 required Hindi & Marathi voices VERIFIED available.\n")

    # 2. Test synthesis + MP3->24kHz WAV conversion for all 4 voices
    cases = [
        (
            "Hindi (Female)",
            "hi",
            "hi-IN-SwaraNeural",
            "नमस्ते! ABC Business Services में आपका स्वागत है। हम आपकी कैसे सहायता कर सकते हैं?",
            "edge_test_hi_swara.wav",
        ),
        (
            "Hindi (Male)",
            "hi",
            "hi-IN-MadhurNeural",
            "हमारा कार्यालय सोमवार से शुक्रवार सुबह नौ बजे से शाम छह बजे तक खुला रहता है।",
            "edge_test_hi_madhur.wav",
        ),
        (
            "Marathi (Female)",
            "mr",
            "mr-IN-AarohiNeural",
            "नमस्कार! ABC Business Services मध्ये तुमचं स्वागत आहे. आम्ही तुमची कशी मदत करू शकतो?",
            "edge_test_mr_aarohi.wav",
        ),
        (
            "Marathi (Male)",
            "mr",
            "mr-IN-ManoharNeural",
            "आमचे कार्यालय सोमवार ते शुक्रवार सकाळी नऊ ते संध्याकाळी सहा वाजेपर्यंत उघडे असते.",
            "edge_test_mr_manohar.wav",
        ),
    ]

    print(f"{'Case':<18} | {'Voice':<20} | {'Net (s)':<8} | {'Dec (s)':<8} | {'Total':<7} | {'Dur (s)':<7} | {'WAV Spec'}")
    print("-" * 92)

    for label, lang, voice, text, filename in cases:
        wav_bytes, total_t, dur = await provider.synthesize_async(text, language=lang, voice=voice)
        net_t = provider.last_synth_network_time
        dec_t = provider.last_decode_time

        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "wb") as f:
            f.write(wav_bytes)

        # Verify WAV container and PCM metadata
        assert wav_bytes[:4] == b"RIFF", "Not a valid RIFF WAV!"
        with wave.open(out_path, "rb") as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            nf = wf.getnframes()
            assert ch == 1, f"Expected mono, got {ch}"
            assert sw == 2, f"Expected 16-bit PCM, got {sw * 8}-bit"
            assert sr == 24000, f"Expected 24000 Hz, got {sr}"
            assert nf > 0, "Zero audio frames!"

        print(
            f"{label:<18} | {voice:<20} | {net_t:<8.3f} | {dec_t:<8.3f} | {total_t:<7.3f} | {dur:<7.2f} | {sr}Hz/{ch}ch/{sw*8}bit"
        )

    print("-" * 92)
    print("ALL EDGE TTS TESTS PASSED (Valid 24kHz 16-bit Mono PCM WAV)!")


if __name__ == "__main__":
    asyncio.run(run_tests())

