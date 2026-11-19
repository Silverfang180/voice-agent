import os
import sys
import time

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import soundfile as sf
import numpy as np

def run_test():
    print("=" * 65)
    print("MITHI V2 FLASH - GPU INFERENCE FEASIBILITY BENCHMARK")
    print("=" * 65)

    # 1. Inspect CUDA Environment
    print(f"PyTorch Version:      {torch.__version__}")
    print(f"CUDA Available:       {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available to PyTorch inside container!", file=sys.stderr)
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    total_vram_mb = device_props.total_memory / (1024 * 1024)
    print(f"GPU Device:           {device_name}")
    print(f"Compute Capability:   {device_props.major}.{device_props.minor}")
    print(f"Total VRAM:           {total_vram_mb:.2f} MB ({total_vram_mb/1024:.2f} GB)")
    print("-" * 65)

    # 2. Load Model on CUDA
    model_path = os.path.abspath("/app/model")
    sys.path.insert(0, model_path)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_before_load = torch.cuda.memory_allocated() / (1024 * 1024)

    print(f"Loading Mithi model onto device='cuda' from {model_path}...")
    t0 = time.perf_counter()
    from run import Mithi
    tts = Mithi(model_dir=model_path, device="cuda")
    t_load = time.perf_counter() - t0

    vram_after_load = torch.cuda.memory_allocated() / (1024 * 1024)
    vram_peak_load = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"Model Load Time:      {t_load:.3f} s")
    print(f"VRAM Allocated:       {vram_after_load:.2f} MB (Peak during load: {vram_peak_load:.2f} MB)")
    print("-" * 65)

    os.makedirs("/app/output", exist_ok=True)

    results = []

    # 3. Test Cases: Hindi and Marathi
    test_cases = [
        {
            "name": "Hindi Test",
            "lang": "hi",
            "voice": "charvi",
            "text": "नमस्ते! ABC Business Services में आपका स्वागत है। हम आपकी कैसे सहायता कर सकते हैं?",
            "output_path": "/app/output/gpu_test_hi.wav"
        },
        {
            "name": "Marathi Test",
            "lang": "mr",
            "voice": "satvik",
            "text": "नमस्कार! ABC Business Services मध्ये तुमचं स्वागत आहे. आम्ही तुमची कशी मदत करू शकतो?",
            "output_path": "/app/output/gpu_test_mr.wav"
        }
    ]

    for tc in test_cases:
        print(f"Starting {tc['name']} ({tc['lang']}, voice={tc['voice']})...")
        print(f"Text: '{tc['text']}'")

        torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        audio = tts.synthesize(tc['text'], lang=tc['lang'], voice=tc['voice'])
        t_synth = time.perf_counter() - t_start

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        reserved_vram_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

        # Save audio
        tts.save(audio, tc['output_path'])

        # Verify output audio file
        data, samplerate = sf.read(tc['output_path'])
        duration = len(data) / float(samplerate)
        rtf = t_synth / duration if duration > 0 else 0.0
        max_amplitude = float(np.max(np.abs(data)))

        res = {
            "name": tc["name"],
            "lang": tc["lang"],
            "voice": tc["voice"],
            "text": tc["text"],
            "synth_time_s": t_synth,
            "duration_s": duration,
            "rtf": rtf,
            "peak_vram_mb": peak_vram_mb,
            "reserved_vram_mb": reserved_vram_mb,
            "samplerate": samplerate,
            "output_path": tc["output_path"],
            "max_amplitude": max_amplitude
        }
        results.append(res)

        print(f"  Synthesis Time:     {t_synth:.3f} s")
        print(f"  Audio Duration:     {duration:.2f} s")
        print(f"  Real-Time Factor:   {rtf:.3f}x ({'FASTER' if rtf < 1.0 else 'SLOWER'} than real-time)")
        print(f"  Peak VRAM Used:     {peak_vram_mb:.2f} MB (Reserved: {reserved_vram_mb:.2f} MB)")
        print(f"  Max Amplitude:      {max_amplitude:.4f}")
        print(f"  Output:             {tc['output_path']}")
        print("-" * 65)

    # 4. Final Summary
    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY RESULTS (GPU: " + device_name + ")")
    print("=" * 65)
    print(f"Model Load Time:      {t_load:.3f} s")
    print(f"{'Language':<10} | {'Voice':<8} | {'Audio Dur':<10} | {'GPU Synth':<10} | {'RTF':<8} | {'Peak VRAM':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['lang']:<10} | {r['voice']:<8} | {r['duration_s']:<9.2f}s | {r['synth_time_s']:<9.3f}s | {r['rtf']:<7.3f}x | {r['peak_vram_mb']:<7.1f} MB")
    print("=" * 65)

if __name__ == "__main__":
    run_test()

