import argparse
import os
import sys
import time
import tracemalloc

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Test Mithi v2 Flash TTS in Docker container.")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize.")
    parser.add_argument("--language", type=str, choices=["hi", "mr", "en"], required=True, help="Language code.")
    parser.add_argument("--voice", type=str, required=True, help="Voice name (e.g. charvi, satvik).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu"], help="Inference device (cpu).")
    parser.add_argument("--threads", type=int, default=4, help="CPU thread count (default: 4).")
    parser.add_argument("--model-dir", type=str, default="/app/model", help="Directory where model repository resides.")
    parser.add_argument("--output", type=str, default="/app/output/output.wav", help="Path to save output WAV.")
    args = parser.parse_args()

    # Track memory
    tracemalloc.start()

    import torch
    torch.set_num_threads(args.threads)

    # Check model directory
    model_path = os.path.abspath(args.model_dir)
    sys.path.insert(0, model_path)

    # Check if model files are downloaded in model_dir
    run_py = os.path.join(model_path, "run.py")
    if not os.path.isfile(run_py):
        print(f"Model files not found in '{model_path}'. Downloading 'Oomni/mithi-v2-flash'...")
        t_dl_start = time.perf_counter()
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="Oomni/mithi-v2-flash", local_dir=model_path)
        t_dl = time.perf_counter() - t_dl_start
        print(f"Model downloaded in {t_dl:.2f} s.")
    else:
        print(f"Using cached model files from '{model_path}'.")

    # Load model
    print(f"Loading Mithi model (device={args.device}, threads={args.threads})...")
    t_load_start = time.perf_counter()
    try:
        from run import Mithi
        tts = Mithi(model_dir=model_path, device=args.device)
    except Exception as e:
        print(f"Error loading Mithi model: {e}", file=sys.stderr)
        sys.exit(1)
    t_load = time.perf_counter() - t_load_start

    # Generate speech
    print(f"Generating speech (lang={args.language}, voice={args.voice})...")
    t_synth_start = time.perf_counter()
    try:
        audio = tts.synthesize(args.text, lang=args.language, voice=args.voice)
    except Exception as e:
        print(f"Error during speech synthesis: {e}", file=sys.stderr)
        sys.exit(1)
    t_synth = time.perf_counter() - t_synth_start

    # Save audio
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    try:
        tts.save(audio, args.output)
    except Exception as e:
        print(f"Error saving audio file: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate audio duration
    import soundfile as sf
    try:
        data, samplerate = sf.read(args.output)
        duration = len(data) / float(samplerate)
    except Exception:
        duration = 0.0

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rtf = t_synth / duration if duration > 0 else 0.0

    print("\n" + "=" * 55)
    print("MITHI V2 FLASH TTS BENCHMARK")
    print("=" * 55)
    print(f"Text:             {args.text}")
    print(f"Language:         {args.language}")
    print(f"Voice:            {args.voice}")
    print(f"Device:           {args.device} (threads={args.threads})")
    print(f"Output File:      {args.output}")
    print(f"Sample Rate:      24000 Hz (mono)")
    print("-" * 55)
    print(f"Model Load Time:  {t_load:.2f} s")
    print(f"Synthesis Time:   {t_synth:.2f} s")
    print(f"Audio Duration:   {duration:.2f} s")
    print(f"Real-Time Factor: {rtf:.2f}x ({'faster' if rtf < 1.0 else 'slower'} than real-time)")
    print(f"Peak RAM Usage:   {peak_mem / (1024 * 1024):.2f} MB")
    print("=" * 55)


if __name__ == "__main__":
    main()

