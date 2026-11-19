import argparse
import os
import sys
import time

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

LANGUAGE_PROMPTS = {
    "hi": "नमस्ते। हिंदी भाषा में बातचीत।",
    "mr": "नमस्कार। मराठी भाषेतील संभाषण।",
}


def main():
    parser = argparse.ArgumentParser(
        description="Test faster-whisper transcription for Hindi and Marathi audio."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file (e.g., sample_hi.wav, sample_mr.mp3)."
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["hi", "mr"],
        default=None,
        help="Target language code: 'hi' (Hindi) or 'mr' (Marathi). If omitted, language is detected automatically."
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable the language-specific initial_prompt even if a language is specified."
    )
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio_path)
    if not os.path.isfile(audio_path):
        print(f"Error: Audio file not found at '{args.audio_path}' (resolved: '{audio_path}').", file=sys.stderr)
        sys.exit(1)

    # Determine initial_prompt
    if args.no_prompt or not args.language:
        initial_prompt = None
    else:
        initial_prompt = LANGUAGE_PROMPTS.get(args.language)

    prompt_display = f"Enabled ('{initial_prompt}')" if initial_prompt else "Disabled"

    print(f"Loading Whisper 'base' model on CPU (compute_type='int8')...")
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        print(f"Error: Failed to import faster-whisper: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
    except Exception as e:
        print(f"Error initializing WhisperModel: {e}", file=sys.stderr)
        sys.exit(1)

    language_mode = args.language if args.language else "Auto-detect"
    print(f"Transcribing '{args.audio_path}' (Language: {language_mode}, Initial Prompt: {prompt_display})...")

    start_time = time.perf_counter()
    try:
        segments, info = model.transcribe(
            audio_path,
            language=args.language,
            initial_prompt=initial_prompt,
            beam_size=5
        )
        # Segments is a generator; materialize to complete transcription
        segment_list = list(segments)
        processing_time = time.perf_counter() - start_time
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)

    transcription = " ".join(seg.text.strip() for seg in segment_list).strip()

    print("\n" + "=" * 50)
    print("WHISPER TRANSCRIPTION RESULT")
    print("=" * 50)
    print(f"Audio File:          {args.audio_path}")
    print(f"Audio Duration:      {info.duration:.2f} s")
    print(f"Target Language:     {args.language if args.language else 'Auto-detect'}")
    print(f"Detected Language:   {info.language}")
    print(f"Language Confidence: {info.language_probability * 100:.2f}%")
    print(f"Initial Prompt:      {prompt_display}")
    print(f"Processing Time:     {processing_time:.2f} s")
    if info.duration > 0:
        rtf = processing_time / info.duration
        print(f"Real-Time Factor:    {rtf:.2f}x ({'faster' if rtf < 1.0 else 'slower'} than real-time)")
    print("-" * 50)
    print(f"Transcription:\n{transcription if transcription else '[No speech detected]'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
