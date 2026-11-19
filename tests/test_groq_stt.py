import argparse
import os
import sys
import time
import wave
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_env_file():
    """Load environment variables from project's .env file using python-dotenv with fallback."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        env_path = PROJECT_ROOT / ".env"
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("\"'")
                        if key and key not in os.environ:
                            os.environ[key] = val
            except Exception:
                pass


def get_audio_duration_seconds(file_path: str) -> float:
    """Attempt to calculate audio duration from WAV header, returning 0.0 on fallback."""
    try:
        with wave.open(file_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Hindi and Marathi audio using Groq hosted Whisper STT."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio recording (e.g., sample_hi.wav, sample_mr.wav)."
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["hi", "mr"],
        required=True,
        help="Target language code: 'hi' (Hindi) or 'mr' (Marathi)."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["whisper-large-v3", "whisper-large-v3-turbo"],
        default="whisper-large-v3",
        help="Groq Whisper model to use (default: 'whisper-large-v3')."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the transcript text file. Defaults to 'transcript_<basename>_<model>_<lang>.txt'."
    )
    args = parser.parse_args()

    # Verify audio file existence
    audio_path = os.path.abspath(args.audio_path)
    if not os.path.isfile(audio_path):
        print(f"Error: Audio file not found at '{args.audio_path}' (resolved: '{audio_path}').", file=sys.stderr)
        sys.exit(1)

    # Check for API Key in environment or .env
    load_env_file()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        print(
            "Error: GROQ_API_KEY environment variable is not set.\n"
            "Please set your Groq API key in your environment:\n"
            "  PowerShell: $env:GROQ_API_KEY = \"your_api_key_here\"\n"
            "  or create a .env file with: GROQ_API_KEY=your_api_key_here",
            file=sys.stderr
        )
        sys.exit(1)

    # Import Groq SDK
    try:
        from groq import Groq, AuthenticationError, RateLimitError, APIConnectionError, APIStatusError
    except ImportError:
        print("Error: The 'groq' Python package is not installed. Run: pip install groq", file=sys.stderr)
        sys.exit(1)

    # Calculate audio duration locally
    audio_duration = get_audio_duration_seconds(audio_path)

    # Initialize client (never print or expose the API key)
    client = Groq(api_key=api_key.strip())

    print(f"Connecting to Groq Speech-to-Text API...")
    print(f"Audio file: {args.audio_path}")
    print(f"Model:      {args.model}")
    print(f"Language:   {args.language}")

    start_time = time.perf_counter()
    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_file.read()),
                model=args.model,
                language=args.language,
                response_format="verbose_json"
            )
        api_duration = time.perf_counter() - start_time
    except AuthenticationError:
        print("Error: Authentication failed. Please verify that your GROQ_API_KEY is valid.", file=sys.stderr)
        sys.exit(1)
    except RateLimitError as e:
        print(f"Error: Groq rate limit exceeded: {e}", file=sys.stderr)
        sys.exit(1)
    except APIConnectionError as e:
        print(f"Error: Network connection failed while reaching Groq API: {e}", file=sys.stderr)
        sys.exit(1)
    except APIStatusError as e:
        print(f"Error: Groq API returned status {e.status_code}: {e.message}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during transcription request: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse response
    transcript_text = getattr(response, "text", str(response)).strip()
    reported_duration = getattr(response, "duration", None)
    if reported_duration and reported_duration > 0:
        audio_duration = reported_duration

    # Output file path
    if args.output:
        out_file_path = args.output
    else:
        stem = Path(audio_path).stem
        sanitized_model = args.model.replace("-", "_")
        transcripts_dir = PROJECT_ROOT / "transcripts"
        os.makedirs(transcripts_dir, exist_ok=True)
        out_file_path = transcripts_dir / f"transcript_{stem}_{sanitized_model}_{args.language}.txt"

    # Save transcript as UTF-8
    try:
        with open(out_file_path, "w", encoding="utf-8") as out_f:
            out_f.write(transcript_text + "\n")
        saved_status = f"Saved to '{out_file_path}'"
    except Exception as e:
        saved_status = f"Failed to save file: {e}"

    print("\n" + "=" * 55)
    print("GROQ WHISPER STT RESULT")
    print("=" * 55)
    print(f"Audio File:        {args.audio_path}")
    if audio_duration > 0:
        print(f"Audio Duration:    {audio_duration:.2f} s")
    print(f"Model:             {args.model}")
    print(f"Language:          {args.language}")
    print(f"API Request Time:  {api_duration:.2f} s")
    if audio_duration > 0:
        speedup = audio_duration / api_duration if api_duration > 0 else 0
        print(f"Processing Speed:  {speedup:.2f}x real-time")
    print(f"Transcript File:   {saved_status}")
    print("-" * 55)
    print(f"Transcript:\n{transcript_text if transcript_text else '[No speech transcribed]'}")
    print("=" * 55)


if __name__ == "__main__":
    main()

