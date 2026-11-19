"""Wrapper to execute Mithi v2 Flash TTS in Docker container from Windows."""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run Mithi v2 Flash TTS inference inside Docker (Python 3.12, CPU)."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize."
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["hi", "mr", "en"],
        default="hi",
        help="Language code: 'hi' (Hindi), 'mr' (Marathi), 'en' (English)."
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="charvi",
        help="Voice name: 'charvi' (female), 'satvik' (male), 'vaidehi', 'ahana', 'anirban', etc."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads (default: 4)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output WAV path (default: 'tts_output_<lang>_<voice>.wav')."
    )

    args = parser.parse_args()

    # Determine output filename
    if args.output:
        out_name = os.path.basename(args.output)
    else:
        out_name = f"tts_output_{args.language}_{args.voice}.wav"

    project_dir = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(project_dir, "mithi-tts", "model")
    script_path = os.path.join(project_dir, "mithi-tts", "test_mithi_tts.py")

    container_output = f"/app/voiceagent/{out_name}"

    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{model_dir}:/app/model",
        "-v", f"{project_dir}:/app/voiceagent",
        "-v", f"{script_path}:/app/test_mithi_tts.py",
        "mithi-v2-flash:latest",
        "python", "/app/test_mithi_tts.py",
        "--language", args.language,
        "--voice", args.voice,
        "--device", "cpu",
        "--threads", str(args.threads),
        "--text", args.text,
        "--output", container_output
    ]

    print(f"Executing Mithi TTS in Docker container...")
    print(f"Language: {args.language} | Voice: {args.voice} | Threads: {args.threads}")
    print(f"Output:   {out_name}\n")

    result = subprocess.run(docker_cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

