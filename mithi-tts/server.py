"""Persistent HTTP TTS microservice for Mithi v2 Flash running in Docker."""
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import soundfile as sf
import torch

# Ensure model directory is in sys.path
MODEL_DIR = os.environ.get("MITHI_MODEL_DIR", "/app/model")
sys.path.insert(0, MODEL_DIR)

print(f"Initializing persistent Mithi TTS service...")
print(f"Loading weights from '{MODEL_DIR}' on CPU...")
t0 = time.perf_counter()
from run import Mithi
tts_engine = Mithi(model_dir=MODEL_DIR, device="cpu")
print(f"Mithi TTS engine loaded in {time.perf_counter() - t0:.2f} s. Ready for requests.")


class MithiRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Concise logging
        sys.stdout.write(f"[Mithi-TTS] {args[0]} {args[1]} - {args[2]}\n")

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = json.dumps({"status": "healthy", "model": "mithi-v2-flash", "device": "cpu"}).encode("utf-8")
            self.wfile.write(resp)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/synthesize":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                req_data = json.loads(body)

                text = req_data.get("text", "")
                lang = req_data.get("language", "hi")
                voice = req_data.get("voice", "charvi")
                threads = int(req_data.get("threads", 4))

                if not text:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "text is required"}).encode("utf-8"))
                    return

                torch.set_num_threads(threads)

                t_start = time.perf_counter()
                audio = tts_engine.synthesize(text, lang=lang, voice=voice)
                synth_time = time.perf_counter() - t_start

                # Encode to WAV buffer
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, audio, 24000, format="WAV")
                wav_bytes = wav_buffer.getvalue()
                duration = len(audio) / 24000.0

                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav_bytes)))
                self.send_header("X-Synthesis-Time", f"{synth_time:.3f}")
                self.send_header("X-Audio-Duration", f"{duration:.3f}")
                self.end_headers()
                self.wfile.write(wav_bytes)

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port=5002):
    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, MithiRequestHandler)
    print(f"Mithi TTS server listening on http://0.0.0.0:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("MITHI_PORT", 5002))
    run_server(port)

