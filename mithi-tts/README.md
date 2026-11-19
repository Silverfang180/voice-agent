# Mithi v2 Flash TTS Docker Environment

This directory provides an isolated Linux Docker container running Python 3.12 to execute **Mithi v2 Flash** Text-to-Speech ([Oomni/mithi-v2-flash](https://huggingface.co/Oomni/mithi-v2-flash)) on CPU without dependency conflicts with Windows Python 3.14.

---

## Structure

* **`Dockerfile`**: Builds a lightweight Python 3.12-slim image, installs CPU PyTorch, Mithi dependencies (including `misaki 0.9.4` and `espeakng-loader`), and downloads the spaCy English model.
* **`requirements.txt`**: Standard Mithi inference dependencies.
* **`test_mithi_tts.py`**: Benchmark and test runner measuring model load time, synthesis time, audio duration, real-time factor, and RAM usage.
* **`model/`**: Persistent directory mounted into the container to store downloaded model weights.
* **`output/`**: Directory where generated 24 kHz mono WAV files are stored and accessible on Windows.

---

## Build the Docker Image

Run from the `mithi-tts` directory:

```bash
docker build -t mithi-v2-flash:latest .
```

---

## Run Inference & Benchmark Commands

### 1. Hindi Voice Test (Charvi - Female)
```bash
docker run --rm \
  -v "${PWD}/model:/app/model" \
  -v "${PWD}/output:/app/output" \
  -v "${PWD}/test_mithi_tts.py:/app/test_mithi_tts.py" \
  mithi-v2-flash:latest \
  python test_mithi_tts.py \
    --language hi \
    --voice charvi \
    --device cpu \
    --threads 4 \
    --text "नमस्ते! ABC Business Services में आपका स्वागत है। हम आपकी कैसे सहायता कर सकते हैं?" \
    --output /app/output/tts_output_hi_charvi.wav
```

### 2. Marathi Voice Test (Satvik - Male)
```bash
docker run --rm \
  -v "${PWD}/model:/app/model" \
  -v "${PWD}/output:/app/output" \
  -v "${PWD}/test_mithi_tts.py:/app/test_mithi_tts.py" \
  mithi-v2-flash:latest \
  python test_mithi_tts.py \
    --language mr \
    --voice satvik \
    --device cpu \
    --threads 4 \
    --text "नमस्कार! ABC Business Services मध्ये तुमचं स्वागत आहे. आम्ही तुमची कशी मदत करू शकतो?" \
    --output /app/output/tts_output_mr_satvik.wav
```

Both resulting WAV audio files will appear immediately in `mithi-tts/output/` on Windows.

