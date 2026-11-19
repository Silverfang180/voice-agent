# Multilingual AI Phone Receptionist & Voice Agent

A production-ready multilingual AI phone receptionist built for **ABC Business Services**, supporting **Hindi** and **Marathi** inbound calls via **LiveKit SIP**, powered by **Groq Whisper** (STT), **Groq Qwen 27B** (LLM), and high-fidelity Neural TTS (**Microsoft Edge TTS** / **Google Gemini 3.8 Flash-Lite** / **Mithi v2 Flash**).

---

## 🚀 Key Features

- **LiveKit Cloud SIP Integration**: Registers as worker agent `abc-receptionist` to directly answer inbound phone/SIP calls from trunks and SIP clients (e.g., MicroSIP, Asterisk, Twilio, Telnyx).
- **Sub-Second Intelligence**: Groq-hosted `whisper-large-v3` for speech transcription (~350ms–700ms) and `qwen/qwen3.8-27b` for grounded context reasoning (~300ms–600ms).
- **Multilingual (Hindi & Marathi)**: Automatically detects and converses in Hindi and Marathi with native Devanagari script processing.
- **Echo Guard & Barge-in Protection**: Robust half-duplex state management with echo guard interval to prevent self-interruption from speaker echo.
- **Modular Multi-TTS Support**:
  - **Microsoft Edge Neural TTS** (Default, zero daily quotas, fast ~1.5s–2.0s streaming response).
  - **Google Gemini 3.8 Flash-Lite TTS** (Expressive voice synthesis with automatic backoff).
  - **Mithi v2 Flash TTS** (Optional local Docker CPU/GPU microservice for Indic speech).
- **Grounded Business FAQ**: Dynamic JSON business profile (`business_faq.json`) ensures no hallucinated answers, strictly adhering to business hours, location, and services.
- **Dual Runtime Interface**: Run as an autonomous LiveKit SIP Agent worker or as a FastAPI REST microservice.

---

## 🏗️ Architecture

```
                                      [ Inbound Phone Call (SIP) ]
                                                   │
                                                   ▼
                                         [ LiveKit Cloud Room ]
                                                   │
               ┌───────────────────────────────────┴───────────────────────────────────┐
               │                                                                       │
        [ Caller Audio Stream ]                                              [ Published Agent Audio ]
               │                                                                       ▲
               ▼                                                                       │
     Silero Neural VAD                                                        PyAV Audio Decoding
     (Turn Detection)                                                        (24 kHz Mono 16-bit PCM)
               │                                                                       │
               ▼                                                                       │
     1. Groq Whisper Large-v3 STT  ──────► Devanagari Transcript                       │
               │                                                                       │
               ▼                                                                       │
     2. Groq Qwen 27B LLM          ──────► Grounded 1-2 Sentence Reply                 │
               │                                                                       │
               ▼                                                                       │
     3. Neural TTS Provider        ──────► Edge TTS / Gemini / Mithi ──────────────────┘
```

---

## 📁 Repository Structure

```text
├── livekit_phone_agent.py      # Realtime LiveKit SIP agent worker (abc-receptionist)
├── voice_pipeline.py           # Unified STT -> LLM -> TTS orchestration pipeline
├── groq_llm.py                 # Grounded AI Receptionist Brain with conversational memory
├── edge_tts_provider.py        # High-performance Edge Neural TTS provider
├── gemini_tts.py               # Google Gemini 3.8 Flash-Lite TTS provider
├── business_faq.json           # Fictional business profile, hours, services, and contacts
├── main.py                     # FastAPI REST API backend
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment configuration
├── samples/                    # Audio samples for unit testing (Hindi & Marathi)
├── mithi-tts/                  # Docker container definition for local Mithi TTS
│   ├── Dockerfile
│   ├── server.py
│   └── requirements.txt
└── test_*.py                   # Automated test scripts for STT, LLM, TTS & Pipeline
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10 to 3.12 (or Python 3.14 with pre-compiled wheels)
- LiveKit Cloud account & project
- Groq Cloud API key
- (Optional) Google Gemini API key

### 2. Environment Setup
Clone the repository and install the dependencies:

```bash
git clone https://github.com/Silverfang180/voice-agent.git
cd voice-agent

python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Credentials
Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=gsk_...
GROQ_LLM_MODEL=qwen/qwen3.8-27b
GROQ_STT_MODEL=whisper-large-v3

LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...

TTS_PROVIDER=edge
```

---

## 📞 Running the LiveKit SIP Phone Agent

To connect the agent to LiveKit Cloud and answer calls:

```powershell
python -m livekit.agents start --log-format colored livekit_phone_agent.py
```

The agent registers with name `abc-receptionist`. When an incoming call arrives through your SIP dispatch rule, the agent automatically:
1. Greets the caller in Hindi/Marathi.
2. Listens for speech turns using Silero VAD.
3. Transcribes, reasons, and speaks back naturally in the caller's language.

---

## 🌐 Running the FastAPI REST Backend

For web-based or HTTP integration:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- **Health Check**: `GET /health`
- **Voice Turn**: `POST /api/voice/turn` (Accepts `multipart/form-data` with `audio`, `language`, `session_id`)
- **Audio Retrieval**: `GET /api/voice/audio/{filename}`

---

## 🧪 Testing

Run test suites individually:

```powershell
# Test Groq STT
python test_groq_stt.py

# Test LLM Reasoning & Grounding
python test_groq_llm.py

# Test Edge Neural TTS
python test_edge_tts.py

# Test Full End-to-End Voice Pipeline
python test_voice_pipeline.py
```

---

## 🔒 Security & Privacy

- All sensitive keys (`.env`) are strictly ignored by `.gitignore`.
- Conversation state is scoped to ephemeral session IDs and retained in-memory only.
- Audio synthesis temp buffers are automatically managed.
