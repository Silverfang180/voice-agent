"""FastAPI Unified Voice Backend for ABC Business Services AI Voice Agent."""
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.voice_pipeline import OUTPUT_DIR, VoicePipeline

app = FastAPI(
    title="Multilingual AI Receptionist Voice Backend",
    description="Unified backend connecting Groq Whisper STT, Groq Receptionist Brain, and Mithi v2 Flash TTS.",
    version="1.0.0"
)

# Enable CORS for future frontend/web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[VoicePipeline] = None


@app.on_event("startup")
def startup_event():
    global pipeline
    print("Initializing VoicePipeline...")
    pipeline = VoicePipeline()
    print("VoicePipeline ready.")


@app.get("/health")
def health_check():
    """Health status reporting backend, Groq, Edge TTS, Gemini, and Mithi status without exposing secrets."""
    groq_api_key_set = bool(os.environ.get("GROQ_API_KEY", "").strip())
    gemini_api_key_set = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    groq_model = os.environ.get("GROQ_LLM_MODEL", "qwen/qwen3.8-27b")

    tts_provider = getattr(pipeline, "tts_provider", "edge") if pipeline else "uninitialized"
    mithi_status = pipeline.check_mithi_health() if pipeline else {"status": "uninitialized"}

    gemini_model = "not_configured"
    if pipeline and pipeline.gemini_tts:
        gemini_model = pipeline.gemini_tts.model_name

    edge_status = {"configured": False}
    if pipeline and pipeline.edge_tts:
        edge_status = {
            "configured": True,
            "voice_hi": pipeline.edge_tts.hi_voice,
            "voice_mr": pipeline.edge_tts.mr_voice,
        }

    return {
        "status": "healthy",
        "backend": "FastAPI",
        "tts_provider": tts_provider,
        "groq_status": {
            "api_key_configured": groq_api_key_set,
            "llm_model": groq_model,
            "stt_model": "whisper-large-v3-turbo"
        },
        "edge_tts_status": edge_status,
        "gemini_status": {
            "api_key_configured": gemini_api_key_set,
            "tts_model": gemini_model
        },
        "mithi_service": mithi_status,
        "supported_languages": ["hi", "mr"]
    }


@app.post("/api/voice/turn")
async def voice_turn(
    audio: UploadFile = File(..., description="User voice recording (.wav, .mp3, etc.)"),
    language: str = Form("hi", description="Language code ('hi' or 'mr')"),
    session_id: str = Form("default", description="Conversation session ID for memory isolation"),
    voice: Optional[str] = Form(None, description="Optional voice name ('charvi', 'satvik')")
):
    """Execute complete voice turn: STT -> AI Receptionist Brain -> TTS."""
    if language not in ["hi", "mr"]:
        raise HTTPException(status_code=400, detail="Language must be 'hi' (Hindi) or 'mr' (Marathi).")

    if not pipeline:
        raise HTTPException(status_code=503, detail="Voice pipeline is not initialized.")

    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        turn_result = pipeline.process_voice_turn(
            audio_file_bytes=audio_bytes,
            filename=audio.filename or "input.wav",
            language=language,
            session_id=session_id,
            voice=voice
        )

        # Include direct audio download URL
        turn_result["audio_url"] = f"/api/voice/audio/{turn_result['audio_filename']}"

        return JSONResponse(status_code=200, content=turn_result)

    except Exception as e:
        print(f"Error during voice turn processing: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voice/audio/{filename}")
def get_generated_audio(filename: str):
    """Retrieve generated WAV audio file for playback."""
    file_path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)


@app.post("/api/voice/session/reset")
def reset_session(session_id: str = Form("default")):
    """Reset conversation memory for a given session ID."""
    if pipeline:
        pipeline.brain.reset_session(session_id)
    return {"status": "success", "session_id": session_id, "message": "Session history cleared."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

