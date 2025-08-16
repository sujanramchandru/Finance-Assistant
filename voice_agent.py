# stt_agent.py

import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import tempfile
import numpy as np
import uvicorn
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# CORS for allowing requests from orchestrator/frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific IP/port in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model (choose base/small for speed)
model = WhisperModel("base", device="cpu")

# Audio config
SAMPLE_RATE = 16000
DURATION = 7  # seconds

# Orchestrator endpoint to POST results to
ORCHESTRATOR_URL = "http://localhost:8000/receive_transcription"

@app.get("/stt")
def record_and_transcribe():
    try:
        print("🎙️ Listening from mic...")
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()

        # Save to temp .wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            write(tmpfile.name, SAMPLE_RATE, recording)
            wav_path = tmpfile.name

        print("📼 Transcribing...")
        segments, _ = model.transcribe(wav_path)
        transcription = " ".join([seg.text for seg in segments])

        print(f"📝 Transcribed: {transcription}")

        # Send transcription to orchestrator
        response = requests.post(ORCHESTRATOR_URL, json={"transcription": transcription})
        print(f"📨 Sent to orchestrator, response: {response.status_code}")

        return {"message": "Transcription complete", "sent": True}

    except Exception as e:
        return {"error": str(e)}
    
# Run this file
if __name__ == "__main__":
    uvicorn.run("voice_agent:app", host="0.0.0.0", port=8001, reload=True)
