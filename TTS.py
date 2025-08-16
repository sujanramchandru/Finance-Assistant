from fastapi import FastAPI
from pydantic import BaseModel
import pyttsx3

app = FastAPI()

class TTSRequest(BaseModel):
    text: str

@app.post("/speak")
def speak(req: TTSRequest):
    # Create new engine instance for each request
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(req.text)
    engine.runAndWait()
    engine.stop()
    return {"message": f"Spoken: {req.text}"}
