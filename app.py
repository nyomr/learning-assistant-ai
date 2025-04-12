from groq import Groq
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import time
import os

# App
load_dotenv()
app = FastAPI()

# Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class TranscriptionResponse(BaseModel):
    status_code: int
    transcription: str
    inference_time: float


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):

    try:
        file_content = await file.read()

        start_time = time.time()
        transcription = client.audio.transcriptions.create(
            file=(file.filename, file_content),
            model="distil-whisper-large-v3-en",
            response_format="verbose_json",
        )
        end_time = time.time()
        return TranscriptionResponse(
            status_code=200,
            transcription=transcription.text,
            inference_time=round(end_time - start_time, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5050)
