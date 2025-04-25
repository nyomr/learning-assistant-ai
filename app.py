from groq import Groq
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from dotenv import load_dotenv
import time
import os
import tempfile
from yt_dlp import YoutubeDL

# App
load_dotenv()
app = FastAPI()

# Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class TranscriptionResponse(BaseModel):
    status_code: int
    transcription: str
    inference_time: float


class YoutubeLink(BaseModel):
    url: str


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


@app.post("/transcribe-youtube", response_model=TranscriptionResponse)
async def transcribe_youtube(link: YoutubeLink):

    try:
        start_time = time.time()

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "yt_audio.%(ext)s")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_path,
            'quiet': True,
            'prefer_ffmpeg': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([link.url])

        actual_path = temp_path.replace(".%(ext)s", ".mp3")

        with open(actual_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(f.name, f.read()),
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

    finally:
        if actual_path and os.path.exists(actual_path):
            os.remove(actual_path)
            print(f"[INFO] Temporary file has been deleted: {actual_path}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5050)
