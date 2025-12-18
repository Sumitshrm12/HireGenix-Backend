import os
import io
import asyncio
import httpx
import ffmpeg
import base64
from dotenv import load_dotenv

load_dotenv()
# Constants
MAX_RETRIES = 3
INITIAL_TIMEOUT = 5  # seconds base for timeout multiplier

# Environment variables
WHISPER_ENDPOINT = os.getenv("WHISPER_API_ENDPOINT")  # e.g. "https://<region>.stt.speech.azure.com/speech/recognition/conversation/cognitiveservices/v1"
AZURE_SPEECH_KEY = os.getenv("AZURE_API_KEY_SPEECH")

# Validate env vars
if not WHISPER_ENDPOINT:
    raise RuntimeError("Missing environment variable: WHISPER_API_ENDPOINT")
if not AZURE_SPEECH_KEY:
    raise RuntimeError("Missing environment variable: AZURE_API_KEY_SPEECH")

async def convert_webm_to_wav(webm_bytes: bytes) -> bytes:
    """
    Convert in-memory WebM bytes to WAV format using ffmpeg.
    """
    in_buffer = io.BytesIO(webm_bytes)
    # Run ffmpeg, capture stdout as wav bytes
    out, _ = (
        ffmpeg
        .input('pipe:0')
        .output('pipe:1', format='wav')
        .run(input=in_buffer.read(), capture_stdout=True, capture_stderr=True)
    )
    return out

async def transcribe_wav(wav_bytes: bytes) -> str:
    """
    Send WAV bytes to Azure Whisper (HTTP) endpoint with retry logic.

    Raises:
        Exception: last encountered error if all retries fail.

    Returns:
        Transcribed text on success.
    """
    last_error = None

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "audio/wav",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient() as client:
        for attempt in range(MAX_RETRIES):
            try:
                timeout = INITIAL_TIMEOUT * (attempt + 1)
                response = await client.post(
                    WHISPER_ENDPOINT,
                    content=wav_bytes,
                    headers=headers,
                    params={"language": "en-US"},
                    timeout=timeout
                )
                response.raise_for_status()
                data = response.json()
                if data.get("RecognitionStatus") == "Success":
                    return data.get("DisplayText", "")
                else:
                    raise Exception(f"RecognitionStatus: {data.get('RecognitionStatus')}")
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep((attempt + 1) * 1)
                else:
                    raise last_error

    return ""

# Helper to decode base64 webm payload from client

def decode_webm_payload(b64_string: str) -> bytes:
    """
    Decode a base64 string into raw WebM bytes.
    """
    return base64.b64decode(b64_string)