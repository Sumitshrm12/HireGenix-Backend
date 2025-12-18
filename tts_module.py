# tts_module.py
import os, requests, base64

from dotenv import load_dotenv

load_dotenv()


def generate_tts(text: str) -> str:
    ssml = f"<speak version='1.0' xml:lang='en-US'><voice name='en-US-AriaNeural'>{text}</voice></speak>"
    resp = requests.post(
        os.getenv("AZURE_TTS_URL"),
        data=ssml,
        headers={
            "Ocp-Apim-Subscription-Key": os.getenv("AZURE_SPEECH_KEY"),
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm"
        }
    )
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("utf-8")