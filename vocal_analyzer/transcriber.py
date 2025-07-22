import openai
import os


def transcribe_audio(audio_file):
    """Transcribe audio using OpenAI's Whisper API."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    with open(audio_file, "rb") as f:
        transcription = openai.Audio.transcribe("whisper-1", f)
    return transcription["text"]
