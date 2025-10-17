from openai import OpenAI
import os
import tempfile
from pydub import AudioSegment


def transcribe_audio(audio_file):
    """Transcribe audio using OpenAI's Whisper API.

    If the file is larger than 25MB, it will be compressed to MP3 format
    at a lower bitrate to reduce file size.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    # Check file size (25MB = 26214400 bytes)
    file_size = os.path.getsize(audio_file)
    max_size = 25 * 1024 * 1024  # 25MB in bytes

    if file_size > max_size:
        # Compress to MP3 to reduce file size
        print(f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds 25MB limit. Compressing...")
        audio = AudioSegment.from_file(audio_file)

        # Create temporary MP3 file with lower bitrate
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
            # Export at 64kbps mono - good enough for transcription
            audio.export(temp_path, format="mp3", bitrate="64k", parameters=["-ac", "1"])

        try:
            compressed_size = os.path.getsize(temp_path)
            print(f"Compressed to {compressed_size / 1024 / 1024:.1f}MB")

            with open(temp_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            return transcription.text
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    else:
        # File is small enough, use directly
        with open(audio_file, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return transcription.text
