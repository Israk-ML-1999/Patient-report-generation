import openai
import tempfile
import os
from config import OPENAI_API_KEY, OPENAI_TRANSCRIPTION_MODEL

client = openai.OpenAI(api_key=OPENAI_API_KEY)

SUPPORTED_AUDIO_FORMATS = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".mp4",
    "audio/m4a": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
    "audio/flac": ".flac",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}

EXTENSION_MAP = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".mp4": "audio/mp4",
    ".m4a": "audio/m4a",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
}


def get_extension(content_type: str, filename: str) -> str:
    """Determine file extension from content type or filename."""
    ext = SUPPORTED_AUDIO_FORMATS.get(content_type)
    if ext:
        return ext
    # Fallback: infer from filename
    _, file_ext = os.path.splitext(filename.lower())
    if file_ext in EXTENSION_MAP:
        return file_ext
    return None


async def transcribe_audio(file_bytes: bytes, content_type: str, filename: str) -> str:
    """Transcribe audio using OpenAI gpt-4o-transcribe model."""
    ext = get_extension(content_type, filename)
    if not ext:
        raise ValueError(
            f"Unsupported audio format '{content_type}'. "
            "Supported: mp3, wav, mp4, m4a, ogg, webm, flac"
        )

    # Write to temp file — OpenAI SDK requires a file-like object with a name
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=OPENAI_TRANSCRIPTION_MODEL,
                file=audio_file,
                response_format="text",
            )
        return transcript if isinstance(transcript, str) else transcript.text
    finally:
        os.unlink(tmp_path)