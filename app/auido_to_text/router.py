from fastapi import APIRouter, UploadFile, File, HTTPException
from app.auido_to_text.audio_request import AudioTranscriptionResponse
from app.auido_to_text.audio_to_service import transcribe_audio

router = APIRouter()


@router.post(
    "/transcribe",
    response_model=AudioTranscriptionResponse,
    summary="Audio to Text Transcription",
    description=(
        "Upload an audio file to get transcribed text using OpenAI's gpt-4o-transcribe model. "
        "Supported formats: MP3, WAV, MP4, M4A, OGG, WEBM, FLAC."
    ),
)
async def audio_to_text(
    file: UploadFile = File(..., description="Audio file to transcribe"),
):
    if not file:
        raise HTTPException(status_code=400, detail="Audio file is required.")

    file_bytes = await file.read()
    content_type = file.content_type or ""
    filename = file.filename or "audio"

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        transcribed_text = await transcribe_audio(
            file_bytes=file_bytes,
            content_type=content_type,
            filename=filename,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    return AudioTranscriptionResponse(transcribed_text=transcribed_text)