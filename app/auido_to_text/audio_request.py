from pydantic import BaseModel, Field


class AudioTranscriptionResponse(BaseModel):
    transcribed_text: str = Field(..., description="The transcribed text from the audio file")