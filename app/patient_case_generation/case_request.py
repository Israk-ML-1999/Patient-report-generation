from pydantic import BaseModel, Field
from typing import Optional


class PatientCaseResponse(BaseModel):
    example: str = Field(..., description="A fully filled-in patient case based on provided input")
    structure: str = Field(..., description="A structural case template with placeholders and instructions")