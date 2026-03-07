from pydantic import BaseModel, Field
from typing import Optional


class TemplateGenerationResponse(BaseModel):
    example: str = Field(..., description="A fully filled-in example template based on provided input")
    structure: str = Field(..., description="A structural template with placeholders and instructions")