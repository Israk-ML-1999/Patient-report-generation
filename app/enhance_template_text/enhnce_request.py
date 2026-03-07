from pydantic import BaseModel, Field
from typing import Optional


class EnhanceTemplateRequest(BaseModel):
    instruction_query: str = Field(
        ...,
        min_length=1,
        description="Modification instruction (text or transcribed voice command)",
    )
    template_text: str = Field(
        ...,
        min_length=1,
        description="The current template text to be modified",
    )
    previous_text: Optional[str] = Field(
        None,
        description="Previous context text for better memory and understanding (optional)",
    )


class EnhanceTemplateResponse(BaseModel):
    regenerated_template: str = Field(..., description="The enhanced/modified template output")