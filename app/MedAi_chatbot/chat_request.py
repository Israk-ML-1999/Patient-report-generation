from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class ChatTextRequest(BaseModel):
    user_query: str = Field(..., min_length=1, description="User's question or message")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None,
        description=(
            "Last 3 pairs of conversation history. "
            "Each dict should have 'user_query' and 'ai_response' keys."
        ),
    )
    user_information: Optional[str] = Field(
        None,
        description="User profile / patient information as plain text for personalized responses",
    )


class ChatTextResponse(BaseModel):
    answer: str = ""