from fastapi import APIRouter, HTTPException
from app.MedAi_chatbot.chat_request import ChatTextRequest, ChatTextResponse
from app.MedAi_chatbot.llm_service import medai_chat

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatTextResponse,
    summary="MedAI Chatbot",
    description=(
        "Medical AI chatbot for doctors and patients. "
        "Supports conversation history (last 3 pairs) and user information for personalized responses. "
        "Doctors can also request patient question suggestions."
    ),
)
async def medai_chatbot(request: ChatTextRequest):
    try:
        answer = await medai_chat(
            user_query=request.user_query,
            conversation_history=request.conversation_history,
            user_information=request.user_information,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")

    return ChatTextResponse(answer=answer)