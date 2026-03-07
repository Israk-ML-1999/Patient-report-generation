from fastapi import APIRouter, HTTPException
from app.enhance_template_text.enhnce_request import EnhanceTemplateRequest, EnhanceTemplateResponse
from app.enhance_template_text.llm_service import enhance_template

router = APIRouter()


@router.post(
    "/enhance",
    response_model=EnhanceTemplateResponse,
    summary="Enhance / Modify Template",
    description=(
        "Modify an existing medical template based on a text instruction or transcribed voice command. "
        "Optionally pass previous_text as memory context for better understanding."
    ),
)
async def enhance_template_endpoint(request: EnhanceTemplateRequest):
    try:
        regenerated = await enhance_template(
            instruction_query=request.instruction_query,
            template_text=request.template_text,
            previous_text=request.previous_text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Template enhancement failed: {str(e)}")

    return EnhanceTemplateResponse(regenerated_template=regenerated)