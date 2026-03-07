from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from app.template_generation.template_request import TemplateGenerationResponse
from app.template_generation.llm_service import generate_template

router = APIRouter()

ALLOWED_MIME_TYPES = {
    "application/pdf": "application/pdf",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "image/webp": "image/webp",
    "image/heic": "image/heic",
    "image/heif": "image/heif",
}


@router.post(
    "/generate",
    response_model=TemplateGenerationResponse,
    summary="Generate Doctor Template",
    description=(
        "Generate a medical template (patient letter, referral, report, etc.) based on "
        "query/transcribed text, optional doctor/patient names, optional PDF/image context, "
        "and selected output language."
    ),
)
async def doctor_template_generation(
    query: Optional[str] = Form(None, description="Instruction text or transcribed audio text"),
    doctor_name: Optional[str] = Form(None, description="Doctor's name (optional)"),
    patient_name: Optional[str] = Form(None, description="Patient's name (optional)"),
    language: str = Form("English", description="Language for template output (e.g., English, Bengali, French)"),
    file: Optional[UploadFile] = File(None, description="Optional PDF, DOC, or image file for context"),
):
    if not query and not file:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'query' or 'file' must be provided.",
        )

    file_bytes = None
    file_mime = None

    if file:
        content_type = file.content_type or ""
        # Normalize content type
        file_mime = ALLOWED_MIME_TYPES.get(content_type)
        if not file_mime:
            # Try to infer from filename
            filename = (file.filename or "").lower()
            if filename.endswith(".pdf"):
                file_mime = "application/pdf"
            elif filename.endswith((".jpg", ".jpeg")):
                file_mime = "image/jpeg"
            elif filename.endswith(".png"):
                file_mime = "image/png"
            elif filename.endswith(".webp"):
                file_mime = "image/webp"
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type '{content_type}'. Allowed: PDF, JPEG, PNG, WEBP.",
                )
        file_bytes = await file.read()

    try:
        result = await generate_template(
            query=query,
            doctor_name=doctor_name,
            patient_name=patient_name,
            language=language,
            file_bytes=file_bytes,
            file_mime=file_mime,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Template generation failed: {str(e)}")

    example_data = result.get("example", "")
    structure_data = result.get("structure", "")

    if isinstance(example_data, (dict, list)):
        import json
        example_data = json.dumps(example_data, ensure_ascii=False, indent=2)
    elif not isinstance(example_data, str):
        example_data = str(example_data)

    if isinstance(structure_data, (dict, list)):
        import json
        structure_data = json.dumps(structure_data, ensure_ascii=False, indent=2)
    elif not isinstance(structure_data, str):
        structure_data = str(structure_data)

    return TemplateGenerationResponse(
        example=example_data,
        structure=structure_data,
    )