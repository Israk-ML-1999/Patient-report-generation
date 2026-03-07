from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from app.patient_case_generation.case_request import PatientCaseResponse
from app.patient_case_generation.llm_service import generate_patient_case

router = APIRouter()

ALLOWED_MIME_TYPES = {
    "application/pdf": "application/pdf",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "image/webp": "image/webp",
}


@router.post(
    "/generate",
    response_model=PatientCaseResponse,
    summary="Generate Patient Case",
    description=(
        "Generate a detailed patient case document (ambulatory, inpatient, surgery, radiology, etc.) "
        "based on case type, date/time, patient name, user query, transcribed doctor-patient conversation, "
        "optional report file, and a reference template format."
    ),
)
async def patient_case_generation(
    case_type: str = Form(..., description="Case type: Ambulatory, Inpatient, Surgery, Radiology, Psychiatry, etc."),
    language: str = Form("English", description="Language for case output"),
    date_time: Optional[str] = Form(None, description="Date and time of the case"),
    patient_name: Optional[str] = Form(None, description="Patient's name (optional)"),
    user_query: Optional[str] = Form(None, description="Doctor's instructions or context query"),
    audio_text: Optional[str] = Form(None, description="Transcribed doctor-patient conversation text"),
    reference_template: Optional[str] = Form(None, description="Reference template format for AI to follow"),
    file: Optional[UploadFile] = File(None, description="Optional PDF or image report for context"),
):
    if not user_query and not audio_text and not file:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'user_query', 'audio_text', or 'file' must be provided.",
        )

    file_bytes = None
    file_mime = None

    if file:
        content_type = file.content_type or ""
        file_mime = ALLOWED_MIME_TYPES.get(content_type)
        if not file_mime:
            filename = (file.filename or "").lower()
            if filename.endswith(".pdf"):
                file_mime = "application/pdf"
            elif filename.endswith((".jpg", ".jpeg")):
                file_mime = "image/jpeg"
            elif filename.endswith(".png"):
                file_mime = "image/png"
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Allowed: PDF, JPEG, PNG, WEBP.",
                )
        file_bytes = await file.read()

    try:
        result = await generate_patient_case(
            case_type=case_type,
            date_time=date_time,
            patient_name=patient_name,
            language=language,
            user_query=user_query,
            audio_text=audio_text,
            reference_template=reference_template,
            file_bytes=file_bytes,
            file_mime=file_mime,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patient case generation failed: {str(e)}")

    import json as _json

    example_data = result.get("example", "")
    structure_data = result.get("structure", "")

    if isinstance(example_data, (dict, list)):
        example_data = _json.dumps(example_data, ensure_ascii=False, indent=2)
    elif not isinstance(example_data, str):
        example_data = str(example_data)

    if isinstance(structure_data, (dict, list)):
        structure_data = _json.dumps(structure_data, ensure_ascii=False, indent=2)
    elif not isinstance(structure_data, str):
        structure_data = str(structure_data)

    return PatientCaseResponse(
        example=example_data,
        structure=structure_data,
    )