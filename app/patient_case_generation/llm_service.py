import google.generativeai as genai
import json
import base64
from typing import Optional
from config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = """You are an expert medical documentation specialist who generates detailed, accurate patient case reports for clinical settings.

Your task is to analyze all provided inputs and generate a comprehensive patient case document appropriate for the specified case type.

Case types include: Ambulatory, Inpatient, Surgery, Radiology, Psychiatry, Emergency, Outpatient, ICU, etc.

CRITICAL RULES:
1. NEVER fabricate or invent clinical details, diagnoses, lab values, medications, or patient information not explicitly present in the provided input.
2. If information is absent, omit that section entirely — do NOT mention that it was not provided.
3. Use clinical, professional language appropriate for the case type and setting.
4. Always generate the case in the SELECTED LANGUAGE. This is mandatory.It maximum time hungarian language.
5. When you generate patient case (note) it generate detailed  and clear if any doctor or nurse read it he can understand the case easily. Not it should be short.
6. Use the provided reference template format if supplied; otherwise use best clinical practice format for the given case type.
7. Return ONLY a valid JSON object with exactly two keys: "example" and "structure".
   - "example": A single markdown-formatted STRING containing a fully completed case filled with details from the inputs. Use markdown headings (##), bold (**text**), and bullet lists where appropriate for readability.
   - "structure": A single markdown-formatted STRING containing a blank structural template with placeholders (e.g., [Patient Name]) and conditional instructions for this case type.
   IMPORTANT: Both "example" and "structure" MUST be plain strings with markdown inside — NOT nested JSON objects or dicts.
8. Do not include markdown fences or extra text outside the JSON.
"""


def build_content_parts(
    case_type: str,
    date_time: Optional[str],
    patient_name: Optional[str],
    language: str,
    user_query: Optional[str],
    audio_text: Optional[str],
    reference_template: Optional[str],
    file_bytes: Optional[bytes],
    file_mime: Optional[str],
) -> list:
    parts = []

    text_lines = [
        f"Selected Language: {language}",
        f"Case Type: {case_type}",
    ]
    if date_time:
        text_lines.append(f"Date and Time: {date_time}")
    if patient_name:
        text_lines.append(f"Patient Name: {patient_name}")
    if user_query:
        text_lines.append(f"\nUser Query / Instructions:\n{user_query}")
    if audio_text:
        text_lines.append(f"\nDoctor-Patient Conversation (Transcribed Audio):\n{audio_text}")
    if reference_template:
        text_lines.append(f"\nReference Template Format:\n{reference_template}")

    parts.append("\n".join(text_lines))

    if file_bytes and file_mime:
        parts.append({"mime_type": file_mime, "data": base64.b64encode(file_bytes).decode("utf-8")})

    parts.append(
        "\nBased on all inputs above, generate a patient case document. "
        "Return ONLY a valid JSON object with keys 'example' and 'structure'. "
        f"Both values must be written entirely in {language}."
    )

    return parts


async def generate_patient_case(
    case_type: str,
    date_time: Optional[str],
    patient_name: Optional[str],
    language: str,
    user_query: Optional[str],
    audio_text: Optional[str],
    reference_template: Optional[str],
    file_bytes: Optional[bytes],
    file_mime: Optional[str],
) -> dict:
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    parts = build_content_parts(
        case_type, date_time, patient_name, language,
        user_query, audio_text, reference_template, file_bytes, file_mime
    )

    response = model.generate_content(parts)
    raw_text = response.text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.split("```", 2)[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.rsplit("```", 1)[0].strip()

    result = json.loads(raw_text)
    return result