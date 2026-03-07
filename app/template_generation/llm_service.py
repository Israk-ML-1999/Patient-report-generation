import google.generativeai as genai
import json
import base64
from typing import Optional
from config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = """You are an expert medical documentation assistant specializing in generating professional clinical templates for doctors.

Your task is to analyze the provided inputs (query, doctor name, patient name, and any attached documents/images) and generate a medical template based on the topic or type mentioned.

Template types include but are not limited to: patient letters, referral letters, discharge summaries, clinical notes, radiology reports, prescription letters, etc.

CRITICAL RULES:
1. NEVER invent or fabricate patient details, medical assessments, diagnoses, treatments, or clinical findings not explicitly mentioned in the input.
2. If information is not provided, omit that section entirely — do NOT state that it was not mentioned.
3. Use patient-friendly, clear, and empathetic language. Avoid medical jargon or briefly explain it.
4. When you generate template it generate detailed and clear if any doctor or nurse read it he can understand the case easily. Not it should be short.
5. Always generate the template in the SELECTED LANGUAGE specified by the user. This is mandatory.Other wise generate which language is used in the input.
6. Return your response as a valid JSON object with exactly two keys: "example" and "structure".
   - "example": A fully completed template filled with details from the provided input.
   - "structure": A blank structural template with placeholders and conditional instructions.
7. The JSON must be valid and parseable. Do not include markdown fences or extra text outside the JSON.
"""


def build_content_parts(
    query: Optional[str],
    doctor_name: Optional[str],
    patient_name: Optional[str],
    language: str,
    file_bytes: Optional[bytes],
    file_mime: Optional[str],
) -> list:
    """Build the content parts list for the Gemini API call."""
    parts = []

    # Text context
    text_lines = [f"Selected Language: {language}"]
    if doctor_name:
        text_lines.append(f"Doctor Name: {doctor_name}")
    if patient_name:
        text_lines.append(f"Patient Name: {patient_name}")
    if query:
        text_lines.append(f"\nInstruction / Query / Transcribed Text:\n{query}")

    parts.append("\n".join(text_lines))

    # File context (PDF, image, DOCX treated as inline data)
    if file_bytes and file_mime:
        parts.append({"mime_type": file_mime, "data": base64.b64encode(file_bytes).decode("utf-8")})

    parts.append(
        "\nBased on all the above inputs, generate a medical template. "
        "Return ONLY a valid JSON object with keys 'example' and 'structure'. "
        f"Both values must be written entirely in {language}."
    )

    return parts


async def generate_template(
    query: Optional[str],
    doctor_name: Optional[str],
    patient_name: Optional[str],
    language: str,
    file_bytes: Optional[bytes],
    file_mime: Optional[str],
) -> dict:
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    parts = build_content_parts(query, doctor_name, patient_name, language, file_bytes, file_mime)

    response = model.generate_content(parts)
    raw_text = response.text.strip()

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```", 2)[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.rsplit("```", 1)[0].strip()

    result = json.loads(raw_text)
    return result