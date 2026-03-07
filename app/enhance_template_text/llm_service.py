import google.generativeai as genai
from typing import Optional
from config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = """You are an expert medical documentation editor.

Your task is to modify and regenerate an existing medical template based on a user's instruction.

The instruction may come from typed text or a voice command that has been transcribed — understand and interpret it correctly regardless of minor transcription errors or informal phrasing.

CRITICAL RULES:
1. Apply the instruction precisely to the template_text.
2. Use the previous_text (if provided) as memory and context to better understand what changes are needed.
3. NEVER add new clinical information, patient details, or diagnoses not already present in the original template.
4. Preserve the original language and professional tone of the template.
5. Return ONLY the regenerated template text — no explanations, no preamble, no markdown fences.
6. Maintain the same language used in the original template_text.
7. Return llm response MUST be plain strings with markdown format.  
"""


async def enhance_template(
    instruction_query: str,
    template_text: str,
    previous_text: Optional[str],
) -> str:
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    content_parts = []

    if previous_text:
        content_parts.append(f"Previous Context (for memory and understanding):\n{previous_text}\n")

    content_parts.append(f"Current Template:\n{template_text}\n")
    content_parts.append(
        f"Instruction / Modification Request (may be from voice transcription):\n{instruction_query}\n\n"
        "Now apply the instruction and return ONLY the regenerated template text."
    )

    response = model.generate_content("\n".join(content_parts))
    return response.text.strip()