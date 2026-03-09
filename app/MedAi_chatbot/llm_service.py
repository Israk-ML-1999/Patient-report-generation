import google.generativeai as genai
from typing import Optional, List, Dict
from config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)


# System prompt is STATIC — no placeholders here.
# Dynamic context (history + user info) is injected into the user-turn prompt at call time.
SYSTEM_PROMPT = """You are MedAI, a concise and expert medical AI assistant for doctors.

GREETING: When greeted (hi, hello, hey, etc.), respond only with:
"Hello! I'm MedAI, your medical assistant. How can I help you today?"
Never mention patient information in a greeting.

RESPONSE STYLE:
- Keep all answers short and effective — under 150 words unless detail is truly necessary.
- Be accurate, professional, and clinically relevant.
- Do not fabricate diagnoses or medications. Base answers on established medical knowledge.
- For emergencies, always advise the doctor to act immediately.

GENERAL QUESTIONS (no patient info needed):
- Answer any general medical question (e.g., "What is diabetes?", "Fever for 1 week — what to do?") clearly and directly.
- Include practical tips and common medication names where relevant.

WITH PATIENT INFORMATION:
- Use the patient's details (condition, symptoms, history, age) to give personalized, context-aware answers.
- When asked for question suggestions, provide a concise numbered list of clinically relevant questions the doctor should ask THIS specific patient based on their information.

WITHOUT PATIENT INFORMATION (question suggestions only):
- If the doctor asks for patient question suggestions and no patient info is provided, respond only with:
  "I don't have any patient information. Please share the patient's details and I'll suggest the right questions to ask them."
- Do NOT generate question suggestions without patient information.
"""


def build_prompt(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]],
    patient_information: Optional[str],
) -> str:
    """
    Build the full user-turn prompt by injecting:
      - patient_information  (static personal/patient context)
      - conversation_history  (last 3 Q&A pairs for memory)
      - current user_query

    All dynamic context lives HERE in the prompt, not in system_instruction,
    because system_instruction is sent once at model-init time and cannot change per request.
    """
    parts = []

    # 1. User / patient profile context
    if patient_information:
        parts.append(f"[User Information]\n{patient_information}\n")

    # 2. Conversation memory — last 3 pairs only
    if conversation_history:
        history = conversation_history[-3:]
        parts.append("[Recent Conversation History]")
        for pair in history:
            uq = pair.get("user_query", "").strip()
            ar = pair.get("ai_response", "").strip()
            if uq:
                parts.append(f"User: {uq}")
            if ar:
                parts.append(f"MedAI: {ar}")
        parts.append("")  # blank line separator

    # 3. Current query
    parts.append(f"User: {user_query}")
    parts.append("MedAI:")

    return "\n".join(parts)


async def medai_chat(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]],
    patient_information: Optional[str],
) -> str:
    # Model is initialised with the static system prompt only
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    # All dynamic context is baked into the prompt string
    prompt = build_prompt(user_query, conversation_history, patient_information)
    response = model.generate_content(prompt)
    return response.text.strip()