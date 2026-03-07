import google.generativeai as genai
from typing import Optional, List, Dict
from config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)


# System prompt is STATIC — no placeholders here.
# Dynamic context (history + user info) is injected into the user-turn prompt at call time.
SYSTEM_PROMPT = """You are MedAI, an expert medical AI assistant designed to support both doctors and patients.

Your capabilities:
1. Answer medical questions accurately and professionally.
2. When a DOCTOR is using the system, you can suggest relevant clear questions a doctor might ask their patient based on the patient's condition or information provided.
3. When a PATIENT is using the system, answer in clear, empathetic, and easy-to-understand language.
4. Use the provided conversation history and user information to give personalized, context-aware responses.
5. Always maintain patient safety — if a question suggests an emergency, advise seeking immediate medical care.
6. Do not fabricate diagnoses, medications, or clinical information. Base answers on established medical knowledge.
7. Be concise, accurate, and helpful.

For doctor users asking about patient question suggestions:
- Suggest a numbered list of clinically relevant questions to ask the patient based on their condition/information.
"""


def build_prompt(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]],
    user_information: Optional[str],
) -> str:
    """
    Build the full user-turn prompt by injecting:
      - user_information  (static personal/patient context)
      - conversation_history  (last 3 Q&A pairs for memory)
      - current user_query

    All dynamic context lives HERE in the prompt, not in system_instruction,
    because system_instruction is sent once at model-init time and cannot change per request.
    """
    parts = []

    # 1. User / patient profile context
    if user_information:
        parts.append(f"[User Information]\n{user_information}\n")

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
    user_information: Optional[str],
) -> str:
    # Model is initialised with the static system prompt only
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    # All dynamic context is baked into the prompt string
    prompt = build_prompt(user_query, conversation_history, user_information)
    response = model.generate_content(prompt)
    return response.text.strip()