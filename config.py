import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GEMINI_MODEL = "gemini-3-flash-preview"  #"gemini-2.5-pro"
OPENAI_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"