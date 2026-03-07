from fastapi import FastAPI
from app.template_generation.router import router as template_router
from app.patient_case_generation.router import router as case_router
from app.enhance_template_text.router import router as enhance_router
from app.MedAi_chatbot.router import router as chatbot_router
from app.auido_to_text.router import router as audio_router
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI(
    title="MedAI Doctor Helper System",
    description="AI-powered medical template and case generation system for doctors",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(template_router, prefix="/api/v1/template-generation", tags=["Doctor Template Generation"])
app.include_router(case_router, prefix="/api/v1/patient-case", tags=["Patient Case Generation"])
app.include_router(enhance_router, prefix="/api/v1/enhance-template", tags=["Enhance Template Text"])
app.include_router(chatbot_router, prefix="/api/v1/medai-chatbot", tags=["MedAI Chatbot"])
app.include_router(audio_router, prefix="/api/v1/audio-to-text", tags=["Audio to Text"])


@app.get("/", tags=["Health Check"])
def root():
    return {"status": "ok", "message": "MedAI Doctor Helper System is running"}