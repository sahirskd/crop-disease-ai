from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import router

app = FastAPI(
    title="CropSense AI",
    description="Crop disease detection API powered by EfficientNetB3 + Grad-CAM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.MODEL_NAME}
