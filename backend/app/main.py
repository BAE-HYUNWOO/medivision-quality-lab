from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers import health, predict

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Research-only medical image AI demo with Grad-CAM explanations.",
    version="0.1.0",
)

origins = [origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
app.include_router(predict.router, prefix=settings.api_prefix, tags=["prediction"])


@app.get("/")
def root():
    return {
        "message": "MediVision Quality Lab API",
        "docs": "/docs",
        "research_only": True,
    }
