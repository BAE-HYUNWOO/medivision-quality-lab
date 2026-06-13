from fastapi import APIRouter
from app.core.config import get_settings
from app.core.model_loader import ModelBundle

router = APIRouter()


@router.get("/health")
def health():
    settings = get_settings()
    bundle = ModelBundle(settings.model_path)
    return {
        "status": "ok",
        "app": settings.app_name,
        "model_path": settings.model_path,
        "model_file_exists": bundle.available(),
        "model_service_url": settings.model_service_url,
        "proxy_disabled": settings.disable_model_proxy,
        "research_only": True,
    }
