from __future__ import annotations

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.model_loader import ModelBundle, ModelLoadError
from app.schemas.prediction import GradCamPredictionResponse, PredictionResponse
from app.services.gradcam_service import predict_with_gradcam
from app.services.inference_service import predict_image_bytes

router = APIRouter()


def should_proxy_to_model_service() -> bool:
    settings = get_settings()
    return bool(settings.model_service_url) and not settings.disable_model_proxy


async def proxy_file(endpoint: str, file: UploadFile) -> dict:
    settings = get_settings()
    if not settings.model_service_url:
        raise RuntimeError("MODEL_SERVICE_URL is not set")
    data = await file.read()
    files = {"file": (file.filename or "image.png", data, file.content_type or "image/png")}
    url = settings.model_service_url.rstrip("/") + endpoint
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, files=files)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if should_proxy_to_model_service():
        return await proxy_file("/api/predict", file)

    settings = get_settings()
    data = await file.read()
    try:
        bundle = ModelBundle(settings.model_path)
        return predict_image_bytes(bundle, data)
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@router.post("/predict-with-gradcam", response_model=GradCamPredictionResponse)
async def predict_gradcam(file: UploadFile = File(...)):
    if should_proxy_to_model_service():
        return await proxy_file("/api/predict-with-gradcam", file)

    settings = get_settings()
    data = await file.read()
    try:
        bundle = ModelBundle(settings.model_path)
        return predict_with_gradcam(bundle, data)
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Grad-CAM prediction failed: {exc}")
