from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "MediVision Quality Lab API"
    api_prefix: str = "/api"
    model_path: str = "../models/pneumonia_resnet18_224.pt"
    model_service_url: str | None = None
    disable_model_proxy: bool = False
    allowed_origins: str = "http://localhost:5173,http://localhost:3000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
