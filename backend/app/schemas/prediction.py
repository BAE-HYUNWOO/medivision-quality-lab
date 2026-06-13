from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_index: int
    predicted_label: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    disclaimer: str


class GradCamPredictionResponse(PredictionResponse):
    gradcam_png_base64: str
