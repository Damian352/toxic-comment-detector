from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.inference import LABELS

router = APIRouter(prefix="/api", tags=["inference"])


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


class PredictResponse(BaseModel):
    probabilities: dict[str, float]
    labels: list[str] = Field(default_factory=lambda: list(LABELS))


def get_service() -> ToxicInferenceService:
    from app.main import app

    return app.state.inference  # type: ignore[attr-defined]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict[str, bool | str]:
    from app.main import app

    ready_flag: bool = app.state.model_loaded
    return {"model_loaded": ready_flag, "model_path": str(settings.model_path)}


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    service = get_service()
    try:
        probs = service.predict_proba(body.text.strip())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return PredictResponse(probabilities=probs)
