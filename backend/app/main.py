"""
FastAPI application entry point.

Responsibilities:
- app creation and CORS;
- lifespan: load all inference models on startup;
- mount the `/api/*` router.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.services.projection import ProjectionService
from app.services.registry import InferenceRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models load synchronously before the first request; missing artifacts do not crash startup.
    registry = InferenceRegistry(
        settings.model_path,
        settings.bert_model_dir,
        settings.model_path_pl,
        settings.bert_model_dir_pl,
    )
    app.state.registry = registry
    app.state.projection = ProjectionService(settings.projections_dir)
    registry.load_all()
    yield


app = FastAPI(
    title="Toxic Comment Detector API",
    description=(
        "Inference API for multi-label toxic comment classification. "
        "Supports TF-IDF + Logistic Regression and fine-tuned BERT."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "toxic-comment-detector", "docs": "/docs"}
