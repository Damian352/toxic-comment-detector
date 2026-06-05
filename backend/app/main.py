from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.services.registry import InferenceRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = InferenceRegistry(settings.model_path, settings.bert_model_dir)
    app.state.registry = registry
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
