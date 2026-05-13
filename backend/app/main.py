from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.services.inference import ToxicInferenceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ToxicInferenceService(settings.model_path)
    app.state.inference = service
    app.state.model_loaded = False
    try:
        service.load()
        app.state.model_loaded = True
    except FileNotFoundError:
        app.state.model_loaded = False
    yield


app = FastAPI(
    title="Toxic Comment Detector API",
    description="Inference API for multi-label toxic comment classification.",
    version="0.1.0",
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
