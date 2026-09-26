from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.model import BeerClassifier, get_class_names, load_model, predict


BASE_DIR = Path(__file__).resolve().parent.parent


class BeerFeatures(BaseModel):
    og: float = Field(..., gt=0, le=2, description="Original Gravity")
    abv: float = Field(..., ge=0, le=100, description="Alcohol by Volume")
    ph: float = Field(..., ge=0, le=14, description="Asitlik seviyesi")
    ibu: float = Field(..., ge=0, le=1000, description="Acılık birimi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_error = None
    try:
        model, model_path = load_model()
        app.state.model = model
        app.state.model_path = str(model_path)
    except Exception as exc:
        app.state.model_error = str(exc)
    yield


app = FastAPI(
    title="Bira Sınıflandırıcı",
    description="OG, ABV, pH ve IBU değerlerinden sınıf tahmini yapar.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"model_ready": request.app.state.model is not None},
    )


@app.get("/health")
async def health(request: Request):
    ready = request.app.state.model is not None
    return {
        "status": "ok" if ready else "model_unavailable",
        "model_ready": ready,
        "detail": None if ready else request.app.state.model_error,
    }


@app.post("/api/predict")
async def make_prediction(features: BeerFeatures, request: Request):
    model: BeerClassifier | None = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.model_error or "Model yüklenemedi.",
        )

    try:
        return predict(
            model,
            [features.og, features.abv, features.ph, features.ibu],
            get_class_names(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tahmin yapılamadı: {exc}") from exc
