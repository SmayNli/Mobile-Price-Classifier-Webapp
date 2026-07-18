from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.config import FEATURE_COLUMNS, FEATURE_META, MODEL_PATH, SCALER_PATH, CATEGORIES, PRESETS
from app.predict import predict_price

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Mobile Price Classifier",
    description="Kaggle Mobile Price Classification modeli ile fiyat tahmini",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class PredictionRequest(BaseModel):
    battery_power: float = Field(..., ge=400, le=2100)
    blue: int = Field(..., ge=0, le=1)
    clock_speed: float = Field(..., ge=0.3, le=3.5)
    dual_sim: int = Field(..., ge=0, le=1)
    fc: float = Field(..., ge=0, le=22)
    four_g: int = Field(..., ge=0, le=1)
    int_memory: float = Field(..., ge=1, le=70)
    m_dep: float = Field(..., ge=0.05, le=1.1)
    mobile_wt: float = Field(..., ge=70, le=220)
    n_cores: float = Field(..., ge=1, le=10)
    pc: float = Field(..., ge=0, le=24)
    px_height: float = Field(..., ge=0, le=2100)
    px_width: float = Field(..., ge=400, le=2100)
    ram: float = Field(..., ge=200, le=4200)
    sc_h: float = Field(..., ge=4, le=21)
    sc_w: float = Field(..., ge=0, le=20)
    talk_time: float = Field(..., ge=1, le=24)
    three_g: int = Field(..., ge=0, le=1)
    touch_screen: int = Field(..., ge=0, le=1)
    wifi: int = Field(..., ge=0, le=1)


@app.on_event("startup")
def validate_artifacts() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model dosyası bulunamadı: {MODEL_PATH}. "
            "Önce scripts/export_artifacts.py çalıştırın veya mobile_classifier.pth dosyasını models/ klasörüne koyun."
        )
    if not SCALER_PATH.exists():
        raise RuntimeError(
            f"Scaler dosyası bulunamadı: {SCALER_PATH}. "
            "Önce scripts/export_artifacts.py çalıştırın."
        )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "feature_columns": FEATURE_COLUMNS,
            "feature_meta": FEATURE_META,
            "categories": CATEGORIES,
            "presets": PRESETS,
        },
    )


@app.get("/api/features")
async def get_features():
    return {
        "features": FEATURE_META,
        "columns": FEATURE_COLUMNS,
        "categories": CATEGORIES,
        "presets": PRESETS
    }


@app.post("/api/predict")
async def predict(payload: PredictionRequest):
    try:
        result = predict_price(payload.model_dump())
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
