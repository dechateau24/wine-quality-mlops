import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from api.drift import generate_report_html

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "model.joblib")
with open(BASE_DIR / "models" / "features.json", "r", encoding="utf-8") as f:
    FEATURES = json.load(f)

app = FastAPI(
    title="Wine Quality API",
    description="Predit la note de qualite (3 a 9) d'un vin a partir de 11 mesures physico-chimiques.",
)


class Wine(BaseModel):
    fixed_acidity: float = Field(ge=3, le=17, description="Acidite fixe (g/L)")
    volatile_acidity: float = Field(ge=0.05, le=2.0, description="Acidite volatile (g/L)")
    citric_acid: float = Field(ge=0, le=2, description="Acide citrique (g/L)")
    residual_sugar: float = Field(ge=0.5, le=70, description="Sucre residuel (g/L)")
    chlorides: float = Field(ge=0.005, le=0.7, description="Chlorures (g/L)")
    free_sulfur_dioxide: float = Field(ge=1, le=300, description="Dioxyde de soufre libre (mg/L)")
    total_sulfur_dioxide: float = Field(ge=5, le=450, description="Dioxyde de soufre total (mg/L)")
    density: float = Field(ge=0.98, le=1.05, description="Densite (g/cm3)")
    pH: float = Field(ge=2.5, le=4.5, description="pH")
    sulphates: float = Field(ge=0.1, le=2.5, description="Sulfates (g/L)")
    alcohol: float = Field(ge=7, le=16, description="Degre d'alcool (% vol)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "fixed_acidity": 7.4,
                    "volatile_acidity": 0.7,
                    "citric_acid": 0.0,
                    "residual_sugar": 1.9,
                    "chlorides": 0.076,
                    "free_sulfur_dioxide": 11,
                    "total_sulfur_dioxide": 34,
                    "density": 0.9978,
                    "pH": 3.51,
                    "sulphates": 0.56,
                    "alcohol": 9.4,
                }
            ]
        }
    }


class Prediction(BaseModel):
    predicted_quality: float = Field(description="Note de qualite predite")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(wine: Wine):
    data = wine.model_dump()
    # Les colonnes du dataset ont des points (fixed.acidity), le JSON utilise des underscores
    row = {col: data[col.replace(".", "_")] for col in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    prediction = float(model.predict(X)[0])
    return {"predicted_quality": round(prediction, 2)}


@app.get("/drift-report", response_class=HTMLResponse)
def drift_report():
    return generate_report_html()