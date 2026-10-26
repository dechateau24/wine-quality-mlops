import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "model.joblib")
with open(BASE_DIR / "models" / "features.json", "r", encoding="utf-8") as f:
    FEATURES = json.load(f)

app = FastAPI(title="Wine Quality API")


class Wine(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(wine: Wine):
    data = wine.model_dump()
    # Les colonnes du dataset ont des points (fixed.acidity), le JSON utilise des underscores
    row = {col: data[col.replace(".", "_")] for col in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    prediction = float(model.predict(X)[0])
    return {"predicted_quality": round(prediction, 2)}