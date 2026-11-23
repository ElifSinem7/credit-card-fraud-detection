"""Credit Card Fraud Detection API.

Run:
    uvicorn app:app --reload
Docs:
    http://127.0.0.1:8000/docs
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import Field, create_model

MODEL_PATH = Path(__file__).parent / "models" / "fraud_model.joblib"

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
THRESHOLD = bundle["threshold"]
FEATURES = bundle["features"]

# V1..V28 + Amount + Hour: the request schema is generated from the feature list
Transaction = create_model(
    "Transaction",
    **{f"V{i}": (float, ...) for i in range(1, 29)},
    Amount=(float, Field(..., ge=0, description="Transaction amount ($)")),
    Hour=(float, Field(..., ge=0, lt=24, description="Hour of day (0-23)")),
)

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["model_name"], "threshold": THRESHOLD}


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    # Keep the column order the model was trained with
    df_input = pd.DataFrame([transaction.model_dump()])[FEATURES]

    probability = float(pipeline.predict_proba(df_input)[0, 1])
    is_fraud = probability >= THRESHOLD

    return {
        "fraud_prediction": int(is_fraud),
        "fraud_probability": round(probability, 4),
        "threshold": THRESHOLD,
        "status": "Risky Transaction" if is_fraud else "Normal Transaction",
    }
