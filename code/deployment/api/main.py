"""
Exposes the trained house price model over HTTP using FastAPI.

Endpoints:
- GET  /health   -> simple liveness check
- POST /predict  -> predict SalePrice from the 12 key input fields
"""

from fastapi import FastAPI, HTTPException

from model_loader import ModelBundle
from schemas import HouseFeaturesInput, PredictionResponse

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts house sale price (Ames Housing dataset) from key features.",
    version="1.0.0",
)

model_bundle: ModelBundle | None = None


@app.on_event("startup")
def load_model() -> None:
    """Load the trained model once when the API process starts."""
    global model_bundle
    model_bundle = ModelBundle()
    print("[api] Model and feature defaults loaded successfully.")


@app.get("/health")
def health_check() -> dict:
    """Basic liveness/readiness check."""
    return {"status": "ok", "model_loaded": model_bundle is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeaturesInput) -> PredictionResponse:
    """Predict the sale price of a house from the provided key features.

    Any feature not listed in HouseFeaturesInput is filled in with the
    training-data default (median for numeric columns, mode for
    categorical columns) before being passed to the model.
    """
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        predicted_price = model_bundle.predict(features.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(predicted_price=predicted_price)
