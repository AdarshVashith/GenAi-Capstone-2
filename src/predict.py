"""Prediction helpers for the crop yield model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import warnings
import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

# Catch scikit-learn version warnings when loading the older model
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from config import LABEL_ENCODERS_PATH, RF_MODEL_PATH, SCALER_PATH


FEATURE_COLUMNS = [
    "Area",
    "Item",
    "average_rain_fall_mm_per_year",
    "avg_temp",
    "pesticides_tonnes",
]


def load_prediction_artifacts() -> tuple[Any, Any, dict[str, Any]]:
    """Load model artifacts required for inference."""
    required_paths = [RF_MODEL_PATH, SCALER_PATH, LABEL_ENCODERS_PATH]
    missing = [str(path) for path in required_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Add these files before running predictions: "
            + ", ".join(missing)
        )
    model = joblib.load(RF_MODEL_PATH)
    # Patch for scikit-learn >= 1.4 compatibility with older models
    if hasattr(model, "estimators_"):
        for estimator in model.estimators_:
            if not hasattr(estimator, "monotonic_cst"):
                estimator.monotonic_cst = None
    
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    return model, scaler, label_encoders


def encode_features(farm_data: dict[str, Any], label_encoders: dict[str, Any]) -> pd.DataFrame:
    """Encode categorical fields into the format expected by the model."""
    row = {
        "Area": label_encoders["Area"].transform([farm_data["Area"]])[0],
        "Item": label_encoders["Item"].transform([farm_data["Item"]])[0],
        "average_rain_fall_mm_per_year": float(farm_data["average_rain_fall_mm_per_year"]),
        "avg_temp": float(farm_data["avg_temp"]),
        "pesticides_tonnes": float(farm_data["pesticides_tonnes"]),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_yield(farm_data: dict[str, Any]) -> dict[str, float]:
    """Predict crop yield using the persisted Random Forest model."""
    model, scaler, label_encoders = load_prediction_artifacts()
    encoded = encode_features(farm_data, label_encoders)
    scaled = scaler.transform(encoded)
    predicted_value = float(model.predict(scaled)[0])
    return {"predicted_yield": predicted_value}
