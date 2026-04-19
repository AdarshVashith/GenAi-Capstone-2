"""Prediction helpers for the crop yield model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import logging
import warnings
import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from src.preprocess import preprocess_input

# Catch scikit-learn version warnings when loading the older model
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from config import LABEL_ENCODERS_PATH, RF_MODEL_PATH, SCALER_PATH

logger = logging.getLogger(__name__)

BASE_FEATURE_COLUMNS = [
    "Area",
    "Item",
    "average_rain_fall_mm_per_year",
    "avg_temp",
    "pesticides_tonnes",
]
ENGINEERED_FEATURE_COLUMNS = [
    "temp_rainfall",
    "temp_pesticides",
    "rainfall_pesticides",
    "optimal_temp_dist",
]
ALL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

# Ensemble weights — must match src/train.py
RF_WEIGHT = 0.4
GB_WEIGHT = 0.6

# Path for GB model (mirrors train.py)
GB_MODEL_PATH = RF_MODEL_PATH.parent / "gb_model.pkl"


def _patch_rf_model(model: Any) -> None:
    """Patch for scikit-learn >= 1.4 compatibility with older models."""
    if hasattr(model, "estimators_"):
        for estimator in model.estimators_:
            if not hasattr(estimator, "monotonic_cst"):
                estimator.monotonic_cst = None


def load_prediction_artifacts() -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load model artifacts required for inference.

    Returns (rf_model, gb_model, scaler, label_encoders).
    If gb_model is not found, falls back to RF-only mode.
    """
    required_paths = [RF_MODEL_PATH, SCALER_PATH, LABEL_ENCODERS_PATH]
    missing = [str(path) for path in required_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Add these files before running predictions: "
            + ", ".join(missing)
        )

    rf_model = joblib.load(RF_MODEL_PATH)
    _patch_rf_model(rf_model)

    gb_model = None
    if GB_MODEL_PATH.exists():
        gb_model = joblib.load(GB_MODEL_PATH)
        logger.debug("Loaded ensemble models (RF + GB).")
    else:
        logger.warning("GB model not found at %s; using RF-only mode.", GB_MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    return rf_model, gb_model, scaler, label_encoders


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and domain-knowledge features (must match train.py)."""
    working = df.copy()
    working["temp_rainfall"] = working["avg_temp"] * working["average_rain_fall_mm_per_year"]
    working["temp_pesticides"] = working["avg_temp"] * working["pesticides_tonnes"]
    working["rainfall_pesticides"] = working["average_rain_fall_mm_per_year"] * working["pesticides_tonnes"]
    working["optimal_temp_dist"] = abs(working["avg_temp"] - 25)
    return working


def encode_features(farm_data: dict[str, Any], label_encoders: dict[str, Any]) -> pd.DataFrame:
    """Encode categorical fields and add engineered features."""
    row = {
        "Area": label_encoders["Area"].transform([farm_data["Area"]])[0],
        "Item": label_encoders["Item"].transform([farm_data["Item"]])[0],
        "average_rain_fall_mm_per_year": float(farm_data["average_rain_fall_mm_per_year"]),
        "avg_temp": float(farm_data["avg_temp"]),
        "pesticides_tonnes": float(farm_data["pesticides_tonnes"]),
    }
    base_df = pd.DataFrame([row], columns=BASE_FEATURE_COLUMNS)
    return engineer_features(base_df)[ALL_FEATURE_COLUMNS]


def predict_yield(farm_data: dict[str, Any]) -> dict[str, float]:
    """Predict crop yield using the persisted ensemble (RF + GB) model."""
    cleaned_data = preprocess_input(farm_data)
    rf_model, gb_model, scaler, label_encoders = load_prediction_artifacts()
    encoded = encode_features(cleaned_data, label_encoders)
    scaled = scaler.transform(encoded)

    pred_rf = float(rf_model.predict(scaled)[0])

    if gb_model is not None:
        pred_gb = float(gb_model.predict(scaled)[0])
        predicted_value = RF_WEIGHT * pred_rf + GB_WEIGHT * pred_gb
    else:
        predicted_value = pred_rf

    logger.info(
        "Predicted yield %.4f for crop=%s area=%s",
        predicted_value,
        farm_data.get("Item"),
        farm_data.get("Area"),
    )
    return {"predicted_yield": predicted_value}
