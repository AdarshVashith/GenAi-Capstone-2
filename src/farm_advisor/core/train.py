"""Training script for producing the model artifacts used by the app."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from farm_advisor.config import (
    DATA_DIR,
    LABEL_ENCODERS_PATH,
    MODEL_DIR,
    RF_MODEL_PATH,
    SCALER_PATH,
    TRAINING_DATA_PATH,
)

logger = logging.getLogger(__name__)


TARGET_COLUMN = "hg/ha_yield"
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

# Ensemble weights (GB outperforms RF on cross-validation)
GB_MODEL_PATH = MODEL_DIR / "gb_model.pkl"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
RF_WEIGHT = 0.4
GB_WEIGHT = 0.6


def generate_demo_dataset() -> pd.DataFrame:
    """Create a synthetic but structured dataset for local development."""
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    area_effect = {"India": 0.25, "China": 0.35, "Brazil": 0.15, "USA": 0.3}
    crop_base = {"Wheat": 3.8, "Rice": 4.4, "Maize": 4.1}

    for area, area_bias in area_effect.items():
        for item, crop_bias in crop_base.items():
            for rainfall in [500, 650, 800, 950, 1100]:
                for avg_temp in [18, 22, 25, 28, 31]:
                    for pesticides in [20, 35, 50, 65]:
                        irrigation_bonus = 0.4 if 700 <= rainfall <= 950 else -0.2
                        temperature_penalty = abs(avg_temp - 25) * 0.08
                        pesticide_effect = min(pesticides / 100, 0.35)
                        noise = rng.normal(0, 0.03)
                        yield_value = round(
                            crop_bias
                            + area_bias
                            + irrigation_bonus
                            + pesticide_effect
                            - temperature_penalty
                            + noise,
                            3,
                        )
                        rows.append(
                            {
                                "Area": area,
                                "Item": item,
                                "average_rain_fall_mm_per_year": rainfall,
                                "avg_temp": avg_temp,
                                "pesticides_tonnes": pesticides,
                                TARGET_COLUMN: max(yield_value, 1.2),
                            }
                        )
    return pd.DataFrame(rows)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and domain-knowledge features to improve generalization."""
    working = df.copy()
    working["temp_rainfall"] = (
        working["avg_temp"] * working["average_rain_fall_mm_per_year"]
    )
    working["temp_pesticides"] = working["avg_temp"] * working["pesticides_tonnes"]
    working["rainfall_pesticides"] = (
        working["average_rain_fall_mm_per_year"] * working["pesticides_tonnes"]
    )
    working["optimal_temp_dist"] = abs(working["avg_temp"] - 25)
    return working


def load_training_data(csv_path: Path = TRAINING_DATA_PATH) -> pd.DataFrame:
    """Load training data from CSV, or fall back to a generated demo dataset."""
    if csv_path.exists():
        return pd.read_csv(csv_path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    demo_df = generate_demo_dataset()
    demo_df.to_csv(csv_path, index=False)
    return demo_df


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
    """Encode categorical features, add engineered features, and return model-ready data."""
    working = df.copy()
    label_encoders: dict[str, LabelEncoder] = {}

    for column in ["Area", "Item"]:
        encoder = LabelEncoder()
        working[column] = encoder.fit_transform(working[column].astype(str))
        label_encoders[column] = encoder

    working = engineer_features(working)

    X = working[ALL_FEATURE_COLUMNS]
    y = working[TARGET_COLUMN]
    return X, y, label_encoders  # type: ignore[return-value]


def train_and_save_artifacts(csv_path: Path = TRAINING_DATA_PATH) -> dict[str, float]:
    """Train an ensemble (RF + GradientBoosting) and persist all required artifacts."""
    df = load_training_data(csv_path)
    X, y, label_encoders = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Gradient Boosting (primary model) ──────────────────────────────
    gb_model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=4,
        random_state=42,
    )
    gb_model.fit(X_train_scaled, y_train)

    # ── Random Forest (secondary model) ────────────────────────────────
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=42,
    )
    rf_model.fit(X_train_scaled, y_train)

    # ── Ensemble predictions ───────────────────────────────────────────
    preds_gb = gb_model.predict(X_test_scaled)
    preds_rf = rf_model.predict(X_test_scaled)
    preds_ensemble = RF_WEIGHT * preds_rf + GB_WEIGHT * preds_gb

    # ── Cross-validation on GB (primary) ───────────────────────────────
    X_all_scaled = scaler.fit_transform(X)
    cv_scores = cross_val_score(gb_model, X_all_scaled, y, cv=5, scoring="r2")

    metrics = {
        "ensemble_mae": float(mean_absolute_error(y_test, preds_ensemble)),
        "ensemble_rmse": float(np.sqrt(mean_squared_error(y_test, preds_ensemble))),
        "ensemble_r2": float(r2_score(y_test, preds_ensemble)),
        "gb_mae": float(mean_absolute_error(y_test, preds_gb)),
        "gb_r2": float(r2_score(y_test, preds_gb)),
        "gb_cv_r2_mean": float(cv_scores.mean()),
        "gb_cv_r2_std": float(cv_scores.std()),
        "rf_mae": float(mean_absolute_error(y_test, preds_rf)),
        "rf_r2": float(r2_score(y_test, preds_rf)),
    }

    # Re-fit scaler on full training data for artifact export
    scaler_final = StandardScaler()
    scaler_final.fit(X_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, RF_MODEL_PATH)
    joblib.dump(gb_model, GB_MODEL_PATH)
    joblib.dump(scaler_final, SCALER_PATH)
    joblib.dump(label_encoders, LABEL_ENCODERS_PATH)

    # Persist metrics for the UI
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    """Train and export the artifact files needed by the advisory app."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    metrics = train_and_save_artifacts()
    logger.info("Saved RF model to %s", RF_MODEL_PATH)
    logger.info("Saved GB model to %s", GB_MODEL_PATH)
    logger.info("Saved scaler to %s", SCALER_PATH)
    logger.info("Saved label encoders to %s", LABEL_ENCODERS_PATH)
    logger.info("Saved metrics to %s", METRICS_PATH)
    logger.info("─── Ensemble Metrics ───")
    logger.info("  MAE:  %.6f", metrics["ensemble_mae"])
    logger.info("  RMSE: %.6f", metrics["ensemble_rmse"])
    logger.info("  R²:   %.6f", metrics["ensemble_r2"])
    logger.info("─── Cross-Validation (GB, 5-fold) ───")
    logger.info(
        "  R²:   %.6f ± %.6f", metrics["gb_cv_r2_mean"], metrics["gb_cv_r2_std"]
    )


if __name__ == "__main__":
    main()
