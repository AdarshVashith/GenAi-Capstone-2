"""Training script for producing the model artifacts used by the app."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import DATA_DIR, LABEL_ENCODERS_PATH, MODEL_DIR, RF_MODEL_PATH, SCALER_PATH, TRAINING_DATA_PATH


TARGET_COLUMN = "hg/ha_yield"
FEATURE_COLUMNS = [
    "Area",
    "Item",
    "average_rain_fall_mm_per_year",
    "avg_temp",
    "pesticides_tonnes",
]


def generate_demo_dataset() -> pd.DataFrame:
    """Create a synthetic but structured dataset for local development."""
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
                        yield_value = round(
                            crop_bias
                            + area_bias
                            + irrigation_bonus
                            + pesticide_effect
                            - temperature_penalty,
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


def load_training_data(csv_path: Path = TRAINING_DATA_PATH) -> pd.DataFrame:
    """Load training data from CSV, or fall back to a generated demo dataset."""
    if csv_path.exists():
        return pd.read_csv(csv_path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    demo_df = generate_demo_dataset()
    demo_df.to_csv(csv_path, index=False)
    return demo_df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
    """Encode categorical features and return model-ready data."""
    working = df.copy()
    label_encoders: dict[str, LabelEncoder] = {}

    for column in ["Area", "Item"]:
        encoder = LabelEncoder()
        working[column] = encoder.fit_transform(working[column].astype(str))
        label_encoders[column] = encoder

    X = working[FEATURE_COLUMNS]
    y = working[TARGET_COLUMN]
    return X, y, label_encoders


def train_and_save_artifacts(csv_path: Path = TRAINING_DATA_PATH) -> dict[str, float]:
    """Train the Random Forest model and persist all required artifacts."""
    df = load_training_data(csv_path)
    X, y, label_encoders = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_split=4,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RF_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
    return metrics


def main() -> None:
    """Train and export the artifact files needed by the advisory app."""
    metrics = train_and_save_artifacts()
    print(f"Saved model to {RF_MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved label encoders to {LABEL_ENCODERS_PATH}")
    print(f"Validation MAE: {metrics['mae']:.4f}")
    print(f"Validation R2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
