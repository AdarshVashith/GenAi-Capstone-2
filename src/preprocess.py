"""Data preprocessing utilities."""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)

REQUIRED_FIELDS = [
    "Area", "Item", "average_rain_fall_mm_per_year", "avg_temp", "pesticides_tonnes"
]


def validate_input(payload: dict[str, Any]) -> None:
    """Raise ValueError if required fields are missing or invalid."""
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"Missing required input fields: {', '.join(missing)}")

    if float(payload["average_rain_fall_mm_per_year"]) < 0:
        raise ValueError("Rainfall cannot be negative.")
    if float(payload["pesticides_tonnes"]) < 0:
        raise ValueError("Pesticide usage cannot be negative.")


def preprocess_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize incoming payloads before model inference."""
    validate_input(payload)
    normalized = dict(payload)
    normalized["Area"] = str(normalized["Area"]).strip()
    normalized["Item"] = str(normalized["Item"]).strip()
    normalized["average_rain_fall_mm_per_year"] = float(normalized["average_rain_fall_mm_per_year"])
    normalized["avg_temp"] = float(normalized["avg_temp"])
    normalized["pesticides_tonnes"] = float(normalized["pesticides_tonnes"])
    logger.debug("Preprocessed input for Area=%s, Item=%s", normalized["Area"], normalized["Item"])
    return normalized

