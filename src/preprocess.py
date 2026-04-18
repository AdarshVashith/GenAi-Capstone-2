"""Data preprocessing utilities."""

from __future__ import annotations

from typing import Any


def preprocess_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize incoming payloads before model inference."""
    normalized = dict(payload)
    normalized["Area"] = str(normalized["Area"]).strip()
    normalized["Item"] = str(normalized["Item"]).strip()
    normalized["average_rain_fall_mm_per_year"] = float(normalized["average_rain_fall_mm_per_year"])
    normalized["avg_temp"] = float(normalized["avg_temp"])
    normalized["pesticides_tonnes"] = float(normalized["pesticides_tonnes"])
    return normalized
