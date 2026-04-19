"""Unit tests for the preprocessing and validation module."""

import pytest
from src.preprocess import preprocess_input, validate_input


def test_preprocess_strips_whitespace():
    """Verify that Area and Item fields are stripped of surrounding whitespace."""
    payload = {
        "Area": "  India  ",
        "Item": " Wheat ",
        "average_rain_fall_mm_per_year": "800",
        "avg_temp": "25",
        "pesticides_tonnes": "50",
    }
    result = preprocess_input(payload)
    assert result["Area"] == "India"
    assert result["Item"] == "Wheat"


def test_preprocess_casts_to_numeric():
    """Ensure numeric fields are properly cast to float."""
    payload = {
        "Area": "India",
        "Item": "Rice",
        "average_rain_fall_mm_per_year": "1200",
        "avg_temp": "30",
        "pesticides_tonnes": "25",
    }
    result = preprocess_input(payload)
    assert isinstance(result["average_rain_fall_mm_per_year"], float)
    assert isinstance(result["avg_temp"], float)
    assert isinstance(result["pesticides_tonnes"], float)


def test_validate_input_missing_fields():
    """Ensure ValueError is raised when required fields are missing."""
    with pytest.raises(ValueError, match="Missing required input fields"):
        validate_input({"Area": "India"})


def test_validate_input_negative_rainfall():
    """Ensure ValueError is raised for negative rainfall values."""
    payload = {
        "Area": "India",
        "Item": "Wheat",
        "average_rain_fall_mm_per_year": -100,
        "avg_temp": 25,
        "pesticides_tonnes": 50,
    }
    with pytest.raises(ValueError, match="Rainfall cannot be negative"):
        validate_input(payload)


def test_validate_input_negative_pesticides():
    """Ensure ValueError is raised for negative pesticide usage."""
    payload = {
        "Area": "Brazil",
        "Item": "Maize",
        "average_rain_fall_mm_per_year": 800,
        "avg_temp": 22,
        "pesticides_tonnes": -10,
    }
    with pytest.raises(ValueError, match="Pesticide usage cannot be negative"):
        validate_input(payload)
