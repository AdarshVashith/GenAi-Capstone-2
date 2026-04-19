"""Unit tests for prediction module."""

import pytest
from src.predict import predict_yield, FEATURE_COLUMNS


def test_predict_yield_mocked(mocker):
    """Test that predict_yield correctly encodes and scales before prediction."""
    # Mock artifacts
    mock_model = mocker.Mock()
    mock_model.predict.return_value = [4.25]
    
    mock_scaler = mocker.Mock()
    mock_scaler.transform.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    
    mock_label_encoder_area = mocker.Mock()
    mock_label_encoder_area.transform.return_value = [1]
    
    mock_label_encoder_item = mocker.Mock()
    mock_label_encoder_item.transform.return_value = [2]
    
    mock_encoders = {
        "Area": mock_label_encoder_area,
        "Item": mock_label_encoder_item
    }
    
    mocker.patch("src.predict.load_prediction_artifacts", return_value=(mock_model, mock_scaler, mock_encoders))
    
    sample_data = {
        "Area": "India",
        "Item": "Wheat",
        "average_rain_fall_mm_per_year": "1000.0",
        "avg_temp": "23.5",
        "pesticides_tonnes": "10.0"
    }
    
    result = predict_yield(sample_data)
    
    assert result == {"predicted_yield": 4.25}
    mock_model.predict.assert_called_once()
    mock_scaler.transform.assert_called_once()
    
    # Assert encoders were called with correct fields
    mock_label_encoder_area.transform.assert_called_with(["India"])
    mock_label_encoder_item.transform.assert_called_with(["Wheat"])
