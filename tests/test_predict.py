"""Unit tests for prediction module."""

from farm_advisor.core.predict import predict_yield


def test_predict_yield_mocked(mocker):
    """Test that predict_yield correctly encodes and scales before prediction."""
    # Mock artifacts
    mock_rf_model = mocker.Mock()
    mock_rf_model.predict.return_value = [4.25]

    mock_gb_model = mocker.Mock()
    mock_gb_model.predict.return_value = [4.30]

    mock_scaler = mocker.Mock()
    # 9 features after engineering (5 base + 4 engineered)
    mock_scaler.transform.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

    mock_label_encoder_area = mocker.Mock()
    mock_label_encoder_area.transform.return_value = [1]

    mock_label_encoder_item = mocker.Mock()
    mock_label_encoder_item.transform.return_value = [2]

    mock_encoders = {"Area": mock_label_encoder_area, "Item": mock_label_encoder_item}

    mocker.patch(
        "farm_advisor.core.predict.load_prediction_artifacts",
        return_value=(mock_rf_model, mock_gb_model, mock_scaler, mock_encoders),
    )

    sample_data = {
        "Area": "India",
        "Item": "Wheat",
        "average_rain_fall_mm_per_year": "1000.0",
        "avg_temp": "23.5",
        "pesticides_tonnes": "10.0",
    }

    result = predict_yield(sample_data)

    # Ensemble: 0.4 * 4.25 + 0.6 * 4.30 = 1.70 + 2.58 = 4.28
    expected_yield = 0.4 * 4.25 + 0.6 * 4.30
    assert result == {"predicted_yield": expected_yield}
    mock_rf_model.predict.assert_called_once()
    mock_gb_model.predict.assert_called_once()
    mock_scaler.transform.assert_called_once()

    # Assert encoders were called with correct fields
    mock_label_encoder_area.transform.assert_called_with(["India"])
    mock_label_encoder_item.transform.assert_called_with(["Wheat"])


def test_predict_yield_rf_only_fallback(mocker):
    """Test graceful fallback to RF-only when GB model is absent."""
    mock_rf_model = mocker.Mock()
    mock_rf_model.predict.return_value = [3.90]

    mock_scaler = mocker.Mock()
    mock_scaler.transform.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

    mock_label_encoder_area = mocker.Mock()
    mock_label_encoder_area.transform.return_value = [0]
    mock_label_encoder_item = mocker.Mock()
    mock_label_encoder_item.transform.return_value = [1]
    mock_encoders = {"Area": mock_label_encoder_area, "Item": mock_label_encoder_item}

    # gb_model is None (not found)
    mocker.patch(
        "farm_advisor.core.predict.load_prediction_artifacts",
        return_value=(mock_rf_model, None, mock_scaler, mock_encoders),
    )

    result = predict_yield(
        {
            "Area": "Brazil",
            "Item": "Rice",
            "average_rain_fall_mm_per_year": "900",
            "avg_temp": "28",
            "pesticides_tonnes": "30",
        }
    )

    assert result == {"predicted_yield": 3.90}
    mock_rf_model.predict.assert_called_once()
