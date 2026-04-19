"""Unit tests for farm agent logic."""

import pytest
from agent.farm_agent import assess_risk_node


def test_assess_risk_node_low_risk():
    """Test low risk classification above 4.0."""
    state = {
        "farm_data": {},
        "yield_prediction": {"predicted_yield": 4.5},
        "retrieved_docs": [],
        "final_report": "",
    }
    result = assess_risk_node(state)
    assert result["yield_prediction"]["risk_level"] == "Low"


def test_assess_risk_node_medium_risk():
    """Test medium risk classification between 2.5 and 4.0."""
    state = {
        "farm_data": {},
        "yield_prediction": {"predicted_yield": 3.0},
        "retrieved_docs": [],
        "final_report": "",
    }
    result = assess_risk_node(state)
    assert result["yield_prediction"]["risk_level"] == "Medium"


def test_assess_risk_node_high_risk():
    """Test high risk classification below 2.5."""
    state = {
        "farm_data": {},
        "yield_prediction": {"predicted_yield": 2.0},
        "retrieved_docs": [],
        "final_report": "",
    }
    result = assess_risk_node(state)
    assert result["yield_prediction"]["risk_level"] == "High"
