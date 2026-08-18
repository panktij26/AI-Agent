import json
import pytest
from tools import calculate, get_weather, AVAILABLE_TOOLS, TOOLS_SCHEMA

def test_calculate_valid_math():
    response = json.loads(calculate("10 * 5 + 2"))
    assert response.get("result") == 52

def test_calculate_invalid_expression():
    response = json.loads(calculate("invalid_expression ++++"))
    assert "error" in response

def test_get_weather_known_city():
    response = json.loads(get_weather("Delhi"))
    assert response.get("location") == "Delhi"
    assert "temp" in response

def test_tool_definitions():
    assert "calculate" in AVAILABLE_TOOLS
    assert "get_weather" in AVAILABLE_TOOLS
    assert len(TOOLS_SCHEMA) >= 2
