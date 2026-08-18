import json
import math

def calculate(expression: str) -> str:
    """Evaluates a basic mathematical expression safely."""
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": f"Invalid expression: {str(e)}"})

def get_weather(location: str) -> str:
    """Mock tool to fetch current weather for a city."""
    # In production, connect this to OpenWeatherMap or another API
    mock_db = {
        "london": {"temp": "15°C", "condition": "Cloudy"},
        "new york": {"temp": "22°C", "condition": "Sunny"},
        "delhi": {"temp": "34°C", "condition": "Hot & Clear"},
    }
    city = location.strip().lower()
    data = mock_db.get(city, {"temp": "20°C", "condition": "Partly Cloudy"})
    return json.dumps({"location": location, **data})

# OpenAI Tool Schemas
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations and evaluations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g., '12 * 45 + sqrt(144)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather updates for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city or location name.",
                    }
                },
                "required": ["location"],
            },
        },
    },
]

# Mapping tool names to executable functions
AVAILABLE_TOOLS = {
    "calculate": calculate,
    "get_weather": get_weather,
}
