from mcp.types import Tool
from server.backend.data_store import get_weather

def get_weather_tool():
    return Tool(
        name="get_weather",
        description="Get real-time weather data for a city",
        inputSchema={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g., Bangalore, London)"
                }
            },
            "required": ["city"]
        }
    )

async def run_get_weather(arguments: dict):
    return await get_weather(arguments["city"])
