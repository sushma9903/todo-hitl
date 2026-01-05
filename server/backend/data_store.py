"""
Backend data layer for MCP server.

This module contains real external integrations used by MCP tools:
- Weather data (OpenWeatherMap)
- Stock market data (Stooq)
- Web search (Google Custom Search)
"""

import os
import httpx
import csv
from io import StringIO
from dotenv import load_dotenv

# load environment variables
load_dotenv()


# -------------------------------------------------
# Weather data using OpenWeatherMap
# -------------------------------------------------

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

async def get_weather(city: str):
    """
    Get current weather data for a city using OpenWeatherMap.
    """
    if not OPENWEATHER_API_KEY:
        return {"error": "OpenWeather API key not configured"}

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            return {
                "city": data.get("name"),
                "country": data.get("sys", {}).get("country"),
                "temperature": data.get("main", {}).get("temp"),
                "feels_like": data.get("main", {}).get("feels_like"),
                "humidity": data.get("main", {}).get("humidity"),
                "pressure": data.get("main", {}).get("pressure"),
                "weather": data.get("weather", [{}])[0].get("description"),
                "wind_speed": data.get("wind", {}).get("speed")
            }

        except httpx.HTTPStatusError as e:
            return {
                "error": "Failed to fetch weather data",
                "details": str(e)
            }


# -------------------------------------------------
# Stock market data using Stooq (no API key)
# -------------------------------------------------

async def get_stock_price(symbol: str):
    """
    Get stock price data from Stooq.
    Automatically normalizes symbols to '<ticker>.us'.
    """
    symbol = symbol.lower()
    if "." not in symbol:
        symbol = f"{symbol}.us"

    url = f"https://stooq.com/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv"

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url)

        reader = csv.DictReader(StringIO(response.text))
        data = next(reader, None)

        if not data or data["Close"] == "N/D":
            return {"error": f"No data found for symbol {symbol}"}

        return {
            "symbol": symbol.upper(),
            "open": data["Open"],
            "high": data["High"],
            "low": data["Low"],
            "close": data["Close"],
            "volume": data["Volume"]
        }


# -------------------------------------------------
# Web search using Google Custom Search (free tier)
# -------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

async def web_search(query: str, num_results: int = 5):
    """
    Perform web search using Google Custom Search API.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return {"error": "Google search not configured"}

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": min(num_results, 10)
    }

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = [
                {
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "link": item.get("link")
                }
                for item in data.get("items", [])
            ]

            return {
                "query": query,
                "results": results
            }

        except httpx.HTTPStatusError as e:
            return {
                "error": "Web search failed",
                "details": str(e)
            }
