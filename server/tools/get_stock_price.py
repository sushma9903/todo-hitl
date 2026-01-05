from mcp.types import Tool
from server.backend.data_store import get_stock_price

def get_stock_price_tool():
    return Tool(
        name="get_stock_price",
        description="Get real stock market price data",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol (e.g., AAPL, TSLA)"
                }
            },
            "required": ["symbol"]
        }
    )

async def run_get_stock_price(arguments: dict):
    return await get_stock_price(arguments["symbol"])
