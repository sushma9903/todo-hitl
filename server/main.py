import asyncio
import json
import sys
import os

from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import backend functions for tools
from server.backend.data_store import (
    get_weather,
    get_stock_price,
    web_search
)

# create MCP server instance
server = Server("mcp-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    List all tools exposed by this MCP server.
    Only showcase tools are exposed.
    """
    return [
        Tool(
            name="get_weather",
            description="Get real-time weather data for a given city.",
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
        ),

        Tool(
            name="get_stock_price",
            description="Get real stock market data for a given stock symbol.",
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
        ),

        Tool(
            name="web_search",
            description="Search the internet using Google Custom Search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (max 10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Execute the requested MCP tool.
    """
    try:
        if name == "get_weather":
            result = await get_weather(arguments.get("city"))

        elif name == "get_stock_price":
            result = await get_stock_price(arguments.get("symbol"))

        elif name == "web_search":
            result = await web_search(
                arguments.get("query"),
                arguments.get("num_results", 5)
            )

        else:
            result = {
                "status": "error",
                "message": f"Unknown tool: {name}"
            }

        return [
            TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]

    except Exception as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"status": "error", "message": str(e)},
                    indent=2
                )
            )
        ]


async def main():
    """
    Start MCP server over STDIO for MCP Inspector.
    """
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())