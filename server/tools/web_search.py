from mcp.types import Tool
from server.backend.data_store import web_search


def web_search_tool() -> Tool:
    """
    MCP Tool definition for web search using Google Custom Search.
    """
    return Tool(
        name="web_search",
        description=(
            "Search the internet using Google Custom Search. "
            "Returns a list of relevant webpages with title, snippet, and link."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'latest AI agent frameworks')"
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


async def run_web_search(arguments: dict):
    """
    Executes the web search tool.
    """
    query = arguments.get("query")
    num_results = arguments.get("num_results", 5)

    return await web_search(query=query, num_results=num_results)
