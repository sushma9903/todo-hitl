# import standard libraries 
import os
import sys
import asyncio
import json
from typing import List, Dict, Any, TypedDict, Annotated, Sequence, Optional
import operator

# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# fetch Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# import LangChain and Groq related libraries
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# import LangGraph for proper agent orchestration
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# import MCP client libraries
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)


class MCPClient:
    """
    Handles connection to an MCP server over STDIO
    and exposes tool discovery and execution
    """
    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.read_stream = None
        self.write_stream = None

    async def connect(self):
        """
        Starts the MCP server process and opens a client session.
        """
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script_path],
            env=None
        )

        # CRITICAL FIX: Properly handle the stdio_client context manager
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self.read_stream, self.write_stream = stdio_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.read_stream, self.write_stream)
        )

        # CRITICAL FIX: Initialize session after ensuring connection is stable
        await self.session.initialize()

    async def close(self):
        """
        Cleanly shuts down MCP session and server process
        """
        try:
            await self.exit_stack.aclose()
        except Exception as e:
            # suppress cleanup errors on Windows
            pass

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        Fetches the list of available tools from the MCP server.
        """
        assert self.session is not None, "MCP session is not initialized"
        response = await self.session.list_tools()
        return response.tools


def create_langchain_tool_from_mcp(mcp_client: MCPClient, tool_schema: Any):
    """
    Creates a LangChain-compatible tool function from MCP tool schema.
    This uses the new function-calling format that LangGraph expects.
    FIXED: Properly handles type mapping from MCP schema to Pydantic types.
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field, create_model
    
    tool_name = tool_schema.name
    tool_description = tool_schema.description
    
    # extract parameters from MCP schema
    properties = tool_schema.inputSchema.get('properties', {})
    required = tool_schema.inputSchema.get('required', [])
    
    # dynamically create Pydantic model for input validation
    fields = {}
    for param_name, param_info in properties.items():
        json_type = param_info.get('type', 'string')
        param_description = param_info.get('description', '')
        is_required = param_name in required
        
        # map JSON schema types to Python types
        if json_type == 'string':
            param_type = str
        elif json_type == 'integer':
            param_type = int
        elif json_type == 'number':
            param_type = float
        elif json_type == 'boolean':
            param_type = bool
        else:
            param_type = str  # default fallback
        
        # create field with proper type and optionality
        if is_required:
            fields[param_name] = (param_type, Field(..., description=param_description))
        else:
            fields[param_name] = (Optional[param_type], Field(None, description=param_description))
    
    # handle case where no parameters are defined
    if not fields:
        fields['_dummy'] = (Optional[str], Field(None, description="No parameters required"))
    
    InputModel = create_model(f"{tool_name}Input", **fields)
    
    # create async tool function
    async def tool_func(**kwargs) -> str:
        """Execute the MCP tool with given arguments"""
        assert mcp_client.session is not None, "MCP session is not initialized"
        
        # remove dummy parameter if present
        kwargs = {k: v for k, v in kwargs.items() if k != '_dummy' and v is not None}
        
        try:
            result = await mcp_client.session.call_tool(tool_name, kwargs)
            
            if result.content:
                return result.content[0].text
            return "No response from tool."
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    # return structured tool
    return StructuredTool(
        name=tool_name,
        description=tool_description,
        args_schema=InputModel,
        coroutine=tool_func
    )


# define the state for our agent graph
class AgentState(TypedDict):
    """State that gets passed between nodes in the graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]


def create_agent_graph(llm_with_tools, tools):
    """
    Creates a LangGraph agent that properly handles tool calls and memory.
    This uses the ReAct pattern but with proper state management.
    """
    
    # define the function that calls the model
    async def call_model(state: AgentState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}
    
    # define the tool execution node
    tool_node = ToolNode(tools)
    
    # define the routing logic
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        # if there are no tool calls, end the workflow
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return "end"
        return "continue"
    
    # build the graph
    workflow = StateGraph(AgentState)
    
    # add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # set entry point
    workflow.set_entry_point("agent")
    
    # add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    # add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


async def run_agent():
    """
    Main function that connects to MCP server and runs the interactive agent loop.
    """
    # 1. connect to MCP server
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(os.path.dirname(script_dir), "server", "main.py")
    mcp_client = MCPClient(server_script_path=server_path)
    
    try:
        await mcp_client.connect()
        
        # 2. fetch MCP tools
        mcp_tools = await mcp_client.list_tools()
        
        # 3. convert MCP tools to LangChain tools (new format)
        tools = [create_langchain_tool_from_mcp(mcp_client, tool) for tool in mcp_tools]
        
        # Debug: Print available tools
        print(f"Loaded {len(tools)} tools from MCP server:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        print()
        
        # 4. bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)
        
        # 5. create agent graph
        agent_graph = create_agent_graph(llm_with_tools, tools)
        
        # 6. create system message with instructions
        system_message = SystemMessage(content="""You are a helpful AI assistant with access to tools.

IMPORTANT: When the user asks about:
- Weather in a city → ONLY use get_weather tool
- Stock prices for a symbol → ONLY use get_stock_price tool
- Web search or latest news → ONLY use web_search tool

Guidelines:
- Use a tool ONLY when the user request is clear, specific, and sufficient to use that tool correctly
- If required information is missing, ambiguous, or unclear, DO NOT guess and DO NOT call any tool
- In such cases, respond politely that you do not have enough information to answer accurately
- Never fetch random or loosely related data just to provide an answer
- Use the most appropriate tool when a tool is genuinely required
- Do not combine tools unless explicitly required
- If a tool call fails, retry once

Behavior rules:
- For general questions (math, jokes, greetings), answer directly without tools
- For internal, private, or organization-specific information (e.g., company policies, cafeteria menus without context), do NOT use web_search
- If information is not publicly verifiable or clearly specified, say you do not have enough information

When the user asks about previous conversations, refer to the message history.

Always prioritize correctness and honesty over providing a speculative answer.""")

        
        # 7. maintain conversation history
        conversation_history = [system_message]
        
        # 8. interactive loop
        print("🤖 MCP AI Agent is ready! Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Add user message to history
            conversation_history.append(HumanMessage(content=user_input))
            
            # Invoke agent with full conversation history
            try:
                response = await agent_graph.ainvoke(
                    {"messages": conversation_history}
                )

            except Exception as e:
                # Retry ONCE for transient tool-call failures
                response = await agent_graph.ainvoke(
                    {"messages": conversation_history}
                )

            # Extract final response
            final_message = response["messages"][-1]

            # Save to memory
            conversation_history.append(final_message)

            # Print response
            print(f"\nAgent: {final_message.content}\n")


    except Exception as e:
        print(f"Failed to initialize agent: {str(e)}")
        
    finally:
        # 9. cleanup
        await mcp_client.close()
        print("MCP connection closed.")


if __name__ == "__main__":
    # CRITICAL FIX: Windows-specific event loop policy
    if sys.platform.startswith('win'):
        # Use ProactorEventLoop for better subprocess handling on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\nAgent stopped by user.")
    except Exception as e:
        print(f"Fatal error: {str(e)}")