# import standard libraries 
import os
import sys
import asyncio
import json
from typing import List, Dict, Any, TypedDict, Annotated, Sequence, Optional, Literal
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
from langgraph.checkpoint.memory import MemorySaver

# import MCP client libraries
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
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

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self.read_stream, self.write_stream = stdio_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.read_stream, self.write_stream)
        )

        await self.session.initialize()

    async def close(self):
        """
        Cleanly shuts down MCP session and server process
        """
        try:
            await self.exit_stack.aclose()
        except Exception as e:
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
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field, create_model
    
    tool_name = tool_schema.name
    tool_description = tool_schema.description
    
    properties = tool_schema.inputSchema.get('properties', {})
    required = tool_schema.inputSchema.get('required', [])
    
    fields = {}
    for param_name, param_info in properties.items():
        json_type = param_info.get('type', 'string')
        param_description = param_info.get('description', '')
        is_required = param_name in required
        
        if json_type == 'string':
            param_type = str
        elif json_type == 'integer':
            param_type = int
        elif json_type == 'number':
            param_type = float
        elif json_type == 'boolean':
            param_type = bool
        else:
            param_type = str
        
        if is_required:
            fields[param_name] = (param_type, Field(..., description=param_description))
        else:
            fields[param_name] = (Optional[param_type], Field(None, description=param_description))
    
    if not fields:
        fields['_dummy'] = (Optional[str], Field(None, description="No parameters required"))
    
    InputModel = create_model(f"{tool_name}Input", **fields)
    
    async def tool_func(**kwargs) -> str:
        """Execute the MCP tool with given arguments"""
        assert mcp_client.session is not None, "MCP session is not initialized"
        
        kwargs = {k: v for k, v in kwargs.items() if k != '_dummy' and v is not None}
        
        try:
            result = await mcp_client.session.call_tool(tool_name, kwargs)
            
            if result.content:
                return result.content[0].text
            return "No response from tool."
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    return StructuredTool(
        name=tool_name,
        description=tool_description,
        args_schema=InputModel,
        coroutine=tool_func
    )


# Enhanced AgentState with todo list
class AgentState(TypedDict):
    """State that gets passed between nodes in the graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    todo_list: Optional[str]


def create_agent_graph(llm_with_tools, tools, available_tool_names: List[str]):
    """
    Creates a LangGraph agent with conditional planning and human-in-the-loop approval.
    Planning only happens when tools are actually needed.
    """
    
    async def call_model(state: AgentState):
        """
        The main agent node that decides whether to call tools or respond directly.
        """
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}
    
    tool_node = ToolNode(tools)
    
    async def generate_plan(state: AgentState):
        """
        Generates a structured execution plan based on the tool calls the agent has decided to make.
        This runs AFTER the agent decides to use tools, not before.
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return {"todo_list": None}
        
        # Extract tool calls
        tool_descriptions = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_descriptions.append(f"- {tool_name} with {tool_args}")
        
        tools_text = "\n".join(tool_descriptions)
        
        # Find the original user query
        user_query = None
        for msg in reversed(messages[:-1]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        
        planning_prompt = f"""User asked: "{user_query}"

Available MCP tools (must be used exactly as named):
{tools_text}

IMPORTANT RULES:
- You MUST refer ONLY to the tools listed above
- DO NOT mention websites, apps, humans, or external sources
- DO NOT suggest manual steps
- Each step must explicitly reference an MCP tool or its output
- This is an internal execution plan, not advice to a user

Write a concise 3-step execution plan in this exact format:
1. Fetch: what data will be retrieved and using which MCP tool
2. Reason: why this MCP tool is required for this query
3. Output: how the MCP tool result will be presented to the user

Tone:
- Professional
- System-oriented
- Third person
- No markdown
- No explanations outside the three steps

Correct example:
1. Fetch current weather data for London using the get_weather MCP tool
2. This tool provides real-time weather data required to answer the user query accurately
3. Present temperature, humidity, and wind conditions in a clear natural language response
"""
        
        planning_messages = [
            SystemMessage(content="You are a planning assistant. Write professional plans in third person without markdown. Follow the example format exactly."),
            HumanMessage(content=planning_prompt)
        ]
        
        # Use a simple LLM call without tool binding for planning
        planning_llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        
        response = await planning_llm.ainvoke(planning_messages)
        # Clean up any markdown that might have slipped through
        cleaned_plan = response.content.replace('**', '').replace('###', '').replace('##', '').strip()
        return {"todo_list": cleaned_plan}
    
    def route_after_agent(state: AgentState) -> Literal["planner", "end"]:
        """
        Routes based on whether the agent decided to use tools.
        If tools are needed, go to planner first. Otherwise, end.
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "planner"
        return "end"
    
    def route_after_planner(state: AgentState) -> Literal["tools"]:
        """
        After planning, always proceed to tools (with interrupt for approval).
        """
        return "tools"
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("planner", generate_plan)
    workflow.add_node("tools", tool_node)
    
    # Set entry point - start with agent
    workflow.set_entry_point("agent")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "planner": "planner",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {"tools": "tools"}
    )
    
    workflow.add_edge("tools", "agent")
    
    # Enable checkpointing and interrupt before tools
    memory = MemorySaver()
    
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["tools"]
    )


async def run_agent():
    """
    Main function with human-in-the-loop approval workflow.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(os.path.dirname(script_dir), "server", "main.py")
    mcp_client = MCPClient(server_script_path=server_path)
    
    try:
        await mcp_client.connect()
        
        mcp_tools = await mcp_client.list_tools()
        tools = [create_langchain_tool_from_mcp(mcp_client, tool) for tool in mcp_tools]
        available_tool_names = [tool.name for tool in tools]
        
        print(f"Loaded {len(tools)} tools from MCP server:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        print()
        
        llm_with_tools = llm.bind_tools(tools)
        agent_graph = create_agent_graph(llm_with_tools, tools, available_tool_names)
        
        system_message = SystemMessage(content="""You are a helpful AI assistant with access to tools.

IMPORTANT INSTRUCTIONS:
- For simple greetings, casual conversation, or general questions, respond directly WITHOUT using any tools
- Use tools when the user asks for real-time, current, latest, or trending information such as weather, stock prices, or news
- Use web_search for any question where the answer can be obtained from the web, including definitions, explanations, background information, or general knowledge.
- Do NOT answer conceptual or informational questions purely from internal knowledge if web_search can be used.
- When you decide to use a tool, use ONLY the tools available: get_weather, get_stock_price, web_search
- NEVER mention or attempt to use tools that don't exist (like brave_search)

Available tools:
- get_weather: for weather information about cities
- get_stock_price: for stock market data
- web_search: for searching the internet

""")

        conversation_history = [system_message]
        config = {"configurable": {"thread_id": "main_conversation"}}
        
        print("MCP AI Agent with Human-in-the-Loop is ready. Type 'exit' to quit.")
        print()

        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            conversation_history.append(HumanMessage(content=user_input))

            if len(conversation_history) > 6:
                conversation_history = conversation_history[-6:]
            
            try:
                state = {
                    "messages": conversation_history,
                    "todo_list": None
                }
                
                result = await agent_graph.ainvoke(state, config)
                last_message = result["messages"][-1]
                
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    if result.get("todo_list"):
                        print()
                        print("Agent's Plan:")
                        print(result["todo_list"])
                        print()
                    
                    print("Planned Tool Calls:")
                    for i, tool_call in enumerate(last_message.tool_calls, 1):
                        print(f"  {i}. {tool_call['name']} with arguments: {tool_call['args']}")
                    
                    approval = input("\nApprove execution? (yes/no): ").strip().lower()
                    
                    if approval in {"yes", "y"}:
                        result = await agent_graph.ainvoke(None, config)
                        final_message = result["messages"][-1]
                        conversation_history = result["messages"]

                        if len(conversation_history) > 6:
                            conversation_history = conversation_history[-6:]
                        
                        print()
                        if hasattr(final_message, "content") and final_message.content:
                            print(f"Agent: {final_message.content}")
                        else:
                            print("Agent: Task completed.")
                    else:
                        print()
                        print("Tool execution cancelled.")
                        conversation_history.append(
                            AIMessage(content="I understand. How else can I help you?")
                        )
                else:
                    final_message = result["messages"][-1]
                    conversation_history = result["messages"]

                    if len(conversation_history) > 6:
                        conversation_history = conversation_history[-6:]
                    
                    if hasattr(final_message, "content"):
                        print(f"Agent: {final_message.content}")
                    else:
                        print(f"Agent: {final_message}")
                        
            except Exception as e:
                print(f"Error: {str(e)}")
                print()
                if conversation_history:
                    conversation_history.pop()

    except Exception as e:
        print(f"Failed to initialize agent: {str(e)}")
    
    finally:
        await mcp_client.close()
        print("MCP connection closed.")

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\nAgent stopped by user.")
    except Exception as e:
        print(f"Fatal error: {str(e)}")