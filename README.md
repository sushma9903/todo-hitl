---

# 🤖 MCP AI Agent with LangGraph

An intelligent AI agent that connects to **Model Context Protocol (MCP)** servers, dynamically discovers tools, and maintains conversation memory using **LangGraph**.

---

## 📋 Overview

This project implements an AI agent that:

* Connects to MCP servers via **STDIO transport**
* Dynamically discovers and invokes tools based on user intent
* Maintains full conversation history and context
* Uses **LangGraph** for explicit state management
* Supports weather queries, stock prices, and web search

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────┐
│                          User                             │
│              (Natural Language Queries)                   │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                  LangGraph AI Agent                       │
│                       (agent.py)                          │
│                                                           │
│   ┌───────────────────────────────────────────────────┐   │
│   │                  Agent Node (LLM)                 │   │
│   │                                                   │   │
│   │ • Interprets user intent                          │   │
│   │ • Uses full conversation history                  │   │
│   │ • Decides whether a tool call is required         │   │
│   └───────────────────────┬───────────────────────────┘   │
│                           │                               │
│   ┌───────────────────────▼───────────────────────────┐   │
│   │                 Tool Node (Executors)             │   │
│   │                                                   │   │
│   │ • Executes selected tools                         │   │
│   │ • Validates inputs using schemas                  │   │
│   │ • Returns results back to the agent               │   │
│   └───────────────────────┬───────────────────────────┘   │
└───────────────────────────┼───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                    MCP Client (STDIO)                     │
│                                                           │
│ • Spawns MCP server as a subprocess                       │
│ • Discovers available tools dynamically                   │
│ • Sends and receives MCP protocol messages                │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                      MCP Server                           │
│                    (server/main.py)                       │
│                                                           │
│ • get_weather                                             │
│ • get_stock_price                                         │
│ • web_search                                              │
│                                                           │
│ Exposes tools via MCP with JSON schemas                   │
│ Contains no agent or decision logic                       │
└───────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

* Python **3.9+**
* Groq API key
* MCP server (included in `server/main.py`)

---

### Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up environment variables:

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Run the agent:

```bash
python agent/agent.py
```

---

## 📦 Dependencies

* `langchain-groq>=0.2.0`
* `langchain>=0.3.0`
* `langgraph>=0.2.0`
* `python-dotenv>=1.0.0`
* `mcp>=1.0.0`
* `pydantic>=2.0.0`

Install all at once:

```bash
pip install langchain-groq langchain langgraph python-dotenv mcp pydantic
```

---

## 🎯 Usage Examples

### Basic Queries

```
You: What is the weather in Paris?
Agent: The current weather in Paris is a clear sky with a temperature of 3.87°C, humidity of 73%, and wind speed of 4.12 m/s.
```

```
You: What is the stock price of TSLA?
Agent: The current stock price of TSLA is $485.4.
```

```
You: Search the web for latest AI news
Agent: Based on the search results, here are the latest AI-related updates...
```

---

### Context-Aware Conversations

```
You: What is the weather in London?
You: What about Tokyo?
You: What about the previous city?
Agent: London.
```

---

### Memory Recall

```
You: What questions have I asked you?
Agent: You have asked the following questions:
1. What is the weather in Paris?
2. What is the stock price of TSLA?
3. Search the web for latest AI news
...
```

---

## 🔧 Technical Details

### Component Breakdown

#### 1. MCPClient

* Manages STDIO connection to MCP server
* Spawns server process as a subprocess
* Discovers available tools dynamically via MCP protocol

#### 2. Tool Conversion

* Converts MCP tool schemas to LangChain `StructuredTools`
* Maps JSON schema types to Python types
* Uses Pydantic models for runtime validation
* Executes tools asynchronously

#### 3. LangGraph Agent

* `StateGraph` manages agent execution flow
* Agent node performs reasoning using full conversation history
* Tool node executes MCP tools when required
* Conditional routing prevents infinite loops

#### 4. Memory System

* Maintains full conversation history in memory
* Passes entire message history to the LLM each turn
* Enables contextual understanding and recall
* No external database required

---

## 🛠️ Customization

### Adding New Tools

1. Add a tool to the MCP server (`server/main.py`)
2. Restart the agent

The agent will automatically discover the new tool.

---

### Changing LLM Model

```python
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-70b-versatile",
    temperature=0
)
```

---

### Adjusting System Prompt

Modify the `system_message` inside `run_agent()`:

```python
system_message = SystemMessage(content="""
Your custom instructions here...
""")
```

---

## 🧪 Testing

### Suggested Test Prompts

**Tool Invocation**

* What is the weather in Paris?
* What is the stock price of AAPL?
* Search the web for Python 3.12 features

**Memory & Context**

* What is the weather in London?
* What about Tokyo?
* What was the temperature in the previous city?

---

## 🐛 Troubleshooting and Notes

* **`GROQ_API_KEY not found`**
  Ensure a `.env` file exists in the project root with a valid `GROQ_API_KEY`, and restart the terminal.

* **`MCP session is not initialized`**
  Indicates the MCP server did not start correctly. Verify `server/main.py` exists and the client connects before discovering or invoking tools.

* **Agent looping or stopping unexpectedly (ReAct)**
  While using ReAct, the agent sometimes re-invoked tools due to dynamic MCP schemas. This was resolved by switching to **LangGraph**, which provides explicit state transitions and reliable termination.

* **Tool input validation errors**
  Occurred when generated tool inputs did not exactly match MCP schemas. Fixed by generating schema-driven tools directly from MCP definitions.

* **Follow-up questions not using context**
  Initially, conversation history was stored but not reasoned over. Using **LangGraph state** enabled correct handling of contextual and memory-based queries.

---

## 📚 Key Concepts

This project is intentionally designed with a clear separation of responsibilities:

- **MCP Server** acts as a stable tool layer  
  It exposes real-world capabilities (weather, stocks, web search) without embedding any AI logic.

- **AI Agent (LangGraph)** handles reasoning and decision-making  
  The agent interprets user intent, decides whether a tool is required, and produces the final response.

- **LangGraph State Management** ensures predictable execution  
  Using an explicit graph prevents common agent issues such as repeated tool calls or infinite loops.

- **Conversation Memory** enables contextual understanding  
  The agent retains full conversation history, allowing follow-up questions and memory-based responses.

This separation makes the system easier to extend, debug, and reason about compared to tightly coupled agent-tool implementations.

---
## 📖 References

- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangChain Agents](https://docs.langchain.com/oss/javascript/langchain/agents)
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)
  
---

## 👤 Author

Built as part of an MCP integration learning project.

---
