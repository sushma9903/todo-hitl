---

# 🤖 MCP AI Agent with Todo Planning & Human-in-the-Loop (LangGraph)

## 📌 Overview

This project extends an existing **MCP + LangGraph AI agent** by adding:

* **Structured Todo-style execution planning**
* **Human-in-the-Loop (HITL) approval before tool execution**
* **Deterministic, explainable control flow using LangGraph**

The goal of this task is to ensure that **every tool invocation is deliberate, transparent, and human-approved**, making the agent safer, more controllable, and production-ready.

This work builds directly on **Task 2**, which implemented a stable MCP-based AI agent with dynamic tool discovery and conversational memory.

---

## 🎯 Objectives of This Task

The assignment required the following:

1. **Add a Todo List tool**
   Generate a structured list of steps describing how the agent plans to solve the user’s query *before* executing tools.

2. **Add Human-in-the-Loop control**
   Pause execution and request explicit user approval before running any tool.
   *(Bonus: request approval after plan generation and before tool execution.)*

3. **Evaluate LangChain middleware**
   Investigate `TodoListMiddleware` and Human-in-the-Loop middleware as referenced in the assignment.

This repository demonstrates **all three**, with clear justification for architectural decisions.

---

## 🧠 High-Level Architecture

```
┌───────────────────────────────────────────────────────────┐
│                          User                             │
│              (Natural Language Queries)                   │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                  LangGraph AI Agent                       │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Agent Node (LLM)                                     │ │
│  │ • Interprets user intent                             │ │
│  │ • Decides whether tools are required                 │ │
│  └───────────────┬─────────────────────────────────────┘ │
│                  │                                       │
│  ┌───────────────▼─────────────────────────────────────┐ │
│  │ Planner Node (Todo Generation)                        │ │
│  │ • Generates a structured execution plan               │ │
│  │ • Explains what tools will be used and why            │ │
│  └───────────────┬─────────────────────────────────────┘ │
│                  │   (Human approval required)           │
│  ┌───────────────▼─────────────────────────────────────┐ │
│  │ Tool Node (MCP Executors)                             │ │
│  │ • Executes approved MCP tools                         │ │
│  │ • Returns results to the agent                        │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────┼─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                    MCP Client (STDIO)                     │
│ • Discovers tools dynamically                             │
│ • Sends tool calls via MCP protocol                       │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                      MCP Server                           │
│ • get_weather                                             │
│ • get_stock_price                                         │
│ • web_search                                              │
│ • No agent logic                                          │
└───────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
task4-todo-hitl/
├── agent/
│   ├── ai-agent.py          # Baseline agent (Task 2)
│   ├── agent-custom.py      # Final working Todo + HITL agent
│   └── agent-prebuilt.py    # Middleware investigation (documented limitation)
│
├── server/
│   ├── backend/
│   ├── tools/
│   └── main.py              # MCP tool server
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🤖 Agent Variants Explained

### 1. `ai-agent.py` — Baseline (Task 2)

This is the **original MCP + LangGraph agent** developed in Task 2.

Features:

* Dynamic MCP tool discovery
* Explicit LangGraph state management
* Conversational memory
* Tool invocation without planning or approval

This file serves as a **reference baseline** for comparison.

---

### 2. `agent-custom.py` — Final Submission (Todo + HITL)

This is the **primary and final implementation** for this task.

It introduces:

* **Todo-style execution planning**

  * Generated only when tools are required
  * Explains *what data will be fetched*, *why*, and *how it will be presented*

* **Human-in-the-Loop approval**

  * Execution pauses after plan generation
  * User must explicitly approve before tools are called
  * Tool execution can be cancelled safely

* **Deterministic LangGraph flow**

  * Agent → Planner → (Approval) → Tools → Agent
  * No ReAct loops or uncontrolled retries

This approach provides **maximum transparency, control, and safety**.

---

### 3. `agent-prebuilt.py` — Middleware Investigation

This file contains an attempted implementation using:

* `TodoListMiddleware`
* LangChain agent middleware APIs referenced in the assignment

**Outcome**:

```
ModuleNotFoundError: No module named 'langchain.agents.middleware'
```

**Reason**:

* Agent middleware is **not available** in the LangChain version used by this project
* This is a **version-level limitation**, not a coding error

This file is intentionally included to:

* Show the investigation was performed
* Justify why a custom LangGraph implementation was the correct solution

---

## 🚀 Quick Start

### Prerequisites

* Python 3.9+
* Groq API key

### Installation

```bash
git clone https://github.com/sushma9903/todo-hitl.git
cd todo-hitl
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run the Agent

```bash
python agent/agent-custom.py
```

---

## 🧪 Example Interaction

```
You: what is the weather in London?

Agent's Plan:
1. Fetch current weather data for London using the get_weather MCP tool
2. This tool provides real-time weather data required to answer the query
3. Present temperature, humidity, and wind conditions in a clear response

Planned Tool Calls:
  - get_weather {'city': 'London'}

Approve execution? (yes/no): yes
```

---

## 🧩 Key Design Decisions

### Why custom Todo planning instead of middleware?

* Middleware APIs were unavailable in the current LangChain version
* LangGraph allows **explicit, inspectable state transitions**
* Planning logic is fully deterministic and testable

### Why Human-in-the-Loop before tools?

* Prevents unintended side effects
* Enables user trust and safety
* Mirrors real-world production approval workflows

### Why MCP separation?

* MCP server handles *only* execution
* Agent handles reasoning, planning, and decisions
* Clean separation of concerns

---

## 📦 Dependencies

* langchain
* langgraph
* langchain-groq
* mcp
* pydantic
* python-dotenv

---

## 🏁 Summary

By the end of this task, the agent:

* Plans before acting
* Explains its intent clearly
* Requires explicit human approval
* Executes tools safely and deterministically
* Knows when **not** to proceed

This results in a **production-grade, explainable AI agent** that is safer and more controllable than a standard tool-calling chatbot.

---
