# finance-agent

**Multi-Agent Financial Research System with LangGraph**

An agentic AI system that takes a prompt (e.g., “Show me positive cash flow companies with >20% profit margin in tech sector”) and intelligently routes it through specialized agents to deliver a complete investment research report.


![High-level Architecture](./Highlevel%20plan%20v_0.jpeg)
*High-level plan v0*

## Features

- **Intelligent Task Decomposition**: Supervisor agent analyzes the user prompt and decides which specialized agents are needed
- **Dynamic Multi-Agent Orchestration** via LangGraph (conditional routing + parallel execution)

- **5 Specialized Agents**:
  - **NewsAgent** — Fetches & summarizes latest news using Tavily Web search.
  - **Financial Stmt. Agent** — Pulls cash flow, earnings, income statements, ratios (yfinance) + generates insights.
  - **SQL Agent** — Natural-language → SQL queries on Duckdb (filters by market cap, sector, profit margin, cashflow positivity, etc.)
  - **Outlook Agent** — Price targets, revenue estimates, analyst recommendations & future outlook.
  - **Sector Agent** — Sector/industry level news and forward-looking outlook.
- **Controller / Synthesizer** — Combines all agent outputs into a polished, actionable final report with confidence scores and sources.

## Architecture

The system follows a **Supervisor → Dynamic Routing → Controller** pattern:

1. **User Prompt** → **Supervisor** (LLM planner)
2. Supervisor decides which agents to activate (can be 1, 2, or all 5)
3. Selected agents run **in parallel** where possible
4. All outputs flow to **Controller** for final synthesis
5. Result: clean Markdown report

This design allows the system to be efficient (only calls necessary agents) while remaining extremely flexible.

## Tech Stack

- **Orchestration**: LangGraph
- **SQL Database**: Duck DB
- **LLM**: Gemini 2.5 Flash (via Google API)
- **Data Sources**: yfinance, DuckDuckGo Search
- **Backend**: Python, Pydantic
- **Persistence**: SQLite / Redis (LangGraph checkpointer)

## How It Works (Example Flow)

**User Prompt**:  
"I want positive cash-flow tech companies with profit margin >20% and strong sector outlook."

**Supervisor decides**:
- SQL Agent (filtering)
- Financial Stmt Agent (cashflow validation)
- Sector Agent (sector outlook)

All agents run → Controller synthesizes → Final report delivered.
