# Multi-Agent Financial Research System with LangGraph

Takes a company ticker + user query and produces a focused investment research report by routing to specialized agents.

## How it works

1. **Supervisor (Jev)** – Decides which agents are needed using calibrated probabilities
2. **Selected agents run in parallel** (LangGraph)
3. **Controller** – One call to Gemini synthesizes all findings into a clean report

Only the relevant agents are used, keeping it fast and cheap.

## Agents

| Agent | What it does |
|-------|--------------|
| **NewsAgent** | Recent news & events (DuckDuckGo) |
| **FinancialStmtAgent** | Income statement, balance sheet, cash flow, earnings (yfinance) |
| **OutlookAgent** | Analyst recommendations, price targets, revenue & growth estimates (yfinance) |
| **SectorAgent** | Sector / industry level news & outlook |

Each sub–agent can also use Jev internally to decide which exact data points to fetch.

## Example

**Ticker:** `AAPL`

**Query:**  `Is Apple cash-flow positive? What is the current outlook around it?`

**Jev decides:**

- FinancialStmtAgent → 0.98
- OutlookAgent → 0.84
- NewsAgent → 0.19
- SectorAgent → 0.05

Only the two high-probability agents run → final report.

**Routing / Decisions**: TypeSafe's Jev model

**Orchestration**: LangGraph

**LLM (final report only)**: Gemini 2.5 Flash

**Data**: yfinance + DuckDuckGo Search

## Installation

```bash
git clone https://github.com/AmishKakka/finance-agent.git
cd finance-agent
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt

# Add your keys to .env
GEMINI-API-KEY=...
JEV-API-KEY=...
```
