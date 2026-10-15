import logging
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from yfinance import ticker
from agents import Agents
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import polars as pl
from setup_data import fetchTickers

logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app: FastAPI):
    agents = Agents()
    graph = agents.buildGraph()
    # Set the global variable at application startup
    app.state.AgentGraph = graph
    app.state.request_counter = 0
    
    yield  # The app runs while paused here
    print("Application shutting down...")

app = FastAPI(title="Warren Financial Agent", lifespan=lifespan)

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    tickerName: str
    query: str

@app.post("/research")
async def analyze(request: AnalysisRequest):
    ticker_name = request.tickerName.strip().upper()
    query = request.query.strip()
    if not ticker_name:
        raise HTTPException(status_code=422, detail="Ticker symbol is required.")
    if not query:
        raise HTTPException(status_code=422, detail="Research query is required.")

    try:
        tickers = fetchTickers()
        supported_tickers = {
            symbol.strip().upper()
            for symbol in tickers["Symbol"].to_list()
            if isinstance(symbol, str)
        }
    except (OSError, KeyError, pl.exceptions.PolarsError) as exc:
        logger.exception("Unable to load the supported ticker list")
        raise HTTPException(
            status_code=503,
            detail="Ticker list is unavailable. Run setup_data.py to create ticker.csv.",
        ) from exc

    if ticker_name not in supported_tickers:
        raise HTTPException(
            status_code=422,
            detail=f"Ticker '{ticker_name}' is not in the supported NASDAQ ticker list.",
        )

    async def stream_report():
        specialist_agents = {"NewsAgent", "FinancialStmtAgent", "OutlookAgent", "SectorAgent"}
        agents_used = []
        try:
            async for event in app.state.AgentGraph.astream_events({
                "tickerName": ticker_name,
                "query": query
            }, version="v2"):
                event_name = event.get("metadata", {}).get("langgraph_node")
                if event_name in specialist_agents and event["event"] == "on_chain_start":
                    if event_name not in agents_used:
                        agents_used.append(event_name)
                        yield json.dumps({
                            "type": "agent",
                            "name": event_name,
                            "status": "started",
                        }) + "\n"
                elif event_name in specialist_agents and event["event"] == "on_chain_end":
                    yield json.dumps({
                        "type": "agent",
                        "name": event_name,
                        "status": "completed",
                    }) + "\n"
                elif event_name == "controller" and event["event"] == "on_chain_start":
                    yield json.dumps({
                        "type": "phase",
                        "status": "generating",
                    }) + "\n"

                if (
                    event["event"] == "on_chat_model_stream"
                    and event.get("metadata", {}).get("langgraph_node") == "controller"
                ):
                    content = event["data"]["chunk"].content
                    if isinstance(content, str) and content:
                        yield json.dumps({"type": "token", "content": content}) + "\n"
            yield json.dumps({"type": "done", "agents": agents_used}) + "\n"
        except Exception:
            logger.exception("Research request failed for ticker %s", ticker_name)
            yield json.dumps({
                "type": "error",
                "detail": "The research service failed to complete the request. Please try again later.",
            }) + "\n"

    return StreamingResponse(stream_report(), media_type="application/x-ndjson")


@app.get("/suggestions")
async def suggestions(q: str = ""):
    tickers = fetchTickers()
    if not q:
        # return first 20 as example
        data = [
            {"symbol": row["Symbol"], "name": row["Security Name"]}
            for row in tickers.head(20).iter_rows(named=True)
        ]
        return {"status": "success", "data": data}

    q_lower = q.lower()
    filtered = tickers.filter(
        (pl.col("Symbol").str.to_lowercase().str.contains(q_lower)) |
        (pl.col("Security Name").str.to_lowercase().str.contains(q_lower))
    )

    data = [
        {"symbol": row["Symbol"], "name": row["Security Name"]}
        for row in filtered.iter_rows(named=True)
    ]

    return {"status": "success", "data": data}


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")