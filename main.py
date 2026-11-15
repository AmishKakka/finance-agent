import logging
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from yfinance import ticker
from agents import Agents, extract_text_content
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessageChunk
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
        selected_agents = []
        report_text = ""

        def line(payload: dict) -> str:
            return json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"

        try:
            async for item in app.state.AgentGraph.astream(
                {"tickerName": ticker_name, "query": query},
                stream_mode=["custom", "messages", "updates"],
            ):
                if not (isinstance(item, tuple) and len(item) == 2):
                    continue
                mode, payload = item
                if mode == "custom" and isinstance(payload, dict):
                    if payload.get("agent") == "supervisor" and payload.get("status") == "decision":
                        selected_agents = [
                            name for name in payload.get("selected", [])
                            if isinstance(name, str)
                        ]
                    try:
                        yield line(payload)
                    except (TypeError, ValueError):
                        logger.exception("Could not serialize a progress event")
                        yield line({
                            "type": "trace",
                            "agent": payload.get("agent") or "supervisor",
                            "status": "error",
                            "message": "A progress update could not be displayed.",
                        })
                elif mode == "messages" and isinstance(payload, tuple) and len(payload) == 2:
                    message, metadata = payload
                    node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
                    if node != "controller":
                        continue
                    content = extract_text_content(getattr(message, "content", ""))
                    if not content:
                        continue
                    # The messages stream can repeat the finished note after the chunks.
                    if not isinstance(message, AIMessageChunk):
                        if report_text:
                            continue
                        report_text = content
                        yield line({"type": "token", "content": content})
                        continue
                    report_text += content
                    yield line({"type": "token", "content": content})
                elif mode == "updates" and isinstance(payload, dict) and not report_text:
                    output = payload.get("controller")
                    if isinstance(output, dict):
                        final_report = extract_text_content(output.get("finalReport", ""))
                        if final_report:
                            report_text = final_report
                            yield line({"type": "token", "content": final_report})

            if not report_text.strip():
                yield line({
                    "type": "error",
                    "detail": "Report generation finished without returning any text. Please try again.",
                })
                return
            yield line({"type": "done", "agents": selected_agents})
        except Exception:
            logger.exception("Research request failed for ticker %s", ticker_name)
            yield line({
                "type": "trace",
                "agent": "controller",
                "status": "error",
                "message": "The research run failed before the report was finished.",
            })
            yield line({
                "type": "error",
                "detail": "The research service failed to complete the request. Please try again later.",
            })

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