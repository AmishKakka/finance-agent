from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from yfinance import ticker
from agents import Agents
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from IPython.display import Markdown
import polars as pl
from setup_data import fetchTickers

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
    try:
        response = app.state.AgentGraph.invoke({
            "tickerName": request.tickerName,
            "query": request.query
        }) # type: ignore[reportArgumentType]
        return {
            "status": "success",
            "report": Markdown(response["finalReport"])
        } # type: ignore[reportArgumentType]
    except Exception as e:
        return {
            "status": "error",
            "message": Markdown(str(e))
        }


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