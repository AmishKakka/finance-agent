from fastapi import FastAPI
from pydantic import BaseModel
from yfinance import ticker
from agents import build_graph, graph_app
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from IPython.display import Markdown
from setup_duck_db import fetchTickers

app = FastAPI(title="Financial Agentic AI")

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
        response = graph_app.invoke({
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
    companies = list(tickers["Symbol"])

    if not q:
        return {
            "status": "success",
            "data": companies
        }
    filtered = [cmp for cmp in companies if q.lower() in cmp.lower()]
    return {
        "status": "success",
        "data": filtered
    }


@app.get("/")
def home():
    return {"message": "Financial Agent is running..."}