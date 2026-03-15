from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Financial Agentic AI")

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    tickerName: str
    query: str

@app.post("/research")
async def analyze(request: AnalysisRequest):
    pass

@app.get("/")
def home():
    return {"message": "Financial Agent is running..."}