from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import DefaultDict, List, TypedDict, Annotated
import tavily
import os
from pydantic import BaseModel, Field
print("Import successfull!!!")


# Declaring schema for sections of the report 
class Section(BaseModel):
    name: str
    description: str

class Sections(BaseModel):
    sections: List[Section]

class Plan(BaseModel):
    agentsNeeded: List[str] = Field(description="List of agents required to complete the task")
    subTasks: DefaultDict = Field(description="Descriptive task for each agent required.")

# Declaring state schema for the nodes
class State(TypedDict):
    tickerName: str
    query: str
    agentsNeeded: List[str]
    subTasks: DefaultDict
    newsQuery: str
    finStmtQuery: str
    cypherQuery: str
    sectorOutlookQuery: str
    outlookQuery: str

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="your-gemini-api-key")
planner = model.with_structured_output(
    schema=Plan.model_json_schema(), 
    method="json_schema"
)

response = planner.invoke([
    SystemMessage(content="""
    You are a financial advisor working on Wall Street for 25+ years with experience in all industries/sector. 
    Also, if the user has provided a query use that, understand what user wants and generate tasks for agents as required.
    Your ONLY job is to analyze the user's query and create the most efficient execution plan using the available specialized agents. 
    Do not generate the final report yourself.
    
        1. News Agent — Fetches and summarizes the latest news for specific companies or sectors using Tavily.
        2. Financial Stmt. Agent — Pulls cash flow, earnings, income statements, ratios from yfinance and generates quantitative insights.
        3. Outlook Agent — Retrieves price targets, revenue estimates, analyst recommendations, and future outlook.
        4. Sector Agent — Sector/industry level news and forward-looking outlook.
        
    Core Rules:
    - Start with CypherAgent for any screening, filtering, or list-based query (e.g., "filter companies with $1B+ sales and 10% net margin").
    - Use the minimum number of agents needed. Never call unnecessary agents.
    - Multiple agents can run in parallel.
    - If the user asks about one specific company, you may skip CypherAgent unless additional filtering is required.

    You MUST respond with **valid JSON only** in the exact format below. No extra text, no explanations, no markdown.

    {
    "agentsNeeded": ["CypherAgent", "NewsAgent", "FinancialStatementAgent"],
    "subTasks": {
        "CypherAgent": "Filter companies where totalRevenue >= 1000000000 AND netMargin >= 0.10",
        "NewsAgent": "Get latest news for the filtered companies",
        "FinancialStatementAgent": "Analyze earnings and cash flow trends for top results"
    }
    }"""),
    HumanMessage(content=f"Here is the company name: APPL and query: What are the financicals of the company? Also, give some recent news around AAPL"),
    ]
)

print(response)