import operator
from urllib import response
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import DefaultDict, List, TypedDict, Annotated
from tavily import TavilyClient
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
    agentsNeeded: List[str]
    subTasks: DefaultDict[str, str]

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
    completedSections: Annotated[List[str], operator.add]
    finalReport: str

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="your-gemini-api-key")


def Supervisor( state: State):
    planner = llm.with_structured_output(Plan)

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
            5. CypherAgent — Filters companies in Neo4j graph by financial metrics
            
        Core Rules:
        - Start with CypherAgent for any screening, filtering, or list-based query (e.g., "filter companies with $1B+ sales and 10% net margin").
        - Use the minimum number of agents needed. Never call unnecessary agents.
        - Multiple agents can run in parallel.
        - If the user asks about one specific company, you may skip CypherAgent unless additional filtering is required.
        - Agent names to use - NewsAgent, FinancialStmtAgent, OutlookAgent, SectorAgent

        You MUST respond with **valid JSON only** in the exact format below. No extra text, no explanations, no markdown.

        {
        "agentsNeeded": ["CypherAgent", "NewsAgent", "FinancialStmtAgent"],
        "subTasks": {
            "CypherAgent": "Filter companies where totalRevenue >= 1000000000 AND netMargin >= 0.10",
            "NewsAgent": "Get latest news for the filtered companies",
            "FinancialStatementAgent": "Analyze earnings and cash flow trends for top results"
        }
        }"""),
        HumanMessage(content=f"Company name: {state["tickerName"]}  and Query: {state["query"]}"),
        ]
    )
    return { "agentsNeeded": response.agentsNeeded,     # type: ignore[reportArgumentType]
            "subTasks": response.subTasks,              # type: ignore[reportArgumentType]
            "completedSections": state.get("completedSections", []),
            "finalReport": state.get("finalReport", "") } 


def assignAgents(state: State):
    return [Send(s, state) for s in state["agentsNeeded"]]

def NewsAgent(state: State):
    tavily = TavilyClient(api_key="your-tavily-api-key")
    response = tavily.search(
        query=f"latest news around {state["tickerName"]}",
        include_answer=True,
        search_depth="fast"
    )

    newsSummary = llm.invoke([
        SystemMessage(content=""" You have to answer as a News reporter expert in Finance who takes raw news info and turns it into
                       clean and understandable news for the common people. Don't say your name. 
                      Also, your response should be as if you are writing a report. Short and to-the-point with a suitable title."""),
        HumanMessage(content=f"Here is the raw news info: {response}")
    ])
    # print(newsSummary)
    return { "completedSections": [newsSummary.content] }

def OutlookAgent(state: State):
    return {"completedSections": [" "]}

def SectorAgent(state: State):
    return {"completedSections": [" "]}

def FinancialStmtAgent(state: State):
    return {"completedSections": [" "]}

def Controller(state: State):
    completedSections = state["completedSections"]
    completeReport = "\n\n----\n\n".join(completedSections)
    return { "finalReport": completeReport }


if __name__ == "__main__":
    orchestrator = StateGraph(State)
    # Adding agents as Nodes to the graph
    orchestrator.add_node("supervisor", Supervisor)
    orchestrator.add_node("NewsAgent", NewsAgent)
    orchestrator.add_node("FinancialStmtAgent", FinancialStmtAgent)
    orchestrator.add_node("OutlookAgent", OutlookAgent)
    orchestrator.add_node("SectorAgent", SectorAgent)
    orchestrator.add_node("controller", Controller)

    # Adding edges between nodes as Path in the graph
    orchestrator.add_edge(START, "supervisor")
    orchestrator.add_conditional_edges("supervisor", 
                                        assignAgents, 
                                        ["NewsAgent", "FinancialStmtAgent", "OutlookAgent", "SectorAgent"])
    orchestrator.add_edge("NewsAgent", "controller")
    orchestrator.add_edge("FinancialStmtAgent", "controller")
    orchestrator.add_edge("OutlookAgent", "controller")
    orchestrator.add_edge("SectorAgent", "controller")
    orchestrator.add_edge("controller", END)
    graph = orchestrator.compile()
    
    result = graph.invoke({
        "tickerName": "NVDA",
        "query": "",              
        "completedSections": [],     
        "finalReport": ""
    }) # type: ignore[reportArgumentType]

    print("\n\n")
    print(result["finalReport"])    