import operator
from urllib import response
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import DefaultDict, List, TypedDict, Annotated
from ddgs import DDGS
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()
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
    sectorOutlookQuery: str
    outlookQuery: str
    completedSections: Annotated[List[str], operator.add]
    finalReport: str


gemini_key = os.getenv("GEMINI-API-KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=gemini_key)


def Supervisor( state: State):
    '''
        Understand the task/query at hand and see which agents should be used. Then, list all the sub-tasks 
        for the required agents and pass them forward to call them in parallel.
    '''
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
        - Agent names to use --> NewsAgent, FinancialStmtAgent, OutlookAgent, SectorAgent

        You MUST respond with **valid JSON only** in the exact format below. No extra text, no explanations, no markdown.

        Filter Stocks query - filter companies with $1B+ sales and 10% net margin
        Response - 
        {
        "agentsNeeded": ["CypherAgent", "NewsAgent", "FinancialStmtAgent"],
        "subTasks": {
            "CypherAgent": "Filter companies where totalRevenue >= 1000000000 AND netMargin >= 0.10",
            "NewsAgent": "Get latest news for the filtered companies",
            "FinancialStmtAgent": "Analyze earnings and cash flow trends for top results"
            }
        }
                      
        Company name: AAPL      Query: What does the outlook AAPL look like?
        Response - 
        {
        "agentsNeeded": ["NewsAgent", "FinancialStmtAgent", "OutlookAgent"],
        "subTasks": {
            "NewsAgent": "Get latest news for the AAPL company",
            "FinancialStmtAgent": "Analyze earnings and cash flow trends."
            "OutlookAgent": "Understand what the company is up to and what are its future plans."
            }
        }             
        """),
        HumanMessage(content=f"Company name: {state["tickerName"]}  and Query: {state["query"]}"),
        ]
    )
    return { "agentsNeeded": response.agentsNeeded,     # type: ignore[reportArgumentType]
            "subTasks": response.subTasks,              # type: ignore[reportArgumentType]
            "completedSections": state.get("completedSections", []),
            "finalReport": state.get("finalReport", "") } 


def assignAgents(state: State):
    '''
        Assigning the tasks to different agents and calling them in parallel
    '''
    agentMap = { "NewsAgent": "newsQuery", "FinancialStmtAgent": "finStmtQuery",
                "OutlookAgent": "outlookQuery", "SectorAgent": "sectorOutlookQuery"}
    for a, subtask in state["subTasks"].items():
        print(f"using {a} for this task...")
        state[agentMap[a]] = subtask
    return [Send(s, state) for s in state["agentsNeeded"]]


def NewsAgent(state: State):
    '''
        Get the recent news based on the query or company name and provide a summarization of it.
    '''
    with DDGS() as ddgs:
        results = ddgs.news(
            query=f"latest news around {state['tickerName']}",
            max_results=5,
            region="wt-wt"
        )
    raw_news = "\n".join([f"{r['title']}: {r['body']} ({r['date']})" for r in results])
    
    newsSummary = llm.invoke([
        SystemMessage(content=""" You have to answer as a News reporter expert in Finance who takes raw news info and user provided query, and turns it into
                    clean and understandable news for the common people. Don't say your name. 
                    Also, your response should be as if you are writing a report. Short and to-the-point with a suitable title."""),
        HumanMessage(content=f"Query: {state["newsQuery"]}. Here is the raw news info: {raw_news}")
        ])
    return { "completedSections": [newsSummary.content] }


def OutlookAgent(state: State):
    return {"completedSections": [" "]}


def SectorAgent(state: State):
    '''
        Get the recent news based on the Sector query requested and provide a summarization of it.
    '''
    with DDGS() as ddgs:
        results = ddgs.news(
            query=f"Query: {state.get('query', '')}",
            max_results=5,
            region="wt-wt"
        )
    raw_news = "\n".join([f"{r['title']}: {r['body']} ({r['date']})" for r in results])
    
    sectorSummary = llm.invoke([
        SystemMessage(content=""" You have to answer as a News reporter expert in different Industry sectors like Mining, Technology, Hospitality and more
                       who takes raw news info and user provided query, and then turns it into clean and understandable news for the common people. 
                       Don't say your name. Also, your response should be as if you are writing a report. Short and to-the-point with a suitable title.
                      So, suppose if the user query says - what is the outlook of the mining industry?, so, you will explain the current sector outlook."""),
        HumanMessage(content=f"Query: {state["sectorOutlookQuery"]}. Here is the raw news info: {raw_news}")
        ])
    return { "completedSections": [sectorSummary.content] }


def FinancialStmtAgent(state: State):
    return {"completedSections": [" "]}


def Controller(state: State):
    '''
        Get the inputs from all the agents to generate the report.
    '''
    completedSections = state["completedSections"]
    completeReport = "\n\n----\n\n".join(completedSections)
    return { "finalReport": completeReport }


def build_grpah():
    '''
        Build the entire agent workflow.
        Adding Nodes and conditional edges to activate only required Agents.
    '''
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
    return orchestrator.compile()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app = build_grpah()
    
    result = app.invoke({
        "tickerName": "NVDA",
        "query": "Status of the Mining Sector.",              
        "completedSections": [],     
        "finalReport": ""
    }) # type: ignore[reportArgumentType]

    print("\n\n")
    print(result["finalReport"])    