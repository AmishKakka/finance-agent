import operator
from urllib import response
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Any, DefaultDict, List, TypedDict, Annotated
from ddgs import DDGS
import os
import yfinance as yf
import json
from IPython.display import Markdown
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
    newsQuery: DefaultDict[str, Any]
    finStmtQuery: DefaultDict[str, Any]
    sectorOutlookQuery: DefaultDict[str, Any]
    outlookQuery: DefaultDict[str, Any]
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
        You have 5 specialized agents at your disposal to help the user with its query.
                      
        Agent Capabilities - 
        1. NewsAgent - Get the recent news based on the query or company name provided by the user and provide a summarization of it.
        2. FinancialStmtAgent - Pulls cash flow, earnings, income statements, balance sheet from yfinance and generates insights. 
                Specifically, it can fetch - get_income_stmt(), get_balance_sheet(), get_cashflow(), get_earnings(), get_earnings_dates().
        3. OutlookAgent - Retrieves price targets, revenue estimates, analyst recommendations from yfinance and generates insights.
                Specifically, it can fetch - get_recommendations_summary(), get_analyst_price_targets(), get_revenue_estimate(), get_growth_estimates()
        4. SectorAgent — Sector/industry level news and forward-looking outlook.
        5. SQLAgent - Filters companies based on financial ratios, net margin, profit margin, price, or Sector using SQL.
                      
        Core Rules:
        - Start with SQLAgent for any screening, filtering, or list-based query (e.g., "filter companies with $1B+ sales and 10% net margin").
        - Use minimum number of agents needed. Never call unnecessary agents.
        - Multiple agents can run in parallel.
        - If the user asks about one specific company, you may skip SQLAgent unless additional filtering is required.
        - Agent names to use --> NewsAgent, FinancialStmtAgent, OutlookAgent, SectorAgent, SQLAgent
        - For any agent that you select to work with, provide "task" and "fetch" arguments. The "fetch" argument is what it needs to fetch from
        its resource. So, if it is - 
                      1. NewsAgent - {"task": "Summarize the latest news around Apple", "fetch": ["latest news around Apple", "lastest products by Apple"]}
                      2. FinancialStmtAgent - {"task": "What do Apple's earning look like?", "fetch": ["get_earnings()"]}
                      3. SectorAgent - {"task": "Summarize the Govt's hand in Mining industry", "fetch": ["developments in Mining industry", "What is the Govt. doing for Mining companies"]}
                      4. OutlookAgent -  {"task": "what are analysts' view on Google?", "fetch": ["get_recommendations_summary()"]}
                      
        Now that you know what each agent is capable of, if the user has provided a query use that, understand what user wants and generate tasks for agents as required.
        Your ONLY job is to analyze the user's query and create the most efficient execution plan using the available specialized agents. 
        Do not generate the final report yourself.
        
        Suppose, the user selects AAPL as the company name and enters a query like - What does Apple's balance sheet look like? Are they cash positive?
        In this case, the user only wants to know how Apple has been performing cash flow side and what does its balance sheet look like.
        So, you will call FinancialStmtAgent and say specifically that you want to fetch - ["get_balance_sheet()", "get_cashflow()"], 
        and similarly for other tasks you willcall respective functions for this agent.       
                      
        Response -
        {
        "agentsNeeded": ["FinancialStmtAgent"],
        "subTasks": {
            "FinancialStmtAgent": {
                      "task": "Analyze earnings and cash flow trends for Apple",
                      "fetch": ["get_balance_sheet()", "get_cashflow()"]}
            }
        }

        You MUST respond with **valid JSON only** in the exact format below. No markdown.
        
        Filter Stocks query - filter companies with $1B+ sales and 10% net margin
        Response - 
        {
        "agentsNeeded": ["SQLAgent"],
        "subTasks": {
            "SQLAgent": "Filter companies where totalRevenue >= 1000000000 AND netMargin >= 0.10",
            }
        }
                      
        Company name: AAPL      Query: What does the outlook AAPL look like?
        Response - 
        {
        "agentsNeeded": ["NewsAgent", "FinancialStmtAgent", "OutlookAgent"],
        "subTasks": {
            "NewsAgent": {"task": "Summarize the latest news around Apple", "fetch": ["latest news around Apple", "lastest products by Apple"]},
            "OutlookAgent": {"task": "What do Apple's earning and growth estimates look like?", "fetch": ["get_earnings()", "get_growth_estimates()"]}
            "FinancialStmtAgent": {"task": "What do Apple's earning look like?", "fetch": ["get_earnings()"]}
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
    for a, subtask_str in state["subTasks"].items():
        print(f"using {a} for this task...")
        # Parse the string representation of the subtask dictionary
        subtask_dict = json.loads(subtask_str)
        state[agentMap[a]] = subtask_dict # Assign the parsed dictionary
    return [Send(s, state) for s in state["agentsNeeded"]]


def NewsAgent(state: State):
    '''
        Get the recent news based on the query or company name and provide a summarization of it.
    '''
    try:
      raw_news = ""
      for f in state["newsQuery"]["fetch"]:
        with DDGS() as ddgs:
            results = ddgs.news(
                query=f,
                max_results=5,
                region="wt-wt"
            )
        raw_news += "\n".join([f"{r['title']}: {r['body']} ({r['date']})" for r in results])
      
      newsSummary = llm.invoke([
          SystemMessage(content=""" You have to answer as a News reporter expert in Finance who takes raw news info and user provided query, and turns it into
                      clean and understandable news for the common people. Don't say your name. 
                      Also, your response should be as if you are writing a report. Short and to-the-point with a suitable title."""),
          HumanMessage(content=f"Query: {state["newsQuery"]["task"]}. Here is the raw news info: {raw_news}")
          ])
      return { "completedSections": [newsSummary.content] }
    except:
      return { "completedSections": ["No news found"] }


def OutlookAgent(state: State):
    '''
        Retrieves price targets, revenue estimates, analyst recommendations from yfinance and generates insights.
    '''
    try:
        ticker = yf.Ticker(state["tickerName"])
        outlook_data = {}
        for fetch_item in state["outlookQuery"]["fetch"]:
            function_name = fetch_item.replace('()', '')
            if hasattr(ticker, function_name):
                method_to_call = getattr(ticker, function_name)
                outlook_data[function_name] = str(method_to_call())
            else:
                outlook_data[function_name] = f"Method {function_name} not found for {state['tickerName']}"
        
        outlookSummary = llm.invoke([
            SystemMessage(content="""You are a financial analyst specializing in market outlooks. 
                        Based on the provided financial data, generate a concise summary of the company's outlook. 
                        Focus on key metrics like analyst recommendations, price targets, revenue estimates, and growth estimates. 
                        Present the information clearly and professionally, like a report. Do not say your name."""),
            HumanMessage(content=f"Query: {state["outlookQuery"]["task"]}. Here is the outlook data: {outlook_data}")
        ])
        return { "completedSections": [outlookSummary.content] }
    except Exception as e:
        return { "completedSections": [f"Error fetching outlook data: {e}"] }


def SectorAgent(state: State):
    '''
        Get the recent news based on the Sector query requested and provide a summarization of it.
    '''
    try: 
      raw_news = ""
      for f in state["sectorOutlookQuery"]["fetch"]:
        with DDGS() as ddgs:
            results = ddgs.news(
                query=f,
                max_results=5,
                region="wt-wt"
            )
        raw_news += "\n".join([f"{r['title']}: {r['body']} ({r['date']})" for r in results])
        
      sectorSummary = llm.invoke([
          SystemMessage(content=""" You have to answer as a News reporter expert in different Industry sectors like Mining, Technology, Hospitality and more
                        who takes raw news info and user provided query, and then turns it into clean and understandable news for the common people. 
                        Don't say your name. Also, your response should be as if you are writing a report. Short and to-the-point with a suitable title.
                        So, suppose if the user query says - what is the outlook of the mining industry?, so, you will explain the current sector outlook."""),
          HumanMessage(content=f"Query: {state["sectorOutlookQuery"]["task"]}. Here is the raw news info: {raw_news}")
          ])
      return { "completedSections": [sectorSummary.content] }
    except:
      return { "completedSections": ["No news for this Sector found"] }


def FinancialStmtAgent(state: State):
    '''
        Pulls cash flow, earnings, income statements, ratios (yfinance) and generates insights
    '''
    try:
        ticker = yf.Ticker(state["tickerName"])
        finStmt_data = {}
        for fetch_item in state["finStmtQuery"]["fetch"]:
            function_name = fetch_item.replace('()', '')
            if hasattr(ticker, function_name):
                method_to_call = getattr(ticker, function_name)
                finStmt_data[function_name] = str(method_to_call(as_dict=True))
            else:
                finStmt_data[function_name] = f"Method {function_name} not found for {state['tickerName']}"
        
        finStmtSummary = llm.invoke([
            SystemMessage(content="""You are a financial analyst specializing in market outlooks. 
                        Based on the provided financial data, generate a concise summary of the company's outlook. 
                        Focus on key metrics like analyst recommendations, price targets, revenue estimates, and growth estimates. 
                        Present the information clearly and professionally, like a report. Do not say your name."""),
            HumanMessage(content=f"Query: {state["finStmtQuery"]["task"]}. Here is the outlook data: {finStmt_data}")
        ])
        return { "completedSections": [finStmtSummary.content] }
    except Exception as e:
        return { "completedSections": [f"Error fetching outlook data: {e}"] }


def Controller(state: State):
    '''
        Get the inputs from all the agents to generate the report.
    '''
    completedSections = state["completedSections"]
    completeReport = "\n\n----\n\n".join(completedSections)
    return { "finalReport": completeReport }


def build_graph():
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
    app = build_graph()
    
    result = app.invoke({
        "tickerName": "GOOG",
        "query": "What does the future outlook of Google look like?",              
        "completedSections": [],     
        "finalReport": ""
    }) # type: ignore[reportArgumentType]

    print("\n\n")
    print(result["finalReport"])    
    Markdown(result["finalReport"])