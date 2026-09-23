import operator
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Any, DefaultDict, List, TypedDict, Annotated
from langchain_typesafe import Noul, TypeSafeClassifier
import os
from datetime import datetime
from tavily import TavilyClient
import yfinance as yf
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv
from pydantic import BaseModel
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
    agentsProb: DefaultDict[str, float]
    agentsNeeded: List[str]
    completedSections: Annotated[List[dict[str, Any]], operator.add]
    finalReport: str


class Agents:
    def __init__(self):
        gemini_key = os.getenv("GEMINI-API-KEY") or ""
        jev_key = os.getenv("JEV-API-KEY") or ""
        tavily_key = os.getenv("TAVILY-API-KEY") or ""
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=gemini_key)
        self.tavily = TavilyClient(tavily_key)
        self.supervisor = TypeSafeClassifier(api_key=jev_key)

    def Supervisor(self, state: State):
        response = self.supervisor.invoke(
            {
                "state": f"""
                    Ticker: {state["tickerName"]}
                    User query: {state["query"]}

                    Available specialized agents:
                    - NewsAgent: recent news and event summaries
                    - FinancialStmtAgent: income statement, balance sheet, cash flow, earnings
                    - OutlookAgent: analyst recommendations, price targets, revenue/growth estimates
                    - SectorAgent: sector/industry outlook and news
                    """,
                "questions": {
                    "need_news": Noul(
                        instructions="Does this query require recent news, events, or product announcements about the company?",
                        criteria={
                            "true": "The query asks about latest developments, news, or current events.",
                            "false": "The query is only about financial statements, analyst outlook, or sector trends."
                        } # type: ignore[reportArgumentType]
                    ),
                    "need_fin_stmt": Noul(
                        instructions="Does this query require analysis of the company's financial statements (balance sheet, cash flow, income statement, or earnings)?",
                        criteria={
                            "true": "The query asks about cash position, balance sheet, profitability, earnings, or financial health metrics.",
                            "false": "The query does not need raw financial statement data."
                        } # type: ignore[reportArgumentType]
                    ),
                    "need_outlook": Noul(
                        instructions="Does this query require analyst recommendations, price targets, revenue estimates, or growth outlook?",
                        criteria={
                            "true": "The query asks about future expectations, analyst views, or price targets.",
                            "false": "The query is only about historical statements or current news."
                        } # type: ignore[reportArgumentType]
                    ),
                    "need_sector": Noul(
                        instructions="Does this query require sector or industry-level outlook and news (not company-specific)?",
                        criteria={
                            "true": "The query is about the broader industry, sector trends, or competitive landscape.",
                            "false": "The query is focused on one specific company."
                        } # type: ignore[reportArgumentType]
                    )
                }
            }
        )
        print("Supervisor Node: ", response)
        agent_map = { "need_news": "NewsAgent", "need_fin_stmt": "FinancialStmtAgent",
                      "need_outlook": "OutlookAgent", "need_sector": "SectorAgent"}
        agents_prob = {
            name: noulResponse.noul for name, noulResponse in response.nouls.items()
        }
        agents_needed = []
        for name, prob in agents_prob.items():
                    if prob >= 0.53:
                        agents_needed.append(agent_map[name])
        return {
            "agentsProb": agents_prob,
            "agentsNeeded": agents_needed
        }

    def assignAgents(self, state: State):
        return [Send(agent_name, state) for agent_name in state["agentsNeeded"]]

    def NewsAgent(self, state: State):
        '''
            Get the recent news based on the query or company name.
        '''
        try:
            raw_news = ""
            response = self.tavily.search(
                query=state["query"],
                topic="news",
                search_depth="fast",
                max_results=6,
                time_range="month"
            )
            for r in response["results"]:
                raw_news += "\n".join(f"Score:{r["score"]} /t{r["title"]}: {r["content"]}/t Published on:{r["published_date"]}")
            return { 
                "completedSections": [{
                    "agent": "NewsAgent",
                    "probability": state["agentsProb"]["need_news"],
                    "raw": raw_news,
                }] 
            }
        except:
            return {
                "completedSections": [{
                    "agent": "NewsAgent",
                    "probability": state["agentsProb"]["need_news"],
                    "raw": "No recent news found",
                }]
            }

    def FinancialStmtAgent(self, state: State):
        '''
            Pulls cash flow, earnings, income statements, ratios from yfinance API.
        '''
        # exiting early if company name is not provided
        if state["tickerName"] == "":
            return {
                "completedSections": [{
                    "agent": "OutlookAgent",
                    "probability": state["agentsProb"]["need_outlook"],
                    "raw": "No outlook around this company found",
                }]
            }
        resp = self.supervisor.invoke({
                "state": f"""Ticker: {state['tickerName']}
                    User query: {state['query']}
                    We are deciding which financial statement data to fetch for this company.""",
                "questions": {
                    "get_income_stmt": Noul(
                        instructions="Does the user query require the income statement (revenue, net income, profitability, margins, operating expenses)?",
                        criteria={
                            "true": "The query asks about earnings, profit, revenue, net income, margins, or overall profitability.",
                            "false": "The query is only about balance sheet items, cash flow, or does not need income statement data."
                        } # type: ignore[reportArgumentType]
                    ),
                    "get_balance_sheet": Noul(
                        instructions="Does the user query require the balance sheet (assets, liabilities, equity, cash position, debt)?",
                        criteria={
                            "true": "The query asks about cash position, debt, assets, liabilities, equity, or overall financial health / solvency.",
                            "false": "The query is only about earnings, cash flow from operations, or does not need balance sheet data."
                        } # type: ignore[reportArgumentType]
                    ),
                    "get_cashflow": Noul(
                        instructions="Does the user query require the cash flow statement (operating, investing, financing cash flow, free cash flow)?",
                        criteria={
                            "true": "The query asks whether the company is cash-flow positive, free cash flow, cash generation, or cash burn.",
                            "false": "The query is only about income statement profitability or balance sheet items and does not need cash flow data."
                        } # type: ignore[reportArgumentType]
                    ),
                    "get_earnings_dates": Noul(
                        instructions="Does the user query require upcoming or past earnings dates / earnings calendar?",
                        criteria={
                            "true": "The query explicitly asks about next earnings date, earnings calendar, or when results will be released.",
                            "false": "The query does not ask about earnings timing or calendar."
                        } # type: ignore[reportArgumentType]
                    )
                }
            })
        try:
            ticker = yf.Ticker(state["tickerName"])
            finStmt_data = {}
            for function_name, noulResponse in resp.nouls.items():
                # print(function_name, noulResponse)
                # Lowered the Noul confidence, becuase sometimes for some stocks it works at 0.85 and not for others
                if noulResponse.noul >= 0.65:
                    if hasattr(ticker, function_name):
                        method_to_call = getattr(ticker, function_name)
                        finStmt_data[function_name] = str(method_to_call(as_dict=True))
                    else:
                        finStmt_data[function_name] = f"Method {function_name} not found for {state['tickerName']}"
            print(finStmt_data)
            return { 
                "completedSections": [{
                    "agent": "FinancialStmtAgent",
                    "probability": state["agentsProb"]["need_fin_stmt"],
                    "raw": finStmt_data,
                }] 
            }
        except:
            return {
                "completedSections": [{
                    "agent": "FinancialStmtAgent",
                    "probability": state["agentsProb"]["need_fin_stmt"],
                    "raw": "No financial data found",
                }]
            }

    def OutlookAgent(self, state: State):
        '''
            Retrieves price targets, revenue estimates, analyst recommendations from yfinance and generates insights.
        '''
        if state["tickerName"] == "":
            return {
                "completedSections": [{
                    "agent": "OutlookAgent",
                    "probability": state["agentsProb"]["need_outlook"],
                    "raw": "No outlook around this company found",
                }]
            }
        try:
            ticker = yf.Ticker(state["tickerName"])
            outlook_data = {}
            resp = self.supervisor.invoke({
                "state": f"""Ticker: {state['tickerName']}
                        User query: {state['query']}
                        We are deciding which analyst outlook data to fetch for this company.""",
                "questions": {
                    "get_recommendations_summary": Noul(
                        instructions="Does the user query require analyst recommendations or consensus rating (buy/hold/sell)?",
                        criteria={
                            "true": "The query asks about analyst ratings, recommendations, consensus, or what analysts think of the stock.",
                            "false": "The query is only about price targets, revenue estimates, or growth numbers and does not need the recommendation summary."
                        } # type: ignore[reportArgumentType]
                    ),
                    "get_analyst_price_targets": Noul(
                        instructions="Does the user query require analyst price targets (mean, high, low target price)?",
                        criteria={
                            "true": "The query asks about price target, target price, upside, or what price analysts expect.",
                            "false": "The query does not ask about target prices."
                        } # type: ignore[reportArgumentType]
                    ),
                    "get_revenue_estimate": Noul(
                        instructions="Does the user query require revenue estimates or sales forecasts?",
                        criteria={
                            "true": "The query asks about expected revenue, sales forecasts, or top-line estimates.",
                            "false": "The query does not ask about revenue or sales estimates."
                        } # type: ignore[reportArgumentType]
                    ),
                    "get_growth_estimates": Noul(
                        instructions="Does the user query require growth estimates (EPS growth, revenue growth, long-term growth)?",
                        criteria={
                            "true": "The query asks about growth rates, expected growth, EPS growth, or future expansion.",
                            "false": "The query does not ask about growth estimates."
                        } # type: ignore[reportArgumentType]
                    ),
                }
            })
            for function_name, noulResponse in resp.nouls.items():
                if noulResponse.noul >= 0.65:
                    if hasattr(ticker, function_name):
                        method_to_call = getattr(ticker, function_name)
                        outlook_data[function_name] = str(method_to_call())
                    else:
                        outlook_data[function_name] = f"Method {function_name} not found for {state['tickerName']}"
            return { 
                "completedSections": [{
                    "agent": "OutlookAgent",
                    "probability": state["agentsProb"]["need_outlook"],
                    "raw": outlook_data,
                }] 
            }
        except:
            return {
                "completedSections": [{
                    "agent": "OutlookAgent",
                    "probability": state["agentsProb"]["need_outlook"],
                    "raw": "No outlook around this company found",
                }]
            }
    
    def SectorAgent(self, state: State):
        '''
            Get the recent news around a Sector/Industry.
        '''
        try: 
            raw_news = ""
            response = self.tavily.search(
                            query=state["query"],
                            topic="news",
                            search_depth="fast",
                            max_results=6,
                            time_range="month"
                        )
            for r in response["results"]:
                raw_news += "\n".join(f"Score:{r["score"]} /t{r["title"]}: {r["content"]}/t Published on: {r["published_date"]}")
            return { 
                "completedSections": [{
                    "agent": "SectorAgent",
                    "probability": state["agentsProb"]["need_sector"],
                    "raw": raw_news,
                }] 
            }
        except:
            return {
                "completedSections": [{
                    "agent": "SectorAgent",
                    "probability": state["agentsProb"]["need_sector"],
                    "raw": "No outlook around this sector found",
                }]
            }

    def Controller(self, state: State):
        '''
            Get the inputs from all the agents to generate the report.
        '''
        sections = sorted(
            state["completedSections"],
            key=lambda s: s.get("probability", 0),
            reverse=True
        )

        evidence = "\n\n".join(
            f"### {s['agent']} (relevance: {s['probability']:.2f})\n{s['raw']}" for s in sections
        )

        report = self.llm.invoke([
            SystemMessage(content=f"""You are a senior equity research analyst.
                Write a clear, professional investment research note.
                Use the evidence below. Give more weight to higher-relevance sections.
                Structure the report with short headings. Be concise and factual.
                
                Today's Date: {datetime.now()}
                If the data you receive whether news or financial data and 
                the date or year on them appears to be far-behind today's date, then acknowledge the fact gracefully.
                Work on the evidence to provide insights but also with that acknowledge the date difference."""),
                        HumanMessage(content=f"""Ticker: {state['tickerName']}
                User question: {state['query']}

                Evidence from specialized agents:
                {evidence}

                Produce the final research report.
            """)
        ])
        return {"finalReport": report.content}
    
    def buildGraph(self):
        '''
        Build the entire workflow.
        Adding Nodes and conditional edges to activate only required Agents.
        '''
        orchestrator = StateGraph(State)
        # Adding agents as Nodes to the graph
        orchestrator.add_node("supervisor", self.Supervisor)
        orchestrator.add_node("NewsAgent", self.NewsAgent)
        orchestrator.add_node("FinancialStmtAgent", self.FinancialStmtAgent)
        orchestrator.add_node("OutlookAgent", self.OutlookAgent)
        orchestrator.add_node("SectorAgent", self.SectorAgent)
        orchestrator.add_node("controller", self.Controller)

        # Adding edges between nodes as Path in the graph
        orchestrator.add_edge(START, "supervisor")
        orchestrator.add_conditional_edges("supervisor", 
                                            self.assignAgents, 
                                            ["NewsAgent", "FinancialStmtAgent", "OutlookAgent", "SectorAgent"])
        orchestrator.add_edge("NewsAgent", "controller")
        orchestrator.add_edge("FinancialStmtAgent", "controller")
        orchestrator.add_edge("OutlookAgent", "controller")
        orchestrator.add_edge("SectorAgent", "controller")
        orchestrator.add_edge("controller", END)
        return orchestrator.compile()


# if __name__ == "__main__":
#     agents = Agents()
#     graph = agents.buildGraph()
#     initialState: State = {
#         "tickerName": "",
#         "query": "Outlook around Semiconductor space",
#         "agentsProb": DefaultDict(),
#         "agentsNeeded": [],
#         "completedSections": [],
#         "finalReport": "",
#     }
#     queryResponse = graph.invoke(initialState)
#     console = Console()
#     console.print(Markdown(queryResponse["finalReport"])) 