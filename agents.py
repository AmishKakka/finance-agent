import json
import math
import operator
from langgraph.config import get_stream_writer
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Any, DefaultDict, List, TypedDict, Annotated
from langchain_typesafe import Noul, TypeSafeClassifier
import os
from datetime import datetime
from tavily import TavilyClient
import numpy as np
import yfinance as yf
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv
from pydantic import BaseModel
import logging
load_dotenv()
print("Import successfull!!!")

AGENT_NAMES = {
    "need_news": "NewsAgent",
    "need_fin_stmt": "FinancialStmtAgent",
    "need_outlook": "OutlookAgent",
    "need_sector": "SectorAgent",
}
ROUTING_THRESHOLD = 0.53
TOOL_THRESHOLD = 0.65


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(extract_text_content(block) for block in content)
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    return ""


def emit(payload: dict[str, Any]) -> None:
    """Push a progress event to the active graph stream. No-op outside a run."""
    try:
        get_stream_writer()(payload)
    except Exception:
        logging.debug("Progress event was not streamed", exc_info=True)


def sanitize(value: Any) -> Any:
    """Make tool payloads safe to embed in a JSON event."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return sanitize(to_dict())
        except Exception:
            return str(value)
    return str(value)


def format_tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    to_string = getattr(value, "to_string", None)
    if callable(to_string) and not isinstance(value, dict):
        try:
            return to_string() # pyright: ignore[reportReturnType]
        except Exception:
            pass
    try:
        return json.dumps(sanitize(value), indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def tavily_results(response: Any) -> list[dict[str, Any]]:
    results = response.get("results", []) if isinstance(response, dict) else []
    cleaned = []
    for item in results:
        if not isinstance(item, dict):
            continue
        cleaned.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "content": item.get("content"),
            "score": item.get("score"),
            "published_date": item.get("published_date"),
        })
    return sanitize(cleaned)


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
        gemini_key = os.getenv("GEMINI_API_KEY") or ""
        jev_key = os.getenv("JEV_API_KEY") or ""
        tavily_key = os.getenv("TAVILY_API_KEY") or ""
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", api_key=gemini_key)
        self.tavily = TavilyClient(tavily_key)
        self.supervisor = TypeSafeClassifier(api_key=jev_key)

    def Supervisor(self, state: State):
        emit({
            "type": "trace",
            "agent": "supervisor",
            "status": "started",
            "message": f"Scoring specialized agents for {state['tickerName']}.",
        })
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
        logging.info("Supervisor Node results: ", response)
        agents_prob = {
            name: noulResponse.noul for name, noulResponse in response.nouls.items()
        }
        probabilities = []
        agents_needed = []
        for name, prob in agents_prob.items():
            score = float(prob)
            selected = score >= ROUTING_THRESHOLD
            agent_name = AGENT_NAMES[name]
            if selected:
                agents_needed.append(agent_name)
            probabilities.append({
                "agent": agent_name,
                "score": score,
                "selected": selected,
            })
        emit({
            "type": "trace",
            "agent": "supervisor",
            "status": "decision",
            "message": "Agent scores are in.",
            "threshold": ROUTING_THRESHOLD,
            "probabilities": probabilities,
            "selected": agents_needed,
        })
        emit({
            "type": "trace",
            "agent": "supervisor",
            "status": "completed",
            "message": (
                "Selected " + ", ".join(agents_needed) + "."
                if agents_needed
                else "No specialized agent cleared the threshold."
            ),
        })
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
        request = {
            "query": state["query"],
            "topic": "news",
            "search_depth": "fast",
            "max_results": 6,
            "time_range": "month",
        }
        emit({
            "type": "trace",
            "agent": "NewsAgent",
            "status": "started",
            "message": "Fetching recent company news.",
        })
        emit({
            "type": "trace",
            "agent": "NewsAgent",
            "status": "request",
            "message": "Calling Tavily search.",
            "request": request,
        })
        try:
            raw_news = ""
            response = self.tavily.search(**request)
            results = response["results"] if isinstance(response, dict) else []
            for r in results:
                raw_news += "\n".join(f"Score:{r["score"]} /t{r["title"]}: {r["content"]}/t Published on:{r["published_date"]}")
            emit({
                "type": "trace",
                "agent": "NewsAgent",
                "status": "response",
                "title": "Tavily response",
                "source": "tavily",
                "results": tavily_results(response),
            })
            emit({
                "type": "trace",
                "agent": "NewsAgent",
                "status": "completed",
                "message": f"Tavily returned {len(results)} result{'s' if len(results) != 1 else ''}.",
            })
            return { 
                "completedSections": [{
                    "agent": "NewsAgent",
                    "probability": state["agentsProb"]["need_news"],
                    "raw": raw_news,
                }] 
            }
        except Exception as exc:
            emit({
                "type": "trace",
                "agent": "NewsAgent",
                "status": "error",
                "message": f"Tavily request failed: {type(exc).__name__}: {exc}",
            })
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
            emit({
                "type": "trace",
                "agent": "FinancialStmtAgent",
                "status": "error",
                "message": "No ticker was provided, so no statements were fetched.",
            })
            return {
                "completedSections": [{
                    "agent": "OutlookAgent",
                    "probability": state["agentsProb"]["need_outlook"],
                    "raw": "No outlook around this company found",
                }]
            }
        emit({
            "type": "trace",
            "agent": "FinancialStmtAgent",
            "status": "started",
            "message": f"Deciding which yfinance statements to pull for {state['tickerName']}.",
        })
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
        calls = []
        for function_name, noulResponse in resp.nouls.items():
            score = float(noulResponse.noul)
            calls.append({
                "name": function_name,
                "score": score,
                "selected": score >= TOOL_THRESHOLD,
            })
        emit({
            "type": "trace",
            "agent": "FinancialStmtAgent",
            "status": "decision",
            "message": "Statement calls scored.",
            "threshold": TOOL_THRESHOLD,
            "calls": calls,
        })
        try:
            ticker = yf.Ticker(state["tickerName"])
            finStmt_data = {}
            for call in calls:
                function_name = call["name"]
                # Lowered the Noul confidence, becuase sometimes for some stocks it works at 0.85 and not for others
                if not call["selected"]:
                    continue
                emit({
                    "type": "trace",
                    "agent": "FinancialStmtAgent",
                    "status": "request",
                    "message": f"Calling yfinance Ticker.{function_name}(as_dict=True).",
                    "request": {
                        "ticker": state["tickerName"],
                        "method": function_name,
                        "arguments": {"as_dict": True},
                    },
                })
                if hasattr(ticker, function_name):
                    method_to_call = getattr(ticker, function_name)
                    result = method_to_call(as_dict=True)
                    finStmt_data[function_name] = str(result)
                    body = format_tool_result(result)
                else:
                    body = f"Method {function_name} not found for {state['tickerName']}"
                    finStmt_data[function_name] = body
                emit({
                    "type": "trace",
                    "agent": "FinancialStmtAgent",
                    "status": "response",
                    "title": f"yfinance {function_name}",
                    "source": "yfinance",
                    "body": body,
                })
            emit({
                "type": "trace",
                "agent": "FinancialStmtAgent",
                "status": "completed",
                "message": (
                    "Fetched " + ", ".join(finStmt_data) + "."
                    if finStmt_data
                    else "No statement call cleared the threshold."
                ),
            })
            return { 
                "completedSections": [{
                    "agent": "FinancialStmtAgent",
                    "probability": state["agentsProb"]["need_fin_stmt"],
                    "raw": finStmt_data,
                }] 
            }
        except Exception as exc:
            emit({
                "type": "trace",
                "agent": "FinancialStmtAgent",
                "status": "error",
                "message": f"yfinance request failed: {type(exc).__name__}: {exc}",
            })
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
            emit({
                "type": "trace",
                "agent": "OutlookAgent",
                "status": "error",
                "message": "No ticker was provided, so no outlook data was fetched.",
            })
            return {
                "completedSections": [{
                    "agent": "OutlookAgent",
                    "probability": state["agentsProb"]["need_outlook"],
                    "raw": "No outlook around this company found",
                }]
            }
        emit({
            "type": "trace",
            "agent": "OutlookAgent",
            "status": "started",
            "message": f"Deciding which yfinance outlook calls to make for {state['tickerName']}.",
        })
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
            calls = []
            for function_name, noulResponse in resp.nouls.items():
                score = float(noulResponse.noul)
                calls.append({
                    "name": function_name,
                    "score": score,
                    "selected": score >= TOOL_THRESHOLD,
                })
            emit({
                "type": "trace",
                "agent": "OutlookAgent",
                "status": "decision",
                "message": "Outlook calls scored.",
                "threshold": TOOL_THRESHOLD,
                "calls": calls,
            })
            for call in calls:
                function_name = call["name"]
                if not call["selected"]:
                    continue
                emit({
                    "type": "trace",
                    "agent": "OutlookAgent",
                    "status": "request",
                    "message": f"Calling yfinance Ticker.{function_name}().",
                    "request": {
                        "ticker": state["tickerName"],
                        "method": function_name,
                        "arguments": {},
                    },
                })
                if hasattr(ticker, function_name):
                    method_to_call = getattr(ticker, function_name)
                    result = method_to_call()
                    outlook_data[function_name] = str(result)
                    body = format_tool_result(result)
                else:
                    body = f"Method {function_name} not found for {state['tickerName']}"
                    outlook_data[function_name] = body
                emit({
                    "type": "trace",
                    "agent": "OutlookAgent",
                    "status": "response",
                    "title": f"yfinance {function_name}",
                    "source": "yfinance",
                    "body": body,
                })
            emit({
                "type": "trace",
                "agent": "OutlookAgent",
                "status": "completed",
                "message": (
                    "Fetched " + ", ".join(outlook_data) + "."
                    if outlook_data
                    else "No outlook call cleared the threshold."
                ),
            })
            return { 
                "completedSections": [{
                    "agent": "OutlookAgent",
                    "probability": state["agentsProb"]["need_outlook"],
                    "raw": outlook_data,
                }] 
            }
        except Exception as exc:
            emit({
                "type": "trace",
                "agent": "OutlookAgent",
                "status": "error",
                "message": f"yfinance request failed: {type(exc).__name__}: {exc}",
            })
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
        request = {
            "query": state["query"],
            "topic": "news",
            "search_depth": "fast",
            "max_results": 6,
            "time_range": "month",
        }
        emit({
            "type": "trace",
            "agent": "SectorAgent",
            "status": "started",
            "message": "Fetching sector and industry news.",
        })
        emit({
            "type": "trace",
            "agent": "SectorAgent",
            "status": "request",
            "message": "Calling Tavily search.",
            "request": request,
        })
        try: 
            raw_news = ""
            response = self.tavily.search(**request)
            results = response["results"] if isinstance(response, dict) else []
            for r in results:
                raw_news += "\n".join(f"Score:{r["score"]} /t{r["title"]}: {r["content"]}/t Published on: {r["published_date"]}")
            emit({
                "type": "trace",
                "agent": "SectorAgent",
                "status": "response",
                "title": "Tavily response",
                "source": "tavily",
                "results": tavily_results(response),
            })
            emit({
                "type": "trace",
                "agent": "SectorAgent",
                "status": "completed",
                "message": f"Tavily returned {len(results)} result{'s' if len(results) != 1 else ''}.",
            })
            return { 
                "completedSections": [{
                    "agent": "SectorAgent",
                    "probability": state["agentsProb"]["need_sector"],
                    "raw": raw_news,
                }] 
            }
        except Exception as exc:
            emit({
                "type": "trace",
                "agent": "SectorAgent",
                "status": "error",
                "message": f"Tavily request failed: {type(exc).__name__}: {exc}",
            })
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
        logging.info("Evidence found: %s", evidence)
        emit({
            "type": "trace",
            "agent": "controller",
            "status": "started",
            "message": "Summarizing the agent evidence into the research report.",
            "sections": [
                {
                    "agent": section.get("agent"),
                    "probability": float(section.get("probability", 0) or 0),
                }
                for section in sections
            ],
        })

        report_chunks = []
        for chunk in self.llm.stream([
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
        ]):
            text = extract_text_content(chunk.content)
            if text:
                report_chunks.append(text)
        logging.info("chunks: %s", report_chunks)
        final_report = "".join(report_chunks)
        if not final_report.strip():
            emit({
                "type": "trace",
                "agent": "controller",
                "status": "error",
                "message": "The report model returned no text.",
            })
            raise RuntimeError("The report model returned no text.")
        emit({
            "type": "trace",
            "agent": "controller",
            "status": "completed",
            "message": "Report written. The note is in the chat.",
        })
        return {"finalReport": final_report}
    
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
 