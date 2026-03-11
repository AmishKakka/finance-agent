import { StateGraph, StateSchema, ReducedValue, GraphNode, Send } from "@langchain/langgraph";
import * as tavilyCore from "@tavily/core";
import * as zod from "zod";

const AgentState = new StateSchema ({
    agentName: zod.enum(["NewsAgent", "FinancialStmtAgent", "OutlookAgent", "SectorAgent"])
});


const NewsAgent: GraphNode<typeof AgentState> = async (state) => {
    // Declare your tavily client and get recent news around the company
    const client = tavilyCore.tavily ({ apiKey: "tvly-YOUR_API_KEY" });
    const response = await client.search("recent news around APPL company", {
                                        includeAnswer: "basic",
                                        topic: "news",
                                        searchDepth: "basic"
                                    });
};
