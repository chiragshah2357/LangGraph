from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
import os
import sys
from dotenv import load_dotenv

# Windows consoles default to cp1252 and choke on characters the model emits.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

for _key in ("GROQ_API_KEY", "LANGCHAIN_API_KEY"):
    _val = os.getenv(_key)
    if _val:
        os.environ[_key] = _val


class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

model=ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def make_default_graph():
    graph_workflow=StateGraph(State)

    def call_model(state):
        return {"messages":[model.invoke(state['messages'])]}
    
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    agent=graph_workflow.compile()
    return agent

def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        if getattr(state["messages"][-1], "tool_calls", None):
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    return agent

if __name__ == "__main__":
    agent = make_alternative_graph()
    result = agent.invoke({"messages": [HumanMessage(content="What is 12.5 + 7.25? Use the add tool.")]})
    for message in result["messages"]:
        message.pretty_print()

