# 4. OBSERVABILITY & TRACING
#
# Two ways to watch a LangGraph run:
#
#   local, no account:  app.stream(input, stream_mode="updates")
#                       -> yields {node_name: state_update} after each step
#   LangSmith:          set the LANGSMITH_* env vars (practice/.env) and every
#                       run uploads a full nested trace automatically.
#
# Below: stream step-by-step when ENV=dev, otherwise just app.invoke() for the
# final state.  ->  ENV=dev python observability.py

import _trace  # noqa: F401  -- loads practice/.env so LANGSMITH_* is set
import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

class State(TypedDict, total=False):
    n: int
    doubled: int
    label: str

def double(state):
    return {"doubled": state["n"] * 2}

def label(state):
    return {"label": f"{state['n']} -> {state['doubled']}"}

graph = StateGraph(State)

graph.add_node("double", double)
graph.add_node("label", label)

graph.add_edge(START, "double")
graph.add_edge("double", "label")
graph.add_edge("label", END)

app = graph.compile()

if os.getenv("ENV") == "dev":
    print("[dev] streaming each step:")
    for step in app.stream({"n": 21}, stream_mode="updates"):
        print("  ", step)
else:
    print("[prod] final state only:")
    print("  ", app.invoke({"n": 21}))
