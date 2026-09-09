# Workflow using Persistent state using SqlLite

import _trace  # noqa: F401  -- enables LangSmith tracing
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

class State(TypedDict, total=False):
    count: int
    log: list[str]

def step(state: State) -> dict:
    prior_count = state.get("count", 0)          # only 0 the very first time — after that it's from the checkpoint
    count = prior_count + 1
    log = state.get("log", []) + [f"run #{count}"]
    print(f"[step] persisted count coming in: {prior_count} -> writing: {count}")
    return {"count": count, "log": log}

graph = StateGraph(State)   
graph.add_node("step", step)
graph.add_edge(START, "step")
graph.add_edge("step", END)

with SqliteSaver.from_conn_string("state.db") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": "demo-1"}}

    prior = app.get_state(cfg)
    print("Before this run, state.db has:", prior.values or "(nothing — first run)")

    result = app.invoke({}, cfg)                  # no initial values passed — comes from checkpoint
    print("After this run:", result)

print("\nRun this script again (`python state_sqllite.py`). "
      "count should climb 1, 2, 3... not reset to 0 — that's the persistence.")