# 2. CONDITIONAL BRANCHING (+ a picture of the graph)
#
# Two kinds of edge in one graph:
#   START -> greeting            a PLAIN edge — greeting always runs first
#   greeting -> number / text    a CONDITIONAL edge — router picks one
#
#   START ──► greeting ──►(router)──► number ──► END
#                                └──► text   ──► END
#
# Running this also writes graph.png — a diagram of the nodes and edges.

import _trace  # noqa: F401  -- enables LangSmith tracing
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

class State(TypedDict, total=False):
    value: str
    result: str

def greeting(state: State) -> dict:
    print(f"  hello! you gave me: {state['value']!r}")
    return {}                                   # runs for every input, changes nothing

def router(state: State) -> str:
    return "number" if state["value"].strip().lstrip("-").isdigit() else "text"

def number(state: State) -> dict:
    return {"result": f"{state['value']} is a number"}

def text(state: State) -> dict:
    return {"result": f"{state['value']!r} is text"}

def draw_graph(g) -> None:
    """Print the mermaid text and write graph.png (via mermaid.ink, needs internet)."""
    print(g.draw_mermaid())
    try:
        g.draw_mermaid_png(output_file_path="graph.png")
        print("wrote graph.png  (open it to see the diagram)")
    except Exception as e:
        print(f"couldn't write graph.png: {e}  — paste the mermaid text above into mermaid.live")

graph = StateGraph(State)

graph.add_node("greeting", greeting)
graph.add_node("number", number)
graph.add_node("text", text)

graph.add_edge(START, "greeting")               # plain edge: always start with greeting

graph.add_conditional_edges(                    # conditional edge: greeting -> number OR text
    "greeting",
    router,
    ["number", "text"],
)

graph.add_edge("number", END)
graph.add_edge("text", END)

app = graph.compile()

# ---- draw the graph ----
draw_graph(app.get_graph())

# ---- run it ----
print()
for value in ["hello", "42", "-7", "the quick brown fox"]:
    out = app.invoke({"value": value})
    print(f"{value!r:22} -> {out['result']}")
