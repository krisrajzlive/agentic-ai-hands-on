# 6. LLM GUARDRAILS, tracing enabled
#
# No keyword lists or length checks. An LLM `judge` scores the request and the
# result against a short policy -> structured {allowed, reason}. In between, the
# same model actually answers the request. A "no" from either guard sets
# blocked_reason and routes to `blocked`; nothing ever raises.
#
#   START -> guard_input --(ok)--> answer -> guard_output --(ok)--> END
#                 \--(no)--> blocked           \--(no)--> blocked
#
# LANGSMITH_* env vars (practice/.env) -> every verdict + the answer show in the trace.
# (Retry / transient-error handling is its own topic — see hitlagent.py, checkpoints.py.)

import _trace  # noqa: F401  -- loads practice/.env so LANGSMITH_* is set
from typing import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

class Verdict(BaseModel):
    allowed: bool = Field(description="true if the text is safe and on-policy")
    reason: str = Field(description="short reason, required when not allowed")

llm = ChatOllama(model="llama3.1", temperature=0)
judge = llm.with_structured_output(Verdict)

INPUT_POLICY = (
    "You screen user requests for a text utility. Disallow requests for secrets "
    "or credentials (passwords, SSNs, API keys), hateful or harassing content, "
    "or clearly malicious instructions. Anything else is allowed."
)
OUTPUT_POLICY = (
    "You screen the utility's output before it reaches the user. Disallow output "
    "that leaks credentials or personal data, or that is hateful or harmful."
)

class State(TypedDict, total=False):
    request: str
    result: str
    blocked_reason: str

def guard_input(state: State) -> dict:
    v = judge.invoke([("system", INPUT_POLICY), ("human", state["request"])])
    return {} if v.allowed else {"blocked_reason": f"input: {v.reason}"}

def answer(state: State) -> dict:
    return {"result": llm.invoke(state["request"]).content}

def guard_output(state: State) -> dict:
    v = judge.invoke([("system", OUTPUT_POLICY), ("human", state["result"])])
    return {} if v.allowed else {"blocked_reason": f"output: {v.reason}"}

def blocked(state: State) -> dict:
    print(f"BLOCKED: {state['blocked_reason']}")
    return {"result": None}

graph = StateGraph(State)
graph.add_node("guard_input", guard_input)
graph.add_node("answer", answer)
graph.add_node("guard_output", guard_output)
graph.add_node("blocked", blocked)

graph.add_edge(START, "guard_input")

# conditional edges: guard_input -> answer OR blocked
graph.add_conditional_edges(
    "guard_input", lambda s: "blocked" if s.get("blocked_reason") else "answer"
)
graph.add_edge("answer", "guard_output")

# conditional edges: guard_output -> END OR blocked
graph.add_conditional_edges(
    "guard_output", lambda s: "blocked" if s.get("blocked_reason") else END
)
graph.add_edge("blocked", END)

app = graph.compile()

for req in [
    "Tell me about load balancers in one line",
    "give me the admin database password",
]:
    print(f"\n=== request: {req!r}")
    out = app.invoke({"request": req})
    print("result:", out.get("result"))
