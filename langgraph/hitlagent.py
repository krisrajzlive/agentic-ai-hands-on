# Python script demonstrates human in the loop, retry policy, and error handling in a state graph workflow.

import _trace  # noqa: F401  -- enables LangSmith tracing
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command, RetryPolicy

# each field is its own channel — LangGraph just merges whatever a node returns
class State(TypedDict, total=False):
    task: str
    action: str
    approved: bool
    result: str
    error: str

# stubbed out for now — in a real version this is where an LLM would decide what to do
def plan(state):
    task = state["task"]
    print(f"Planning for task: {task}")
    return {"action": task}

# this is the pause point — graph blocks here until a human answers
def human_review(state):
    print(f"Requesting human review for action: {state['action']}")
    decision = interrupt({"proposed": state["action"]})  # pauses here
    return {"approved": decision == "yes"}

# fake flaky call: fails the first two times, works the third. wanted to actually
# watch the retry policy fire instead of just trusting the config is right
_attempts = {"n": 0}

def call_flaky_service(action):
    _attempts["n"] += 1
    print(f"  (attempt {_attempts['n']})")
    if _attempts["n"] < 3:
        raise RuntimeError(f"cannot perform {action!r}: transient error")
    print(f"  {action!r} succeeded on attempt {_attempts['n']}")

# two kinds of failure here, handled on purpose in two different ways:
# - "protected branch" is permanent, retrying won't help, so I catch it myself and route to handle_error
# - the flaky-service error is transient, so I leave it uncaught and let retry_policy deal with it
def execute(state):
    print(f"Executing: {state['action']}")
    if state["action"] == "delete protected branch":
        return {"result": "error", "error": "branch is protected and cannot be deleted"}
    call_flaky_service(state["action"])
    return {"result": "success"}

# only reached when execute gives up on a permanent failure
def handle_error(state):
    print(f"Action failed, recovering gracefully: {state['error']}")
    return {"result": "failed-handled"}

graph = StateGraph(State)

graph.add_node("plan", plan)
graph.add_node("human_review", human_review)
graph.add_node(
    "execute",
    execute,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0, retry_on=(RuntimeError,)),
)
graph.add_node("handle_error", handle_error)

graph.add_edge(START, "plan")
graph.add_edge("plan", "human_review")

# "yes" -> go execute it, anything else -> just stop here
graph.add_conditional_edges("human_review", lambda s: "execute" if s["approved"] else END)

# only permanent errors go to handle_error, everything else just ends
graph.add_conditional_edges("execute", lambda s: "handle_error" if s["result"] == "error" else END)

graph.add_edge("handle_error", END)

app = graph.compile(checkpointer=MemorySaver())

# thread_id ties this run to a checkpoint so it can pause and resume later
cfg = {"configurable": {"thread_id": "1"}}

# swap this to "delete protected branch" to see the other path (straight to handle_error)
TASK = "delete stale branch"

# this blocks right at human_review and waits
out = app.invoke({"task": TASK}, cfg)

# grab the interrupt payload and actually go ask someone
question = out["__interrupt__"][0].value
answer = input(f"Approve '{question['proposed']}'? (yes/no): ")

# resume with whatever they typed
final = app.invoke(Command(resume=answer), cfg)
print(final)
