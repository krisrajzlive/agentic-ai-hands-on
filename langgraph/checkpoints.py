# 3. PERSISTENCE & CHECKPOINTS
#
# A checkpointer saves the graph's state after every step, keyed by thread_id.
# So each thread_id remembers exactly how far it got — independently.
#
# Order pipeline: receive -> charge -> pack -> ship. Four orders:
#   1001  `receive` calls interrupt() — a human must approve it, then we resume  -> PAUSED
#   2002  its `charge` fails twice then succeeds (RetryPolicy), finishes         -> COMPLETED
#   3003  runs straight through, no drama                                        -> COMPLETED
#   4004  its `charge` fails every time, retries are exhausted                   -> FAILED
#
# Each order gets an explicit run-state label so you can tell at a glance
# whether it's done, waiting, or dead — not just which node it last cleared.
#
# InMemorySaver keeps all this in RAM. SqliteSaver puts it in a file so it
# survives a restart (see state_sqllite.py).

import _trace  # noqa: F401  -- enables LangSmith tracing
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy, interrupt, Command


class State(TypedDict, total=False): # total=False means all keys are optional, so we can add them as we go
    order_id: str
    status: str
    log: list[str]

def receive(s: State) -> dict:
    oid = s["order_id"]
    if oid == "1001":                               # this order needs a human to approve it
        answer = interrupt({"order": oid, "ask": "accept this order? (yes/no)"})
        if answer != "yes":
            return {"status": "rejected", "log": s.get("log", []) + ["order rejected by staff"]}
    return {"status": "received", "log": s.get("log", []) + ["order received"]}

# counts attempts per order so charge() can fail the first couple of times.
# RetryPolicy re-calls the SAME node, so the counter has to live outside it.
_charge_attempts: dict[str, int] = {}

def charge(s: State) -> dict:
    oid = s["order_id"]
    _charge_attempts[oid] = _charge_attempts.get(oid, 0) + 1
    n = _charge_attempts[oid]

    if oid == "2002" and n < 3:                     # flaky gateway — recovers on the 3rd try
        print(f"  charge {oid}: attempt {n} failed (gateway timeout)")
        raise RuntimeError("payment gateway timeout")

    if oid == "4004":                              # this card is just declined, every time
        print(f"  charge {oid}: attempt {n} failed (card declined)")
        raise RuntimeError("card declined")

    return {"status": "charged", "log": s.get("log", []) + [f"card charged $49.00 (attempt {n})"]}

def pack(s: State) -> dict:
    return {"status": "packed", "log": s.get("log", []) + ["items boxed"]}

def ship(s: State) -> dict:
    return {"status": "shipped", "log": s.get("log", []) + ["handed to courier"]}

graph = StateGraph(State)

graph.add_node("receive", receive)
graph.add_node(                                     # retry charge up to 3x on a RuntimeError
    "charge", charge,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.2, retry_on=(RuntimeError,)),
)
graph.add_node("pack", pack)
graph.add_node("ship", ship)

graph.add_edge(START, "receive")
graph.add_conditional_edges(                        # rejected -> stop, otherwise carry on
    "receive",
    lambda s: END if s.get("status") == "rejected" else "charge",
    ["charge", END],
)
graph.add_edge("charge", "pack")
graph.add_edge("pack", "ship")
graph.add_edge("ship", END)

app = graph.compile(checkpointer=InMemorySaver())

# ---- draw the graph ----
try:
    app.get_graph().draw_mermaid_png(output_file_path="checkpoints_graph.png")
    print("wrote checkpoints_graph.png")
except Exception as e:                                  # no internet -> show the mermaid text
    print(f"couldn't write the png ({e}); mermaid source:\n")
    print(app.get_graph().draw_mermaid())

c1 = {"configurable": {"thread_id": "order-1001"}}
c2 = {"configurable": {"thread_id": "order-2002"}}
c3 = {"configurable": {"thread_id": "order-3003"}}
c4 = {"configurable": {"thread_id": "order-4004"}}

# set : run the four orders, each in its own thread_id. The checkpointer remembers where each one is.
failed: set[str] = set()

# execute the four orders, each in its own thread_id. The checkpointer remembers where each one is.
paused = app.invoke({"order_id": "1001", "log": []}, c1)   # runs receive, hits interrupt(), pauses
app.invoke({"order_id": "2002", "log": []}, c2)            # charge retries, then finishes
app.invoke({"order_id": "3003", "log": []}, c3)            # straight through

try:
    app.invoke({"order_id": "4004", "log": []}, c4)        # charge never succeeds
except Exception as e:
    failed.add("order-4004")
    print(f"  charge order-4004: gave up after all retries ({e})")

# derive a one-word lifecycle label for a thread from its checkpoint:
#   FAILED     - a node raised and used up its retries; we caught it in Python
#                (LangGraph doesn't flag this on the snapshot, hence the `failed` set)
#   COMPLETED  - snap.next is empty -> the graph reached END, nothing pending
#   PAUSED     - snap.next has a node -> stopped early (interrupt) and can resume
def run_state(cfg, snap) -> str:
    if cfg["configurable"]["thread_id"] in failed:
        return "FAILED"
    return "COMPLETED" if not snap.next else "PAUSED"

# print a summary of where each order is in the pipeline. The checkpointer remembers
# the state of each thread_id, so we can resume them independently.
print("\nwhere each order stands:")
for cfg in (c1, c2, c3, c4):
    tid = cfg["configurable"]["thread_id"]
    snap = app.get_state(cfg) # 
    next_step = snap.next[0] if snap.next else "-"
    print(f"  {tid}: [{run_state(cfg, snap):9}] status={snap.values.get('status', '-'):9} next={next_step}")

# order-2002's charge only landed on the 3rd try — the log records which attempt
print("\norder-2002 log:")
for line in app.get_state(c2).values["log"]:
    print(f"  - {line}")

# order-1001 is paused INSIDE `receive`, waiting on a human. Read the payload we
# passed to interrupt(), ask, and resume with Command(resume=<answer>).
tid = c1["configurable"]["thread_id"]
question = paused["__interrupt__"][0].value
answer = input(f"\n{tid} — {question['ask']} ")
app.invoke(Command(resume=answer), c1)

snap = app.get_state(c1)
print(f"resumed {tid} -> {snap.values['status']}")
for line in snap.values["log"]:
    print(f"  - {line}")
