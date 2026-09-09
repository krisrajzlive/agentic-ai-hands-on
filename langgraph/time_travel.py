# TIME TRAVEL — inspect a past checkpoint, fork it, run a different future.
#
# The checkpointer saves a snapshot after EVERY step. Three moves:
#   1. app.get_state_history(cfg)  -> every snapshot, newest first. Each carries
#      a checkpoint_id in snap.config.
#   2. app.invoke(None, <old snap.config>)  -> REPLAY the graph forward from that
#      checkpoint (deterministic — same inputs, same path).
#   3. app.update_state(<old snap.config>, {..})  -> fork: same past, a changed
#      value, a brand-new branch. invoke(None, ...) then runs that branch.
#
#   START -> add_two -> times_ten -> END

import _trace  # noqa: F401  -- loads practice/.env so LANGSMITH_* is set
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict, total=False):
    x: int
    steps: list[str]


def add_two(s: State) -> dict:
    return {"x": s["x"] + 2, "steps": s["steps"] + ["add_two"]}


def times_ten(s: State) -> dict:
    return {"x": s["x"] * 10, "steps": s["steps"] + ["times_ten"]}


graph = StateGraph(State)
graph.add_node("add_two", add_two)
graph.add_node("times_ten", times_ten)

graph.add_edge(START, "add_two")
graph.add_edge("add_two", "times_ten")
graph.add_edge("times_ten", END)

app = graph.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "demo"}}

# ---- 1. a normal run: 1 -> (+2) 3 -> (*10) 30 ----
final = app.invoke({"x": 1, "steps": []}, cfg)
print(f"original run: x={final['x']}  steps={final['steps']}")

# ---- 2. the history: one snapshot per step, newest first ----
print("\ncheckpoint history (newest first):")
history = list(app.get_state_history(cfg))
for snap in history:
    ckpt = snap.config["configurable"]["checkpoint_id"][-8:]
    nxt = snap.next[0] if snap.next else "END"
    x = snap.values.get("x", "-")
    print(f"  ...{ckpt}  x={x!s:>4}  next={nxt}")

# ---- 3. rewind to the snapshot taken AFTER add_two but BEFORE times_ten ----
before_times_ten = next(s for s in history if s.next == ("times_ten",))
print(f"\nrewound to ...{before_times_ten.config['configurable']['checkpoint_id'][-8:]} "
      f"(x={before_times_ten.values['x']}, about to run times_ten)")

# ---- 4. fork it: overwrite x at that point, then run forward ----
forked_cfg = app.update_state(before_times_ten.config, {"x": 99})
branched = app.invoke(None, forked_cfg)      # only times_ten runs
print(f"forked run:   x={branched['x']}  steps={branched['steps']}")

print(f"\noriginal timeline still intact: x={history[0].values['x']}")
