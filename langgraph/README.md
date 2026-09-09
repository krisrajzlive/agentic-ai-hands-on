# langgraph

Stateful graphs. Run from inside this folder:

```
cd langgraph
python observability.py
```

Ordered simplest → most involved:

| # | file | level | what it does |
|---|---|---|---|
| 1 | `observability.py` | basic | 2 linear nodes; `ENV=dev` → `app.stream(stream_mode="updates")` per-step, otherwise `app.invoke()` for the final state. Either way the run auto-uploads to LangSmith. |
| 2 | `state_sqllite.py` | basic | `SqliteSaver` persistence — re-run the script and `count` climbs 1, 2, 3 instead of resetting. Writes `state.db` (gitignored). |
| 3 | `conditional_app.py` | intermediate | plain edge + conditional edge (`add_conditional_edges` + path map: `router → number / text`); prints the mermaid text and writes `graph.png` |
| 4 | `guardrails.py` | intermediate | LLM guardrails — a judge model scores the request, then the model answers it, then the judge scores the answer (`with_structured_output` → `{allowed, reason}`); a "no" from either guard routes to a `blocked` node instead of raising |
| 5 | `time_travel.py` | intermediate | `get_state_history()` lists every checkpoint; rewind to one (`invoke(None, snap.config)`), or `update_state()` to fork it with a changed value and run a different future — original timeline stays intact |
| 6 | `hitlagent.py` | advanced | HITL interrupt + retry + two failure kinds: permanent (caught → `handle_error`) vs transient (uncaught → retried) |
| 7 | `checkpoints.py` | advanced | 4 orders on 4 threads through `receive→charge→pack→ship`; one pauses for HITL (`interrupt()`, yes/no on stdin), one fails twice then succeeds, one exhausts its `RetryPolicy`. Labels each `PAUSED / COMPLETED / FAILED` + prints history. All in RAM (`InMemorySaver`). |

## Notes

- **Graph picture** (`conditional_app.py`): `app.get_graph().draw_mermaid_png(output_file_path="graph.png")` renders via the mermaid.ink web API (needs internet). No internet → paste the printed `graph TD; ...` into mermaid.live.
- `state_sqllite.py` writes `state.db` here — gitignored. `checkpoints.py` keeps everything in RAM.

## Tracing

Every script here does `import _trace` first — it loads `practice/.env` and
turns on LangSmith tracing (project from `LANGSMITH_PROJECT`). Token counts
ride on the LLM spans. No env vars set = it no-ops.

Note: most of these graph demos run pure-Python nodes (no model calls), so
LangSmith shows the graph/run structure but token counts are 0. `guardrails.py`
is the exception — its judge model calls do show tokens.

