# langchain

Plain LangChain — no LangGraph, no LlamaIndex. `langsmith_tracing.py` is the
one that's *about* LangSmith (`@traceable`, dashboard spans).

Run scripts from inside this folder:

```
cd langchain
python chain.py
python agent.py
```

None of these need data files.

Ordered simplest → most involved:

| # | file | level | what it does |
|---|---|---|---|
| 1 | `memory.py` | basic | 3-turn chat; "memory" is just appending to a message list (`InMemoryChatMessageHistory`) and re-sending it |
| 2 | `few_shot_prompting.py` | basic | same instruction + question, prepend 0 / 1 / 3 worked `(human, ai)` example turns; watch a made-up tag format lock in as the "shots" are added |
| 3 | `chain.py` | basic | **CHAIN** — two fixed steps: triage the message into a Pydantic schema (`with_structured_output`), then pipe those fields into a second prompt that drafts a reply. Structured output **and** prompt chaining, no loop. |
| 4 | `langsmith_tracing.py` | intermediate | a chain, tracing to LangSmith via env vars; `@traceable` on a plain function so it shows as a span |
| 5 | `agent.py` | intermediate | **AGENT** — `create_tool_calling_agent` + `AgentExecutor` runs the model→tool→model loop; 2 throwaway tools; needs `langchain-classic` |

`tracing.py` is a helper, not a lesson: a callback that prints the exact prompt
sent to the model. `from tracing import TRACE_CONFIG`, then pass `config=TRACE_CONFIG`
to any `.invoke()`.

## Tracing

Every script here does `import _trace` first — it loads `practice/.env` and
turns on LangSmith tracing (project from `LANGSMITH_PROJECT`). Token counts
ride on the LLM spans. No env vars set = it no-ops.

