# llamaindex

LlamaIndex examples on local Ollama. Run from inside this folder:

```
cd llamaindex
python query_engine.py       # Ollama must be running
```

All the RAG scripts read the little KB in `docs/` (`SimpleDirectoryReader("docs")`).

Ordered simplest → most involved:

| # | file | level | what it does |
|---|---|---|---|
| 1 | `chunking.py` | basic · no Ollama | `SentenceSplitter(chunk_size, chunk_overlap)` in tokens; prints every node, finds the biggest tail↔head overlap |
| 2 | `document_classifier.py` | basic · no Ollama | CLIP zero-shot, **one label per image** — embed each image in `images/` + one prompt per class, softmax the cosine scores, take the top. Needs `sentence-transformers`. |
| 3 | `objects_in_image.py` | basic · no Ollama | **open-vocab detection** — OWL-ViT (`google/owlvit-base-patch32`) finds one box + score per instance in `images/complex.jpg`; lists the hits above a threshold. Whole-image CLIP can't do this on a busy scene. Needs `transformers` + `torch`. |
| 4 | `query_engine.py` | basic | baseline RAG — `docs → VectorStoreIndex → as_query_engine() → answer + source nodes`. `SimpleVectorStore`, cosine. |
| 5 | `response_modes.py` | basic | one query, three synthesis modes: `compact` vs `refine` vs `tree_summarize` (refine/tree = one LLM call per node) |
| 6 | `prompt_compression.py` | intermediate | long user prompt over a token budget → LLM rewrites it as a short query that keeps the meaning → run it against the index |
| 7 | `retriever_rerank.py` | intermediate | retrieve top-k, then a node postprocessor: `SimilarityPostprocessor` (score cutoff) + `LLMRerank` (LLM reorders) |
| 8 | `milvus_persist.py` | intermediate | embed the Berkshire letter and store the vectors in a **Milvus** collection (`MilvusVectorStore` + `StorageContext`); the index lives in the DB, not RAM. Needs a Milvus server on `localhost:19530`. embed-once/reuse via `has_collection`. |
| 9 | `milvus_tenant_db.py` | intermediate | **multi-tenancy — database per tenant.** `MilvusClient(db_name=…)`; a client bound to one DB can't see another's collections. Strongest split, for a few big tenants. |
| 10 | `milvus_tenant_collection.py` | intermediate | **multi-tenancy — collection per tenant** (`kb_acme`, `kb_globex`). Own schema/index/dim each; you query a named collection, nothing to filter. |
| 11 | `milvus_tenant_partition.py` | intermediate | **multi-tenancy — partition key.** one shared collection, `tenant_id` field `is_partition_key`; a `tenant_id ==` filter scans only that partition. Scales to millions; isolation is logical (app must always filter). |
| 12 | `router_query_engine.py` | advanced | `RouterQueryEngine` + `PydanticSingleSelector` — the LLM reads the tool descriptions and picks a Q&A engine vs a summary engine per question. No args inferred yet. |
| 13 | `router_tool_calling.py` | advanced | `llm.predict_and_call([tool])`, single-shot — the model calls `search_section(query, section)` and works out **both** values itself |
| 14 | `research_agent.py` | advanced | `ReActAgent` multi-step loop: `search_kb` → "7 years" → `multiply(365, 7)` → 2555 → answer. Step 2 depends on step 1; streams each `ToolCall` / `ToolCallResult`. |

Rows 12–14 are a progression — build them in that order, each adds one idea (route an engine → infer the args → loop over tools).
Rows 9–11 are the three Milvus multi-tenancy strategies, weakest overhead → strongest isolation.

## Tracing

Every script here does `import _trace` first — it loads `practice/.env` and
turns on LangSmith tracing (project from `LANGSMITH_PROJECT`). Token counts
ride on the LLM spans. No env vars set = it no-ops.

`_trace` also prints a `[tokens ~approx]` line to the terminal when the run
finishes — a `TokenCountingHandler` sum (tiktoken approximation; the exact
counts are in LangSmith).
