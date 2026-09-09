# agentic-ai-hands-on

Small, runnable, heavily-commented examples for **LangChain**, **LangGraph**, and
**LlamaIndex** — all running locally against **Ollama**, no cloud LLM key needed.

Each folder has its own README with a complexity-ordered table of every script.

| folder | what's inside |
|---|---|
| [`langchain/`](langchain/README.md) | chains, structured output, memory, few-shot prompting, an agent, LangSmith `@traceable` |
| [`langgraph/`](langgraph/README.md) | state graphs, conditional edges, checkpoints & persistence, human-in-the-loop, LLM guardrails, time-travel, observability |
| [`llamaindex/`](llamaindex/README.md) | baseline RAG, chunking, rerankers, router / tool-calling / ReAct agents, Milvus persistence & multi-tenancy, CLIP + OWL-ViT vision |

## Setup

### 1. Python + dependencies (uv)

Python 3.11.

```
uv venv
uv pip install -r requirements.txt
```

### 2. Ollama

Install [Ollama](https://ollama.com), then pull the two models every script uses:

```
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 3. Environment (optional — LangSmith tracing)

```
copy .env.example .env
```

Fill in `LANGSMITH_API_KEY` (free key at <https://smith.langchain.com>). Without
it the scripts still run — they just don't trace.

### 4. Run anything

```
cd langgraph
python observability.py
```

## Milvus (only for `llamaindex/milvus_*.py`)

`milvus_persist.py` and `milvus_tenant_db.py` / `_collection.py` / `_partition.py`
need a running Milvus server.

### Install with Docker on Windows

With **Docker Desktop** running, from the repo root:

```
cd milvus
docker compose up -d
```

`milvus/docker-compose.yml` (Milvus **v2.5.15** standalone) is committed here — it
brings up three containers (`milvus-etcd`, `milvus-minio`, `milvus-standalone`)
and publishes port **19530**. First run pulls ~1 GB of images.

Check it came up:

```
docker ps --filter name=milvus
curl http://localhost:9091/healthz
```

> Don't use Milvus's `standalone_embed.sh` on Windows — it calls `sudo`, which is
> disabled here. `docker compose` (above) is the Windows path.

Stop / tear down (from the `milvus/` folder):

```
docker compose down          # keep the stored vectors
docker compose down -v       # also wipe them
```

### Milvus web UI

| | URL | notes |
|---|---|---|
| **Built-in dashboard** | <http://localhost:9091/webui/> | ships with Milvus 2.5+, nothing to install — collections, segments, config, running queries |
| **Attu** (full admin GUI) | <http://localhost:8000> | browse & query data, manage indexes, view schema |

Attu is a separate container:

```
docker run -d --name attu -p 8000:3000 -e MILVUS_URL=host.docker.internal:19530 zilliz/attu:v2.5
```

Stop it with `docker rm -f attu`.

### The PDF for `milvus_persist.py`

It indexes a PDF at `llamaindex/docs/2025ar.pdf`. That file is **gitignored**
(large, third-party) — drop any PDF there with that name to run the example.

## Tracing

Every script does `import _trace` first (a tiny per-folder shim that loads
`.env`). LangChain and LangGraph trace to LangSmith natively; LlamaIndex bridges
via OpenInference → OpenTelemetry → LangSmith's OTLP endpoint. No env vars = no-op.
