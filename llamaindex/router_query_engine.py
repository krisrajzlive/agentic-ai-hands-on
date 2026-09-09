# ROUTER QUERY ENGINE — the simplest "agentic" RAG.
#
# One document, two ways to query it:
#   - a VECTOR query engine  -> good for pointed Q&A ("what's the minimum retention?")
#   - a SUMMARY query engine -> good for "sum up the whole doc"
#
# A RouterQueryEngine sits in front. For each question the LLM selector reads the
# tool descriptions and picks ONE engine to run. That's the whole agentic move
# here: no retrieval-always, the model chooses.
#
# run:  cd llamaindex && python router_query_engine.py     (needs Ollama running)

import _trace  # noqa: F401  -- enables LangSmith tracing

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="llama3.1", request_timeout=120)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# --- one document, split once, shared by both indexes ---
docs = SimpleDirectoryReader(input_files=["docs/retention.md"]).load_data()
nodes = SentenceSplitter(chunk_size=128, chunk_overlap=20).get_nodes_from_documents(docs)

vector_index = VectorStoreIndex(nodes)     # embeds nodes -> similarity search
summary_index = SummaryIndex(nodes)        # keeps every node -> walks all of them

# --- wrap each engine as a tool; the description is what the selector reads ---
qa_tool = QueryEngineTool.from_defaults(
    query_engine=vector_index.as_query_engine(similarity_top_k=2),
    name="retention_qa",
    description="Answers a SPECIFIC question about the retention policy — numbers, "
                "limits, schedules, one fact at a time.",
)
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_index.as_query_engine(response_mode="tree_summarize"),
    name="retention_summary",
    description="Use ONLY when asked to summarise or give an overview of the whole "
                "retention document.",
)

# PydanticSingleSelector makes the LLM return the choice as a structured object
# (a function call), not free-form JSON in text — small local models like
# llama3.1 are far more reliable that way. LLMSingleSelector is the text version.
router = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),   # LLM reads descriptions, returns an index
    query_engine_tools=[qa_tool, summary_tool],
    verbose=True,                                      # prints which tool it picked + why
)

for q in [
    "What is the maximum retention period and why is it capped there?",   # -> retention_qa
    "Give me a two-sentence summary of the retention document.",          # -> retention_summary
]:
    print(f"\nQ: {q}")
    print("A:", router.query(q))
