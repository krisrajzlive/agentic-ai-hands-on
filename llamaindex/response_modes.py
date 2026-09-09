# RESPONSE MODES — same retrieved nodes, different ways to turn them into an answer.
#
#   compact         (default) pack as many nodes as fit into ONE prompt, answer once
#   refine          answer from node 1, then "here's node 2, improve the answer", ...
#                   -> one LLM call PER node, slower, but no context-window limit
#   tree_summarize  summarise the nodes in pairs, up a tree — best for
#                   "summarise / give me an overview of everything"
#
# run:  cd llamaindex && python response_modes.py     (needs Ollama running; makes several LLM calls)

import _trace  # noqa: F401  -- enables LangSmith tracing
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# context_window=8192: without it, llama-index asks Ollama for the model's full
# trained context (131072 for llama3.1) as num_ctx, and Ollama tries to allocate
# a ~23 GB KV/compute buffer -> "out-of-memory during startup". 8k is plenty here.
Settings.llm = Ollama(model="llama3.1", request_timeout=120, context_window=8192)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

index = VectorStoreIndex.from_documents(SimpleDirectoryReader("docs").load_data())
query = "Give an overview of retention, restore, and encryption for this backup product."

# The same retrieved nodes, but different ways to turn them into an answer. Compact is the default.
for mode in ("compact", "refine", "tree_summarize"):
    qe = index.as_query_engine(similarity_top_k=3, response_mode=mode)
    print(f"\n===== {mode} =====")
    print(qe.query(query))
