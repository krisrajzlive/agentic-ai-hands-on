# COMPRESS THE USER'S PROMPT TO FIT A TOKEN BUDGET, THEN QUERY THE INDEX.
#
# The user types a long, rambling question. Our LLM API is rate-limited by
# tokens, so before spending them we check the size — if the prompt is over
# budget, ask the LLM to rewrite it as a short query that keeps the same
# meaning. Then run that (shorter) query against the index.
#
#   long prompt --> over budget? --> [LLM rewrites it tight] --> query_engine --> answer
#                        |
#                        +-- already small --> query_engine --> answer
#
# run:  cd llamaindex && python prompt_compression.py     (needs Ollama running)

import _trace  # noqa: F401  -- enables LangSmith tracing

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="llama3.1", request_timeout=120)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

TOKEN_BUDGET = 30   # most tokens we'll spend on the query text

def approx_tokens(text: str) -> int:
    return len(text) // 4          # ~4 chars per token rule of thumb

def compress(prompt: str) -> str:
    out = Settings.llm.complete(
        "Rewrite the question as a short search query. Keep every key term, name "
        "and constraint; drop the pleasantries. One line, no preamble.\n\n" + prompt
    )
    return str(out).strip()


index = VectorStoreIndex.from_documents(SimpleDirectoryReader("docs").load_data())
query_engine = index.as_query_engine(similarity_top_k=3)

LONG_PROMPT = (
    "Hi there, I hope you can help me out. I've been trying to get my head around "
    "how this backup product handles keeping old data around, and what I really "
    "need to know is the longest possible amount of time I can configure the "
    "system to hold on to a recovery point before it gets pruned, and also why "
    "that particular ceiling exists. Thanks so much!"
)

n = approx_tokens(LONG_PROMPT)

if n <= TOKEN_BUDGET:
    query = LONG_PROMPT
    print(f"[budget] prompt ~{n} tok, within {TOKEN_BUDGET} -- using as-is")
else:
    query = compress(LONG_PROMPT)
    print(f"[budget] prompt ~{n} tok, over {TOKEN_BUDGET} -- compressed to ~{approx_tokens(query)} tok")
    print(f"[compressed] {query}")

print("\nANSWER:", query_engine.query(query))
