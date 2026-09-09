# RETRIEVE then RERANK — two-stage retrieval.
#
# Stage 1: the vector retriever pulls the top-k by embedding similarity — fast,
#          but a bit blunt.
# Stage 2: a "node postprocessor" trims / reorders that shortlist:
#   - SimilarityPostprocessor : drop anything below a score cutoff
#   - LLMRerank               : ask the LLM to reorder by true relevance
#                               (slower, sharper — a second opinion)
#
# run:  cd llamaindex && python retriever_rerank.py     (needs Ollama running)

import _trace  # noqa: F401  -- enables LangSmith tracing
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Settings is a global config bag — set the models once, everything picks them up
Settings.llm = Ollama(model="llama3.1", request_timeout=120)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

index = VectorStoreIndex.from_documents(SimpleDirectoryReader("docs").load_data())
retriever = index.as_retriever(similarity_top_k=4)

query = "can support reset my lost backup passphrase?"
hits = retriever.retrieve(query)

print("stage 1 — vector retriever, top 4:")
for h in hits:
    print(f"  {h.score:.3f}  {h.metadata['file_name']}")

# Stage 2a: SimilarityPostprocessor — drop anything below a score cutoff
kept = SimilarityPostprocessor(similarity_cutoff=0.6).postprocess_nodes(hits)
print(f"\nstage 2a — SimilarityPostprocessor(cutoff=0.6): kept {len(kept)} of {len(hits)}")
for h in kept:
    print(f"  {h.score:.3f}  {h.metadata['file_name']}")

# Stage 2b: LLMRerank — ask the LLM to reorder by true relevance
reranked = LLMRerank(top_n=1).postprocess_nodes(hits, query_str=query)
print("\nstage 2b — LLMRerank, top 1 (scores are the LLM's, not cosine):")

# query re-ranked 
resp = index.as_query_engine(reranked=reranked).query(query)
print("\nFINAL ANSWER:", resp)
for h in reranked:
    print(f"  {h.score:.3f}  {h.metadata['file_name']}")
