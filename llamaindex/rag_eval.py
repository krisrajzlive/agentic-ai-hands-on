# RAG EVALUATION — retrieval metrics + generation metrics, on this folder's own KB.
#
# Two families, two different kinds of "ground truth":
#   RETRIEVAL   Hit Rate / MRR / MAP / NDCG — did we fetch the right chunk, ranked
#               well? Ground truth = "this query's answer lives in THIS node."
#               Pure math, no LLM judge needed.
#   GENERATION  Faithfulness / Relevancy / Correctness / Semantic Similarity — is
#               the ANSWER good? Mostly LLM-as-judge (except semantic similarity,
#               which is just embedding cosine against a reference answer).
#
# Together they triangulate where a RAG pipeline breaks:
#   bad retrieval scores                -> fix the retriever
#   good retrieval, bad faithfulness    -> the LLM is ignoring/going past its context
#   good faithfulness, bad correctness  -> the retrieved context itself was wrong
#
# run:  cd llamaindex && python rag_eval.py     (needs Ollama running; several LLM calls, slow on CPU)

import _trace  # noqa: F401  -- enables LangSmith tracing

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import (
    RetrieverEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="llama3.1", request_timeout=300, context_window=8192)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# ---- the same little KB the other llamaindex scripts use ----
# explicit input_files (not the whole docs/ folder) so we never accidentally pull
# in docs/2025ar.pdf if you've dropped one there for milvus_persist.py.
docs = SimpleDirectoryReader(
    input_files=["docs/retention.md", "docs/restore.md", "docs/encryption.md"]
).load_data()
nodes = SentenceSplitter(chunk_size=128, chunk_overlap=20).get_nodes_from_documents(docs)
index = VectorStoreIndex(nodes)
retriever = index.as_retriever(similarity_top_k=2)
query_engine = index.as_query_engine(similarity_top_k=2)

# ---- the eval set: a query, a short unique phrase that pins down the ONE node
# that should be retrieved (our retrieval ground truth), and a reference answer
# (our generation ground truth, for correctness + semantic similarity) ----
EVAL_SET = [
    {
        "query": "What is the maximum retention period?",
        "anchor": "immutable storage",
        "reference": "The maximum retention period is 7 years, the limit of the underlying immutable storage.",
    },
    {
        "query": "Can support reset a lost customer-managed passphrase?",
        "anchor": "cannot reset it",
        "reference": "No. If a customer-managed passphrase is lost the data is unrecoverable; support cannot reset it.",
    },
    {
        "query": "What happens if you restore a file to its original location and it still exists?",
        "anchor": ".restored suffix",
        "reference": "Northwind writes the restored copy next to the existing file with a .restored suffix, so nothing is overwritten.",
    },
]


def gold_node_id(anchor: str) -> str:
    """The one node whose text contains `anchor` -- the "correct" retrieval target."""
    return next(n.node_id for n in nodes if anchor in n.text)


# ===== 1. RETRIEVAL METRICS: Hit Rate, MRR, MAP (as per-query "ap"), NDCG =====
retriever_eval = RetrieverEvaluator.from_metric_names(
    ["hit_rate", "mrr", "ap", "ndcg"], retriever=retriever
)

print("===== retrieval metrics =====")
totals = {"hit_rate": 0.0, "mrr": 0.0, "ap": 0.0, "ndcg": 0.0}
for case in EVAL_SET:
    result = retriever_eval.evaluate(query=case["query"], expected_ids=[gold_node_id(case["anchor"])])
    scores = {name: metric.score for name, metric in result.metric_dict.items()}
    for name, val in scores.items():
        totals[name] += val
    print(f"\n{case['query']!r}")
    print(f"  hit_rate={scores['hit_rate']:.2f}  mrr={scores['mrr']:.2f}  "
          f"ap={scores['ap']:.2f}  ndcg={scores['ndcg']:.2f}")

n = len(EVAL_SET)
print("\nmean over all queries (this IS 'MAP' / 'the' hit rate etc. across the eval set):")
print("  " + "  ".join(f"{name}={total / n:.2f}" for name, total in totals.items()))

# ===== 2. GENERATION METRICS: Faithfulness, Relevancy, Correctness, Semantic Similarity =====
faithfulness = FaithfulnessEvaluator()          # is the answer grounded in the retrieved context?
relevancy = RelevancyEvaluator()                # do the context + answer actually address the query?
correctness = CorrectnessEvaluator()            # LLM judge vs. a reference answer, scored 1-5
similarity = SemanticSimilarityEvaluator()      # embedding cosine vs. the reference answer, no LLM judge

print("\n\n===== generation metrics =====")
for case in EVAL_SET:
    response = query_engine.query(case["query"])   # this is the RAG pipeline actually answering

    f = faithfulness.evaluate_response(query=case["query"], response=response)
    r = relevancy.evaluate_response(query=case["query"], response=response)
    c = correctness.evaluate(query=case["query"], response=str(response), reference=case["reference"])
    s = similarity.evaluate(response=str(response), reference=case["reference"])

    print(f"\n{case['query']!r}")
    print(f"  answer: {str(response)!r}")
    print(f"  faithfulness={'pass' if f.passing else 'FAIL'}  "
          f"relevancy={'pass' if r.passing else 'FAIL'}  "
          f"correctness={c.score}/5  "
          f"semantic_similarity={s.score:.2f}")

print("\nFaithfulness / Relevancy / Correctness are LLM-as-judge -- llama3.1 grading")
print("llama3.1 here, so read these as a rough signal, not ground truth. Semantic")
print("similarity is the one score above with no judge involved at all.")
