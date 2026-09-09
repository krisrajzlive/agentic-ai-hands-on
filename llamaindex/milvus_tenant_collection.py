# MULTI-TENANCY: A COLLECTION PER TENANT
#
# One database, one collection per tenant (kb_acme, kb_globex). Each has its own
# schema, index, and dimension, so tenants can even use different embedding
# models. You query a specific collection, so there's nothing to filter. Scales
# to hundreds / low thousands of tenants before per-collection overhead bites.
#
# run:  cd llamaindex && python milvus_tenant_collection.py     (Milvus + Ollama running)

import _trace  # noqa: F401  -- loads practice/.env so LANGSMITH_* is set

from pymilvus import MilvusClient
from llama_index.embeddings.ollama import OllamaEmbedding

URI = "http://localhost:19530"
DIM = 768
emb = OllamaEmbedding(model_name="nomic-embed-text")

DOCS = {
    "acme":   ["Acme's return window is 30 days.", "Acme ships only within the US."],
    "globex": ["Globex's return window is 90 days.", "Globex ships worldwide."],
}
QUESTION = "What is the return window?"

client = MilvusClient(uri=URI)

# --- write: one collection named per tenant ---
for tenant, lines in DOCS.items():
    name = f"kb_{tenant}"
    if client.has_collection(name):
        client.drop_collection(name)
    client.create_collection(name, dimension=DIM, auto_id=True)
    client.insert(name, [{"vector": emb.get_text_embedding(t), "text": t} for t in lines])

# --- read: point the search at acme's collection; globex's is never touched ---
hits = client.search("kb_acme", data=[emb.get_query_embedding(QUESTION)],
                     limit=1, output_fields=["text"])
print("query kb_acme ->", hits[0][0]["entity"]["text"])
print("collections:", [c for c in client.list_collections() if c.startswith("kb_")])
