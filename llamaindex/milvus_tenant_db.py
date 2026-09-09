# MULTI-TENANCY: A DATABASE PER TENANT
#
# Strongest split. Each tenant gets its own Milvus database; a client opened on
# one database literally cannot see another's collections. Good for a handful of
# big tenants / compliance boundaries. Costs: a database's worth of overhead each.
#
# run:  cd llamaindex && python milvus_tenant_db.py     (Milvus + Ollama running)

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

# --- one database per tenant ---
admin = MilvusClient(uri=URI)
for tenant in DOCS:
    if tenant not in admin.list_databases():
        admin.create_database(tenant)

# --- write: a client bound to each tenant's database ---
for tenant, lines in DOCS.items():
    c = MilvusClient(uri=URI, db_name=tenant)
    if c.has_collection("kb"):
        c.drop_collection("kb")
    c.create_collection("kb", dimension=DIM, auto_id=True)
    c.insert("kb", [{"vector": emb.get_text_embedding(t), "text": t} for t in lines])

# --- read: this client is on acme's database, so globex is simply not reachable ---
acme = MilvusClient(uri=URI, db_name="acme")
hits = acme.search("kb", data=[emb.get_query_embedding(QUESTION)],
                   limit=1, output_fields=["text"])
print("query as acme ->", hits[0][0]["entity"]["text"])
print("(globex's 'kb' lives in a different database; acme's client can't query it)")
