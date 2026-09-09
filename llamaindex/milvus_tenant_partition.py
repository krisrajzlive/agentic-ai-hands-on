# MULTI-TENANCY: PARTITION KEY ON tenant_id
#
# One shared collection. A scalar field `tenant_id` is marked is_partition_key,
# so Milvus hashes it into physical partitions and, when a search filters on
# tenant_id, scans only that tenant's partition. Scales to millions of tenants
# with no per-tenant objects to manage. Isolation is logical — your app MUST
# always add the tenant_id filter.
#
# run:  cd llamaindex && python milvus_tenant_partition.py     (Milvus + Ollama running)

import _trace  # noqa: F401  -- loads practice/.env so LANGSMITH_* is set

from pymilvus import MilvusClient, DataType
from llama_index.embeddings.ollama import OllamaEmbedding

URI = "http://localhost:19530"
NAME = "kb_shared"
DIM = 768
emb = OllamaEmbedding(model_name="nomic-embed-text")

DOCS = {
    "acme":   ["Acme's return window is 30 days.", "Acme ships only within the US."],
    "globex": ["Globex's return window is 90 days.", "Globex ships worldwide."],
}
QUESTION = "What is the return window?"

client = MilvusClient(uri=URI)
if client.has_collection(NAME):
    client.drop_collection(NAME)

# --- schema: tenant_id is the partition key ---
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
schema.add_field("tenant_id", DataType.VARCHAR, max_length=64, is_partition_key=True)
schema.add_field("text", DataType.VARCHAR, max_length=1024)

index_params = client.prepare_index_params()
index_params.add_index("vector", metric_type="COSINE")
client.create_collection(NAME, schema=schema, index_params=index_params)

# --- write: every row carries its tenant_id; Milvus shards on it ---
rows = [
    {"vector": emb.get_text_embedding(t), "tenant_id": tenant, "text": t}
    for tenant, lines in DOCS.items()
    for t in lines
]
client.insert(NAME, rows)

# --- read: the tenant_id filter restricts the search to acme's partition ---
hits = client.search(
    NAME, data=[emb.get_query_embedding(QUESTION)], limit=1,
    filter='tenant_id == "acme"', output_fields=["text", "tenant_id"],
)
print("filter tenant_id == acme ->", hits[0][0]["entity"])
print("(drop the filter and this same collection returns globex's rows too)")
