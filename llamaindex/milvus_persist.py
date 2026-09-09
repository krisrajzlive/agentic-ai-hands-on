# PERSIST EMBEDDINGS INTO MILVUS  (embed once, reuse after)
#
# First run:  the collection doesn't exist -> read the CEO's shareholder letter
#             from the Berkshire annual report, chunk it, embed with Ollama,
#             store the vectors in a new Milvus collection.
# Later runs: the collection exists -> just reconnect and query, no re-embedding.
#
#   2025ar.pdf (letter pages) -> chunks -> nomic-embed-text (768-d) -> Milvus
#
# needs a Milvus server on localhost:19530 (see the repo README for docker setup)
# and a PDF at docs/2025ar.pdf (gitignored — drop any PDF there with that name).
#
# run:  cd llamaindex && python milvus_persist.py     (Ollama + Milvus running)

import _trace  # noqa: F401  -- loads practice/.env so LANGSMITH_* is set

from pymilvus import MilvusClient
from pypdf import PdfReader

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.milvus import MilvusVectorStore

MILVUS_URI = "http://localhost:19530"
COLLECTION = "berkshire"
EMBED_DIM = 768               # nomic-embed-text output size
LETTER_PAGES = range(3, 22)   # PDF pages holding the CEO letter (0-indexed)

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3.1", request_timeout=300, context_window=8192)

exists = MilvusClient(uri=MILVUS_URI).has_collection(COLLECTION)

# overwrite=False: create the collection if missing, never drop an existing one
vector_store = MilvusVectorStore(
    uri=MILVUS_URI, collection_name=COLLECTION, dim=EMBED_DIM, overwrite=False
)

if exists:
    print(f"collection '{COLLECTION}' exists — reconnecting, no embedding")
    index = VectorStoreIndex.from_vector_store(vector_store)
else:
    print(f"collection '{COLLECTION}' not found — embedding the letter now")
    pdf = PdfReader("docs/2025ar.pdf")
    letter = "\n".join(pdf.pages[i].extract_text() or "" for i in LETTER_PAGES)
    docs = [Document(text=letter, metadata={"source": "2025ar.pdf", "section": "CEO letter"})]
    print(f"  letter: {len(letter):,} chars from {len(LETTER_PAGES)} pages")

    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64)],
    )
    print(f"  persisted -> Milvus collection '{COLLECTION}'")

# query either way. top_k=6: a compound question spreads its similarity across
# several chunks, so pull a few more or a single-fact ask (the signature) can miss.
qa = index.as_query_engine(similarity_top_k=6)
for q in [
    "Who signed the 2025 shareholder letter - his name and title?",
    "How does he describe Berkshire's performance and future outlook?",
]:
    print(f"\nQ: {q}\nA:", qa.query(q))
