# BASELINE RAG — the simplest LlamaIndex pipeline.
#
#   docs -> nodes -> VectorStoreIndex -> query_engine -> answer (+ source nodes)
#
# Deterministic: every query runs the exact same steps. No agent, no loop.
# (contrast with agentic_rag.py, where the model decides whether to search)
#
# run:  cd llamaindex && python query_engine.py     (needs Ollama running)

import _trace  # noqa: F401  -- enables LangSmith tracing
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Settings is a global config bag — set the models once, everything picks them up
Settings.llm = Ollama(model="llama3.1", request_timeout=120)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# just ONE file — docs/ also has jobdescription.txt, a pdf resume, etc. that
# have nothing to do with this. input_files=[...] pins exactly what to index.
docs = SimpleDirectoryReader(input_files=["docs/retention.md"]).load_data()

# chunk size / overlap made explicit here (tokens). from_documents() would use
# defaults (1024 / 200) — passing a splitter as a transformation overrides that.
splitter = SentenceSplitter(chunk_size=120, chunk_overlap=20)
index = VectorStoreIndex.from_documents(docs, transformations=[splitter])
print(f"indexed {len(docs)} file -> {len(index.docstore.docs)} chunks "
      f"(chunk_size={splitter.chunk_size}, overlap={splitter.chunk_overlap})\n")

query_engine = index.as_query_engine(similarity_top_k=2)  # retrieve 2 chunks per question

resp = query_engine.query("How long are daily backups kept?")

print("ANSWER:", resp)
print("\nSOURCES (what the answer was built from):")
for node in resp.source_nodes:
    snippet = node.text[:60].strip().replace("\n", " ")
    print(f"  {node.score:.3f}  {node.metadata['file_name']}  \"{snippet}...\"")
