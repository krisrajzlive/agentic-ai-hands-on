# ROUTER + TOOL CALLING — the LLM picks the function AND fills in its arguments.
#
# router_query_engine.py let the LLM choose an engine. Here it goes one step
# further: it calls a function and INFERS the argument values from the question.
#
# search_section(query, section) filters the index to one part of the KB. The
# model has to read "how do I get a lost file back?" and decide on its own that
# section="restore" and query="restoring a lost file" — nobody passes that in.
#
# predict_and_call() is the single-shot version: one LLM turn -> one tool call
# -> done. (research_agent.py is the multi-step version.)
#
# run:  cd llamaindex && python router_tool_calling.py     (needs Ollama running)

import _trace  # noqa: F401  -- enables LangSmith tracing
from typing import Literal

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="llama3.1", request_timeout=120)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# all three docs in one index; each node carries file_name in its metadata
index = VectorStoreIndex.from_documents(
    SimpleDirectoryReader(
        input_files=["docs/retention.md", "docs/restore.md", "docs/encryption.md"]
    ).load_data()
)

Section = Literal["retention", "restore", "encryption"]

def search_section(query: str, section: Section) -> str:
    """Answer a question using ONLY one section of the backup KB.

    query:   what to look up, in your own words
    section: which doc to look in — 'retention', 'restore', or 'encryption'
    """
    print(f"    [tool] search_section(query={query!r}, section={section!r})")
    engine = index.as_query_engine(
        similarity_top_k=3,
        filters=MetadataFilters(
            filters=[MetadataFilter(key="file_name", value=f"{section}.md")]
        ),
    )
    return str(engine.query(query))

tool = FunctionTool.from_defaults(fn=search_section)

for q in [
    "How do I get back a single file I deleted by mistake?",     # -> section='restore'
    "If I lose my customer-managed passphrase, can support help?",  # -> section='encryption'
    "How long are monthly recovery points kept?",                 # -> section='retention'
]:
    print(f"\nQ: {q}")
    # The LLM reads the tool description, infers the argument values, and calls the function.
    resp = Settings.llm.predict_and_call([tool], user_msg=q, verbose=False)
    print("A:", resp)
