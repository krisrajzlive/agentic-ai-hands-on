# CHUNKING — how a document becomes nodes.
#
# SentenceSplitter cuts text into chunks of roughly chunk_size TOKENS (not
# characters), never mid-sentence, with chunk_overlap tokens repeated between
# neighbours so a fact split across a boundary isn't lost.
#
# run:  cd llamaindex && python chunking.py     (no Ollama needed — no embeddings here)

import _trace  # noqa: F401  -- enables LangSmith tracing
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# just ONE file — docs/ also holds jobdescription.txt, a pdf resume, etc.
docs = SimpleDirectoryReader(input_files=["docs/retention.md"]).load_data()

splitter = SentenceSplitter(chunk_size=128, chunk_overlap=24)   # tokens
nodes = splitter.get_nodes_from_documents(docs)

print(f"{len(docs)} docs -> {len(nodes)} nodes  (chunk_size=128 tok, overlap=24)\n")
for i, node in enumerate(nodes):
    print(f"--- node {i}  [{node.metadata['file_name']}] ---")
    print(node.text)
    print()

# SentenceSplitter fills the overlap budget with whole trailing sentences, so
# how much two neighbouring chunks share varies. Find the clearest example:
def overlap_len(earlier: str, later: str) -> int:
    """Length of the longest text that is both the END of `earlier` and the
    START of `later` — i.e. how much the two chunks repeat at the seam."""
    for size in range(min(len(earlier), len(later)), 0, -1):
        if earlier[-size:] == later[:size]:
            return size
    return 0

i = max(range(len(nodes) - 1),
        key=lambda j: overlap_len(nodes[j].text, nodes[j + 1].text))
n = overlap_len(nodes[i].text, nodes[i + 1].text)
print(f"biggest overlap: node {i} -> node {i + 1} share {n} characters:")
print("  ", repr(nodes[i].text[-n:]))
