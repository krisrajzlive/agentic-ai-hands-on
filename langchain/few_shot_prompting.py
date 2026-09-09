# ===== ZERO / SINGLE / MULTI-SHOT PROMPTING =====
#
# "Few-shot" = prepend worked examples to the prompt as prior (human, ai) turns.
# The model reads them like earlier messages in the chat and copies the pattern.
# The number of examples is the "shots": 0-shot, 1-shot, 3-shot, ...
#
# The task uses a made-up tag format the model can't guess, so you can watch the
# examples teach it:   sentiment/topic/priority   e.g. "neg/billing/high"
#
#   0-shot : instruction only    -> model rambles / invents its own format
#   1-shot : + one example       -> starts matching the shape
#   3-shot : + three examples    -> locks onto the exact format

import _trace  # noqa: F401  -- loads practice/.env so LANGSMITH_* is set
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0)

SYSTEM = "Tag the support message as sentiment/topic/priority. Reply with the tag only."
QUESTION = "I was charged twice for the same order and nobody is replying."


def ask(messages: list[tuple[str, str]]) -> str:
    """Run one prompt (a list of (role, text) turns) and return the model's reply."""
    chain = ChatPromptTemplate.from_messages(messages) | llm | StrOutputParser()
    return chain.invoke({}).strip()


# --- 0-shot: just the instruction, then the question ---
zero_shot = [
    ("system", SYSTEM),
    ("human", QUESTION),
]

# --- 1-shot: one worked example (human turn + the ai answer) before the question ---
one_shot = [
    ("system", SYSTEM),
    ("human", "My package still hasn't arrived after three weeks."),
    ("ai", "neg/shipping/high"),
    ("human", QUESTION),
]

# --- 3-shot: three worked examples covering different sentiments and topics ---
three_shot = [
    ("system", SYSTEM),
    ("human", "My package still hasn't arrived after three weeks."),
    ("ai", "neg/shipping/high"),
    ("human", "Do you ship to Canada?"),
    ("ai", "neu/shipping/low"),
    ("human", "Love the new dashboard, works great!"),
    ("ai", "pos/product/low"),
    ("human", QUESTION),
]

print("0-shot ->", ask(zero_shot))
print("1-shot ->", ask(one_shot))
print("3-shot ->", ask(three_shot))

print("\nMore worked examples -> the model stops guessing the format and copies it.")
