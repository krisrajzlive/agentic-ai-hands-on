# ===== LangChain + LangSmith =====
#
# The whole point: you DON'T change your chain to get tracing. The LANGSMITH_*
# vars in practice/.env (loaded by _trace) send every .invoke() to the dashboard
# at smith.langchain.com. For your OWN plain functions, add @traceable so they
# show up as spans too. No vars set -> still runs, just traces nothing.

import _trace  # noqa: F401  -- loads practice/.env so LANGSMITH_* is set

from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


@traceable   # plain function -> shows up as its own span when tracing is on
def normalise(question: str) -> str:
    return question.strip().rstrip("?").capitalize() + "?"


# an ordinary chain — nothing in here knows or cares about LangSmith
chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Answer in one short sentence."),
        ("human", "{question}"),
    ])
    | ChatOllama(model="llama3.1", temperature=0)
    | StrOutputParser()
)

q = normalise("  what is langsmith used for  ")
print("Q:", q)
print("A:", chain.invoke({"question": q}))
