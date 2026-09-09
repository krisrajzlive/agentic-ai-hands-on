# ===== CHAIN =====   (its opposite number is agent.py in this folder)
#
# A chain is a FIXED pipeline. I wire the steps; every input runs them in the
# same order; the model never decides what happens next. This one has TWO LLM
# calls, and step 1's output feeds step 2:
#
#   {message}
#     -> [triage]   prompt | model | structured_output  -> TicketTriage object
#     -> [hand off]  pull its fields out for the next prompt
#     -> [reply]    prompt | model | text               -> a draft reply string
#
# Two things on show at once:
#   - STRUCTURED OUTPUT : step 1 returns a validated Pydantic object, not free text
#   - PROMPT CHAINING   : step 2's prompt is built from step 1's result
#
# (An AGENT is the opposite: the model decides at runtime whether to call a
#  tool, which one, and whether to loop again. See agent.py.)

import _trace  # noqa: F401  -- enables LangSmith tracing
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


class TicketTriage(BaseModel):
    category: Literal["shipping", "returns", "billing", "technical", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str = Field(description="One-sentence summary of the customer's issue")


llm = ChatOllama(model="llama3.1", temperature=0)

# --- step 1: message -> validated TicketTriage object (parsed, not text) ---
triage = (
    ChatPromptTemplate.from_messages([
        ("system", "Triage this customer support message into the given schema."),
        ("human", "{message}"),
    ])
    | llm.with_structured_output(TicketTriage)
)

# --- glue: object -> vars for the next prompt. printed so the hand-off shows ---
def hand_off(t: TicketTriage) -> dict:
    print(f"[step 1 -> step 2]  {t}\n")
    return {"category": t.category, "urgency": t.urgency, "summary": t.summary}

# --- step 2: those fields -> a drafted reply (plain string) ---
reply = (
    ChatPromptTemplate.from_messages([
        ("system", "Write a short, polite support reply. Match the urgency: {urgency}."),
        ("human", "Issue ({category}): {summary}"),
    ])
    | llm
    | StrOutputParser()
)

# the whole chain: three fixed steps, no loop, no branching
chain = triage | hand_off | reply

message = ("ur app keeps crashing every time i try to check my order status and "
           "i need to see it TODAY")

print(chain.invoke({"message": message}))
