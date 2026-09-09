# ===== AGENT =====   (its opposite number is chain.py in this folder)
#
# No hand-rolled loop here. create_tool_calling_agent builds the agent,
# AgentExecutor runs the model -> tool -> model loop for me — including
# actually calling the tools. My code is just one .invoke().
#
#     chain :  prompt | model                 -> I compose the fixed steps
#     agent :  AgentExecutor(agent, tools)    -> the framework runs the
#              .invoke({"input": ...})           model/tool loop until the
#                                                model stops asking for tools
#
# The loop and the tool dispatch live inside AgentExecutor now.
# verbose=True prints each step so you can still watch it decide.
#
# NOTE: LangChain 1.x REMOVED AgentExecutor from the main `langchain` package
# (its `create_agent` replacement is LangGraph-backed). To keep this
# no-LangGraph, it now imports from `langchain-classic` — the legacy compat
# package. `uv pip install langchain-classic`.

import _trace  # noqa: F401  -- enables LangSmith tracing
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# the docstring IS the description the model sees for this tool. put the
# "when to use me" guidance HERE, per tool — that scales to hundreds of tools.
# the system message stays short and says nothing about specific tools.
@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text supplied by the user.
    Use only when the user explicitly asks how many words are in some text.
    Not for general questions."""
    return len(text.split())

@tool
def to_upper(text: str) -> str:
    """Convert a piece of text supplied by the user to uppercase and return it.
    Use only when the user explicitly asks for text to be uppercased or
    capitalised. Not for general questions."""
    return text.upper()

tools = [word_count, to_upper]

llm = ChatOllama(model="llama3.1", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use a tool only when it clearly matches "
     "what the user is asking for; otherwise just answer directly."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),   # framework threads tool calls + results in here
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# two separate runs — same executor, and the model picks a different path for each.
# (kept as separate questions rather than one compound one: llama3.1 can't reliably
#  juggle "answer from knowledge" + two tool tasks in a single turn.)
questions = [
    "What is the capital of France?",                                    # no tool needed
    "How many words are in 'the quick brown fox jumps', "
    "and what is that phrase in uppercase?",                             # needs both tools
]

for q in questions:
    print(f"\n================ Q: {q}")
    print("ANSWER:", executor.invoke({"input": q})["output"])
