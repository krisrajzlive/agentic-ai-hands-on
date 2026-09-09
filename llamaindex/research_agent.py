# RESEARCH ASSISTANT AGENT — reasoning over tools in MULTIPLE steps.
#
# router_tool_calling.py was single-shot: one LLM turn, one tool call, stop.
# A research agent runs a loop — call a tool, look at the result, decide the
# next call, ... — until it has enough to answer. Step 2 can depend on step 1.
#
# Question below: "how many days is the maximum retention period?"
#   step 1: search_kb("maximum retention period")  -> "7 years"
#   step 2: multiply(7, 365)                        -> 2555
#   step 3: answer
# The agent picks that order itself; we print each tool result to watch it.
#
# ReActAgent (not FunctionAgent): it drives the loop with a Thought/Action/
# Observation text protocol instead of native function-calling. A small local
# model like llama3.1 follows that far more reliably for multi-step work.
#
# run:  cd llamaindex && python research_agent.py     (needs Ollama running)

import _trace  # noqa: F401  -- enables LangSmith tracing
import asyncio

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent.workflow import ReActAgent, ToolCallResult
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="llama3.1", request_timeout=120)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

index = VectorStoreIndex.from_documents(SimpleDirectoryReader("docs").load_data())

# tool 1: search the KB
kb_tool = QueryEngineTool.from_defaults(
    query_engine=index.as_query_engine(similarity_top_k=2),
    name="search_kb",
    description="Look up a fact about the Northwind Cloud Backup product "
                "(retention, restore, encryption). Returns a short text answer.",
)


# tool 2 + 3: arithmetic the LLM shouldn't do in its head
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b


agent = ReActAgent(
    tools=[kb_tool, FunctionTool.from_defaults(multiply), FunctionTool.from_defaults(divide)],
    llm=Settings.llm,
    system_prompt="You are a research assistant. Break the question into steps. "
                  "Use search_kb for facts and the math tools for any arithmetic — "
                  "never calculate in your head.",
)

Q = "How many days is the maximum retention period? Assume 365 days per year."

async def main():
    handler = agent.run(user_msg=Q)
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            print(f"  {ev.tool_name}({ev.tool_kwargs}) -> {str(ev.tool_output).strip()}")
    print(f"\nA: {await handler}")

asyncio.run(main())