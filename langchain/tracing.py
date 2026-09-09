# quick local "tracing" — same idea as LangSmith's hosted trace view (seeing
# exactly what prompt hit the model), just printed to the terminal instead.
# no account or API key needed for this.
#
# pass config=TRACE_CONFIG into any .invoke() and it fires for every chat-model
# call inside it, no matter how deeply nested — plain chain, agent loop,
# memory-wrapped chain, doesn't matter. langchain passes callbacks down through
# the whole runnable graph on its own, so I don't have to wire this into every
# piece by hand.

from langchain_core.callbacks import BaseCallbackHandler

class PromptTracer(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n----- PROMPT SENT TO LLM -----")
        for msg in messages[0]:   # messages: list[list[BaseMessage]] (batch of 1 here)
            print(f"[{msg.type}] {msg.content}")
        print("-------------------------------")

TRACE_CONFIG = {"callbacks": [PromptTracer()]}   # pass as config=TRACE_CONFIG to any .invoke()
