import _trace  # noqa: F401  -- enables LangSmith tracing
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0)
history = InMemoryChatMessageHistory(messages=[SystemMessage("You are a helpful assistant.")])

def ask(text):
    history.add_user_message(text)
    reply = llm.invoke(history.messages)
    history.add_ai_message(reply.content)
    return reply.content

print(ask("Hello, my name is Raj and I like Python."))
print(ask("What did I just tell you?"))
print(ask("What is my favorite programming language?"))

for m in history.messages:
    print(f"{m.type}: {m.content}")