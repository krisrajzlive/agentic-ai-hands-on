# side-effect import: load practice/.env so LANGSMITH_* is set before anything
# runs — then LangChain traces to LangSmith automatically, token counts on the
# LLM spans. No-ops cleanly if the env vars aren't set.
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
