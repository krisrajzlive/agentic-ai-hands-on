# side-effect import. Three jobs:
#   1. load practice/.env so LANGSMITH_* is available.
#   2. bridge LlamaIndex -> LangSmith: OpenInference instruments LlamaIndex's
#      calls as OpenTelemetry spans, an OTLP exporter ships them to LangSmith's
#      OTEL endpoint. Real token counts ride the LLM spans there.
#   3. attach a TokenCountingHandler and print the totals on exit, so you also
#      see tokens right in the terminal (this count is a tiktoken approximation).
#
# Just `import _trace` at the top of a script. No-ops cleanly with no env vars.

import atexit
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

load_dotenv(Path(__file__).parent.parent / ".env")   # LANGSMITH_* from practice/.env

# --- terminal token counter (always on) ---
_tok = TokenCountingHandler()
Settings.callback_manager = CallbackManager([_tok])


@atexit.register
def _print_tokens() -> None:
    if _tok.total_llm_token_count or _tok.total_embedding_token_count:
        print(
            f"\n[tokens ~approx] llm prompt={_tok.prompt_llm_token_count} "
            f"completion={_tok.completion_llm_token_count} "
            f"llm_total={_tok.total_llm_token_count}  "
            f"embeddings={_tok.total_embedding_token_count}"
        )


# --- LangSmith bridge (only if the env vars are set) ---
if os.getenv("LANGSMITH_TRACING") == "true" and os.getenv("LANGSMITH_API_KEY"):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    base = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    project = os.getenv("LANGSMITH_PROJECT", "default")

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(
        endpoint=f"{base}/otel/v1/traces",
        headers={"x-api-key": os.environ["LANGSMITH_API_KEY"], "Langsmith-Project": project},
    )))
    LlamaIndexInstrumentor().instrument(tracer_provider=provider)
    print(f"[_trace] LlamaIndex -> LangSmith project {project!r}")
