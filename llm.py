"""
Language-model configuration, in one place.

Every client in the application is built here, so the model name, the
timeout, the retry budget and the temperature cannot drift apart between
the planner and the agent. Each is read from the environment, which lets
an evaluation pin a dated model without editing code.

Two things this module deliberately records about every completion:

    what the call cost, in input and output tokens and in wall time
    which model actually answered, as reported by the provider

The second matters because the default model name is a rolling alias.
An alias is convenient for a demo and useless for a benchmark, because the
model behind it can change between two runs. Recording the resolved name
means a result can always be traced to the model that produced it.
"""

import os
import time
from dataclasses import dataclass
from typing import Any

PROVIDER = "google_genai"

# Google keeps this alias pointed at its current default Flash model, so
# the app does not break when a dated name is retired. Pin a dated model
# with AI_ANALYST_MODEL when results must be reproducible.
DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2

MODEL_ENV = "AI_ANALYST_MODEL"
TIMEOUT_ENV = "AI_ANALYST_LLM_TIMEOUT"
RETRIES_ENV = "AI_ANALYST_LLM_MAX_RETRIES"
THINKING_BUDGET_ENV = "AI_ANALYST_THINKING_BUDGET"

# init_chat_model with the google_genai provider reads this variable.
API_KEY_ENV = "GOOGLE_API_KEY"
ALTERNATIVE_API_KEY_ENV = "GEMINI_API_KEY"


@dataclass(frozen=True)
class LLMConfig:
    """
    How to reach the model.

    Attributes:
        provider:
            LangChain provider name.

        model:
            Model name as requested. May be a rolling alias.

        timeout:
            Seconds to wait for one request before giving up.

        max_retries:
            Retries on transient failures, after the first attempt.

        temperature:
            Sampling temperature. Zero, because both callers want the
            same plan for the same question.

        thinking_budget:
            Reasoning tokens the model may spend before replying, or None
            for the provider's default. Planning is a small structured
            task, and the first live evaluation measured around a
            thousand output tokens and fifteen seconds per plan with the
            default. Set it and rerun the evaluation to compare.
    """

    provider: str
    model: str
    timeout: float
    max_retries: int
    temperature: float = 0.0
    thinking_budget: int | None = None

    @property
    def is_rolling_alias(self) -> bool:
        """Whether the model name can point at different models over time."""

        return "latest" in self.model.lower()


@dataclass(frozen=True)
class Completion:
    """
    One exchange with the model, and what it cost.

    Attributes:
        text:
            The reply as plain text.

        latency_ms:
            Wall time of the call, retries included.

        input_tokens:
            Prompt tokens as reported by the provider, when reported.

        output_tokens:
            Reply tokens as reported by the provider, when reported.

        model:
            The model that answered. The provider's own name when it
            reports one, otherwise the name that was requested.
    """

    text: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    model: str


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()

    try:
        value = float(raw)

    except ValueError:
        return default

    return value if value > 0 else default


def _non_negative_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()

    try:
        value = int(raw)

    except ValueError:
        return default

    return value if value >= 0 else default


def _optional_non_negative_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()

    if not raw:
        return None

    value = _non_negative_int(name, -1)

    return value if value >= 0 else None


def api_key() -> str:
    """
    Return the configured Gemini key, or an empty string.

    GEMINI_API_KEY is accepted as an alternative, and mirrored into
    GOOGLE_API_KEY because that is the variable the provider reads.
    """

    key = (
        os.environ.get(API_KEY_ENV, "").strip()
        or os.environ.get(ALTERNATIVE_API_KEY_ENV, "").strip()
    )

    if key and not os.environ.get(API_KEY_ENV, "").strip():
        os.environ[API_KEY_ENV] = key

    return key


def load_config() -> LLMConfig | None:
    """
    Read the model configuration from the environment.

    Returns:
        The configuration, or None when no API key is set. None is the
        signal to stay local rather than an error.
    """

    if not api_key():
        return None

    return LLMConfig(
        provider=PROVIDER,
        model=os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL,
        timeout=_positive_float(TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS),
        max_retries=_non_negative_int(RETRIES_ENV, DEFAULT_MAX_RETRIES),
        thinking_budget=_optional_non_negative_int(THINKING_BUDGET_ENV),
    )


def build_chat_model(config: LLMConfig):
    """
    Build a chat model from a configuration.

    Args:
        config:
            Output of load_config.

    Returns:
        A LangChain chat model with the timeout and retry budget applied.
    """

    # Imported here so the module, and everything that only needs the
    # configuration, loads without the provider package being importable.
    from langchain.chat_models import init_chat_model

    options: dict[str, Any] = {}

    # Only sent when set: a model without a reasoning phase may reject it.
    if config.thinking_budget is not None:
        options["thinking_budget"] = config.thinking_budget

    return init_chat_model(
        config.model,
        model_provider=config.provider,
        temperature=config.temperature,
        timeout=config.timeout,
        max_retries=config.max_retries,
        **options,
    )


def extract_text_content(content: Any) -> str:
    """
    Reduce a chat model's content to plain, readable text.

    Gemini (and some other providers) return content as a list of blocks
    rather than a bare string, e.g. [{"type": "text", "text": "...",
    "extras": {"signature": "..."}}]. The signature is an internal
    verification token with no meaning to a reader, so only the text
    parts are kept.

    Args:
        content:
            The content field of a chat model response.

    Returns:
        The plain text of the response.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict):
                text = block.get("text")

                if isinstance(text, str):
                    parts.append(text)

        # A list with no text in it, such as the empty chunks a provider
        # streams around a tool call, is no text, not the string "[]".
        return "\n".join(parts)

    if content is None:
        return ""

    return str(content)


def token_usage(response: Any) -> tuple[int | None, int | None]:
    """
    Read input and output token counts from a response, when present.

    Args:
        response:
            A chat model message, or anything else.

    Returns:
        (input_tokens, output_tokens), each None when not reported.
    """

    usage = getattr(response, "usage_metadata", None)

    if not isinstance(usage, dict):
        return None, None

    def count(key: str) -> int | None:
        value = usage.get(key)

        return int(value) if isinstance(value, (int, float)) else None

    return count("input_tokens"), count("output_tokens")


def resolved_model(response: Any, requested: str) -> str:
    """
    Name the model that actually answered.

    Args:
        response:
            A chat model message.

        requested:
            The name that was asked for, used when the provider does not
            report one.

    Returns:
        The model name.
    """

    metadata = getattr(response, "response_metadata", None)

    if isinstance(metadata, dict):
        for key in ("model_name", "model"):
            value = metadata.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

    return requested


def complete(chat_model, prompt: str, requested_model: str) -> Completion:
    """
    Send one prompt and return the reply with what it cost.

    Args:
        chat_model:
            A LangChain chat model, or anything with an invoke method.

        prompt:
            The prompt.

        requested_model:
            The configured model name, for when the provider reports none.

    Returns:
        A Completion.
    """

    started = time.perf_counter()
    response = chat_model.invoke(prompt)
    latency_ms = (time.perf_counter() - started) * 1000

    input_tokens, output_tokens = token_usage(response)

    return Completion(
        text=extract_text_content(getattr(response, "content", response)),
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=resolved_model(response, requested_model),
    )


PLANNER_OUTPUT_ENV = "AI_ANALYST_PLANNER_OUTPUT"
PLANNER_OUTPUT_TEXT = "text"
PLANNER_OUTPUT_STRUCTURED = "structured"


def planner_output_mode() -> str:
    """
    How the planner asks for its reply: "text" (JSON requested in the
    prompt, the default) or "structured" (a schema enforced by the
    provider). Anything unrecognised is read as the default.
    """

    value = os.environ.get(PLANNER_OUTPUT_ENV, "").strip().lower()

    if value == PLANNER_OUTPUT_STRUCTURED:
        return PLANNER_OUTPUT_STRUCTURED

    return PLANNER_OUTPUT_TEXT


def complete_structured(
    chat_model,
    prompt: str,
    schema,
    to_text,
    requested_model: str
) -> Completion:
    """
    Send one prompt with an enforced reply schema.

    The parsed reply is written back as text with to_text, so it goes
    through the same reading and validation as a text reply. When the
    provider's reply does not parse against the schema, the raw text is
    returned instead, and the ordinary extractor and retry deal with it.

    Args:
        chat_model:
            A LangChain chat model supporting with_structured_output.

        prompt:
            The prompt.

        schema:
            A Pydantic model class.

        to_text:
            Turns a parsed instance into the text the caller reads.

        requested_model:
            The configured model name.

    Returns:
        A Completion, with usage read from the raw message.
    """

    runnable = chat_model.with_structured_output(schema, include_raw=True)

    started = time.perf_counter()
    response = runnable.invoke(prompt)
    latency_ms = (time.perf_counter() - started) * 1000

    raw = response.get("raw") if isinstance(response, dict) else None
    parsed = response.get("parsed") if isinstance(response, dict) else None

    if parsed is not None:
        text = to_text(parsed)

    else:
        text = extract_text_content(getattr(raw, "content", raw or ""))

    input_tokens, output_tokens = token_usage(raw)

    return Completion(
        text=text,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=resolved_model(raw, requested_model),
    )


EMBEDDING_MODEL_ENV = "AI_ANALYST_EMBEDDING_MODEL"


def build_embedder(config: LLMConfig | None):
    """
    Build an embedding function for glossary retrieval, when configured.

    Off unless AI_ANALYST_EMBEDDING_MODEL names a model: lexical retrieval
    needs nothing, and embedding a glossary sends its text to the
    provider, which is a choice the reader should make on purpose.

    Args:
        config:
            Output of load_config. None means no key, so no embedder.

    Returns:
        A function from a list of texts to a list of vectors, or None.
    """

    model = os.environ.get(EMBEDDING_MODEL_ENV, "").strip()

    if config is None or not model:
        return None

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(model=model)

    def embed(texts):
        return embeddings.embed_documents(list(texts))

    return embed
