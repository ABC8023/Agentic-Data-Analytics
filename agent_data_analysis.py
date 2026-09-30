"""
The AI analyst: its tools, its system prompt, and the model clients for the
agent and the planner.

The analysis itself lives in dataset_store, cleaning, ml, profiling and charts;
this module wires the model-safe tools from them into an agent.
"""

from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent

import llm
import planner_schema
from charts import recommend_visualisations
from llm import extract_text_content
from ml import (
    analyse_target_for_ml,
    train_classification_model_tool,
    train_regression_model_tool,
)
from profiling import (
    get_column_schema,
    get_data_quality_report,
    get_dataset_summary,
    get_stat_overview,
    list_csv_file,
    preload_dataset,
)

load_dotenv()

SYSTEM_PROMPT = """
You are a data science assistant that analyses CSV datasets using
the available tools.

When a dataset id is supplied with the question, pass that id verbatim as
the 'name' argument of every tool call. Do not shorten it, and do not
replace it with the display file name. Use the display file name only when
describing the dataset to the user.

When the user asks about machine learning:

1. The user must explicitly provide or select a target column.
2. Use analyse_target_for_ml before training any model. When it returns
   is_trainable as false, do not call any training tool. Explain the
   blocking_reason it gives and suggest a better target column.
3. Explain whether the target appears suitable for classification
   or regression.
4. Only use train_classification_model_tool when:
   - classification is recommended, and
   - the user explicitly asks to train or compare models.
5. Explain the model leaderboard using accuracy, balanced accuracy
   and macro F1.
6. Explain why the best model was selected.
7. Mention class imbalance, removed columns and warnings when present.
   When test_size_adjusted is true, tell the user the testing size was
   changed and give the reason returned in test_size_adjustment_reason.
8. Do not claim that the best test-set model is guaranteed to perform
   best on unseen production data.

   When target analysis recommends regression:

1. Only use train_regression_model_tool when the user explicitly
   asks to train or compare regression models.
2. Compare models using MAE, RMSE and R².
3. Lower MAE and RMSE indicate smaller prediction errors.
4. Higher R² generally indicates better explanatory performance.
5. Mention when R² is negative.
6. Explain that model performance is based on the held-out test set
   and does not guarantee production performance.

Never invent target names, model metrics or dataset values.
Always obtain them from the available tools.

Never train a model without a target column.

You do not have access to individual rows or cell values, by design.
Use get_column_schema to understand the structure of a dataset: its
columns, their kinds and data types, missing and distinct counts, and
which columns look like identifiers. Never ask the user to paste rows,
and never claim to have read specific records.

Use get_stat_overview for statistical summaries.

Use recommend_visualisations for chart recommendations.

"""

# Every registered tool is wrapped with privacy.model_safe, which strips
# row level content on the way out. local_dataset_preview is deliberately
# absent from this list: it returns real cell values and is for local
# display only. tests/test_privacy.py enforces both facts.
tools=[list_csv_file, preload_dataset, get_dataset_summary,get_data_quality_report,get_column_schema,get_stat_overview,
       recommend_visualisations,analyse_target_for_ml,train_classification_model_tool,train_regression_model_tool]


# The agent is built on first use rather than at import time. Building it
# eagerly instantiates the Gemini client, which raises when no API key is
# configured and would stop the whole application from starting, including
# the tabs that need no language model at all.
_AGENT = None

# Planner calls made by plan_invoker are recorded here, newest last, so the
# interface and the evaluation harness can report what each question cost.
# Bounded, because a long session must not grow it without limit.
MAX_RECORDED_COMPLETIONS = 200
PLANNER_COMPLETIONS: list[llm.Completion] = []


def api_key_is_configured() -> bool:
    """
    Report whether a Gemini API key is available.

    Returns:
        True when GOOGLE_API_KEY or GEMINI_API_KEY is set.
    """

    return llm.load_config() is not None


def get_llm_provider():
    """
    Work out the LLM provider details from the configured API key.

    Kept for callers that want the model name without the rest of the
    configuration. See llm.load_config for the full picture.

    Returns:
        A tuple of (provider, model_name, api_key_env_var), or None when
        no Gemini key is configured.
    """

    config = llm.load_config()

    if config is None:
        return None

    return (config.provider, config.model, llm.API_KEY_ENV)


def get_agent():
    """
    Build the tool-calling agent, reusing it on later calls.

    Returns:
        The configured LangChain agent.

    Raises:
        RuntimeError: If no LLM API key is configured.
    """

    global _AGENT

    if _AGENT is not None:
        return _AGENT

    config = llm.load_config()
    if config is None:
        raise RuntimeError(
            "No Gemini API key is set. Add GOOGLE_API_KEY to a .env "
            "file in the project folder to enable the AI analyst. The "
            "other tabs work without it."
        )

    _AGENT = create_agent(
        model=llm.build_chat_model(config),
        tools=tools,
        system_prompt=SYSTEM_PROMPT
    )

    return _AGENT


# The planner is a separate, deliberately plainer client. It has no tools
# and no system prompt of its own: its only job is to turn a question into
# a query plan, which is then validated and executed locally. Giving it
# tools would let it read the data and start reporting figures, which is
# exactly the arrangement the query plan replaces.
_PLANNER = None


def get_planner():
    """
    Build the plain chat model used for planning, reusing it after that.

    Returns:
        The chat model.

    Raises:
        RuntimeError: If no LLM API key is configured.
    """

    global _PLANNER

    if _PLANNER is not None:
        return _PLANNER

    config = llm.load_config()
    if config is None:
        raise RuntimeError(
            "No Gemini API key is set, so questions the local parser "
            "cannot read have no fallback. Everything the parser does "
            "understand still works."
        )

    _PLANNER = llm.build_chat_model(config)

    return _PLANNER


def record_completion(completion: llm.Completion) -> None:
    """Keep a bounded record of planner calls."""

    PLANNER_COMPLETIONS.append(completion)

    overflow = len(PLANNER_COMPLETIONS) - MAX_RECORDED_COMPLETIONS

    if overflow > 0:
        del PLANNER_COMPLETIONS[:overflow]


def plan_invoker(sink: list | None = None):
    """
    Return a callable that sends a planning prompt and returns the reply.

    Each call is recorded in PLANNER_COMPLETIONS with its latency, token
    counts and the model that answered.

    Args:
        sink:
            Optional list that also receives each call's record, so a
            caller can attribute the calls to one question.

    Returns:
        A function taking a prompt and returning text, or None when no key
        is configured. None is the signal to stay entirely local rather
        than an error, because most questions never need this.
    """

    config = llm.load_config()

    if config is None:
        return None

    structured = llm.planner_output_mode() == llm.PLANNER_OUTPUT_STRUCTURED

    def invoke(prompt: str) -> str:
        if structured:
            completion = llm.complete_structured(
                get_planner(),
                prompt,
                planner_schema.PlannerReply,
                planner_schema.reply_to_text,
                config.model,
            )

        else:
            completion = llm.complete(get_planner(), prompt, config.model)

        record_completion(completion)

        if sink is not None:
            sink.append(completion)

        return completion.text

    return invoke


class AgentStream:
    """
    Run the agent while yielding its answer as it is written.

    The reader sees text within a second or two instead of waiting for the
    whole answer, which for an agent that calls several tools can take a
    while. The final state is kept, because the calculation trail is
    rebuilt from the full message history once the stream ends.

    Only the model's own prose is yielded. Tool-call fragments and tool
    results are part of the working, not the answer, and are shown in the
    trail instead.

    Usage:
        stream = AgentStream(messages)
        shown = st.write_stream(stream.text())
        state = stream.final_state
    """

    def __init__(self, messages: list[dict[str, Any]], agent=None):
        self.messages = messages
        self.agent = agent
        self.final_state: dict[str, Any] | None = None

    def text(self):
        """Yield the answer's text chunks, recording the final state."""

        agent = self.agent or get_agent()

        for mode, payload in agent.stream(
            {"messages": self.messages},
            stream_mode=["messages", "values"],
        ):
            if mode == "values":
                self.final_state = payload

                continue

            chunk, metadata = payload

            if type(chunk).__name__ not in {"AIMessageChunk", "AIMessage"}:
                continue

            if getattr(chunk, "tool_call_chunks", None) or getattr(
                chunk, "tool_calls", None
            ):
                continue

            if metadata.get("langgraph_node", "model") != "model":
                continue

            text = extract_text_content(chunk.content)

            if text:
                yield text

    def answer(self) -> str:
        """The final answer as stored, once the stream has finished."""

        messages = (self.final_state or {}).get("messages") or []

        if not messages:
            return ""

        return extract_text_content(messages[-1].content)
