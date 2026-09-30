"""
One structured event per question, and a running total per session.

The questions this answers, for whoever runs the application:

    What share of questions is answered locally, and what share needs the
    model? The design bets that most never do.

    When the model is used, how slow and how costly is it?

    When a question is not answered, why: the rules could not read it, the
    model judged it off topic or unanswerable, or the model could not be
    reached?

Each question produces one JSON log line on the "ai_analyst.questions"
logger, with a request id so a slow answer can be found again. By default
the event carries no question text, no column names and no figures, so the
log can be shipped anywhere without becoming a copy of the data. Set
AI_ANALYST_LOG_QUESTION_TEXT=1 to include the question when debugging
locally.

Set AI_ANALYST_EVENT_LOG to a file path to also append events there as
JSON Lines.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

LOGGER_NAME = "ai_analyst.questions"
EVENT_LOG_ENV = "AI_ANALYST_EVENT_LOG"
QUESTION_TEXT_ENV = "AI_ANALYST_LOG_QUESTION_TEXT"

logger = logging.getLogger(LOGGER_NAME)


def configure_console_logging() -> None:
    """
    Print events to stderr as bare JSON lines, once per process.

    Called by the application, not on import, so tests and the evaluation
    harness stay quiet. Streamlit reruns the script on every interaction,
    so this checks for its own handler before adding one.
    """

    if any(getattr(h, "_ai_analyst", False) for h in logger.handlers):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._ai_analyst = True

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def new_request_id() -> str:
    """A short identifier for one question."""

    return uuid.uuid4().hex[:12]


def outcome_of(result: dict[str, Any]) -> str:
    """
    Name why a question ended the way it did, from a small fixed set.

    The set is small on purpose: it is a field to count by, and an open
    vocabulary cannot be counted.
    """

    if result["answer"].success:
        return "answered"

    if result.get("model_unreachable"):
        return "model_unreachable"

    if result.get("not_on_topic"):
        return "off_topic"

    if result.get("model_unanswerable"):
        return "unanswerable"

    if result["planned_by"] == "nobody" and result.get("model_replies"):
        return "model_plan_rejected"

    return "not_understood"


def question_event(
    result: dict[str, Any],
    latency_ms: float,
    completions: list[Any],
    request_id: str,
    question: str = "",
    handed_to_agent: bool = False
) -> dict[str, Any]:
    """
    Describe one answered or refused question as a flat event.

    Args:
        result:
            Output of nlq_answer.answer_question.

        latency_ms:
            Wall time of answer_question, model calls included.

        completions:
            The llm.Completion records for the model calls this question
            made.

        request_id:
            From new_request_id.

        question:
            Included only when AI_ANALYST_LOG_QUESTION_TEXT is set.

        handed_to_agent:
            Whether the question then went on to the conversational
            agent.

    Returns:
        The event.
    """

    plan = result.get("plan")

    event = {
        "event": "question",
        "request_id": request_id,
        "route": result["planned_by"],
        "outcome": outcome_of(result),
        "handed_to_agent": handed_to_agent,
        "intent": plan.intent if plan else None,
        "latency_ms": round(latency_ms, 1),
        "model_calls": len(completions),
        "input_tokens": sum(c.input_tokens or 0 for c in completions),
        "output_tokens": sum(c.output_tokens or 0 for c in completions),
        "model": completions[-1].model if completions else None,
        "ignored_word_count": len(result.get("ignored") or []),
    }

    if os.environ.get(QUESTION_TEXT_ENV, "").strip() == "1" and question:
        event["question"] = question

    return event


def agent_event(
    request_id: str,
    latency_ms: float,
    messages: list[Any],
    tool_calls: int,
    failed: bool
) -> dict[str, Any]:
    """
    Describe one call to the conversational agent.

    Args:
        request_id:
            The id of the question that was handed over.

        latency_ms:
            Wall time of the agent call.

        messages:
            The agent's response messages, read for token usage.

        tool_calls:
            How many tools the agent called.

        failed:
            Whether the call raised.

    Returns:
        The event.
    """

    input_tokens, output_tokens, calls = usage_from_messages(messages)

    return {
        "event": "agent",
        "request_id": request_id,
        "latency_ms": round(latency_ms, 1),
        "model_calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool_calls": tool_calls,
        "failed": failed,
    }


def usage_from_messages(messages: list[Any]) -> tuple[int, int, int]:
    """
    Sum token usage over an agent's messages.

    Returns:
        (input_tokens, output_tokens, model_messages), counting only the
        messages that report usage, which are the model's own turns.
    """

    input_tokens = output_tokens = calls = 0

    for message in messages or []:
        usage = getattr(message, "usage_metadata", None)

        if not isinstance(usage, dict):
            continue

        calls += 1
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)

    return input_tokens, output_tokens, calls


def emit(event: dict[str, Any]) -> None:
    """
    Write an event as one JSON line.

    Logging failures never reach the reader: an unwritable log file must
    not cost anyone an answer.
    """

    line = json.dumps(event, sort_keys=True, default=str)

    logger.info(line)

    path = os.environ.get(EVENT_LOG_ENV, "").strip()

    if not path:
        return

    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    except OSError:
        logger.warning(json.dumps({"event": "event_log_unwritable"}))


@dataclass
class SessionUsage:
    """
    Running totals for one browser session, shown next to the chat.

    Attributes:
        questions:
            Questions asked.

        by_route:
            How many were planned by the rules, the model, or nobody.

        handed_to_agent:
            How many went on to the conversational agent.

        model_calls, input_tokens, output_tokens:
            What the planner and the agent cost between them.

        model_latency_ms:
            Wall time of each question that used a model, for a median.
    """

    questions: int = 0
    by_route: dict[str, int] = field(default_factory=dict)
    handed_to_agent: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_latency_ms: list[float] = field(default_factory=list)

    def add_question(self, event: dict[str, Any]) -> None:
        self.questions += 1
        route = event["route"]
        self.by_route[route] = self.by_route.get(route, 0) + 1
        self._add_cost(event)

    def add_agent(self, event: dict[str, Any]) -> None:
        self.handed_to_agent += 1
        self._add_cost(event)

    def _add_cost(self, event: dict[str, Any]) -> None:
        self.model_calls += event["model_calls"]
        self.input_tokens += event["input_tokens"]
        self.output_tokens += event["output_tokens"]

        if event["model_calls"]:
            self.model_latency_ms.append(event["latency_ms"])

    @property
    def answered_locally_share(self) -> float | None:
        """Share of questions the rules answered with no model call."""

        if not self.questions:
            return None

        return self.by_route.get("rules", 0) / self.questions

    @property
    def median_model_latency_ms(self) -> float | None:
        if not self.model_latency_ms:
            return None

        ordered = sorted(self.model_latency_ms)
        middle = len(ordered) // 2

        if len(ordered) % 2:
            return ordered[middle]

        return (ordered[middle - 1] + ordered[middle]) / 2
