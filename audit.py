"""
The calculation behind an assistant answer.

A chat answer is prose. On its own it gives a reader no way to tell a
computed figure from an invented one. This module reconstructs what the
agent actually did: which tools it called, with what arguments, and the
figures that came back. The interface renders that under every answer.

It also makes the absence of work visible. When the agent answers with no
tool call at all, the trail says so, which is the signal that a number in
the prose was not calculated during this turn.

Deliberately dependency-free apart from the standard library, so it can be
tested without Streamlit, LangChain or a model.
"""

import ast
from typing import Any

# Keys that carry plumbing rather than findings.
NOISE_KEYS = frozenset({
    "success",
    "privacy_note",
    "file_name",
    "note",
})

# Shown first when present, because they answer "what did this tell us".
LEADING_KEYS = (
    "error",
    "rows",
    "columns",
    "recommendation_count",
    "recommended_task",
    "is_trainable",
    "best_model_name",
)

MAX_HIGHLIGHTS = 10
MAX_TEXT_LENGTH = 120
MAX_LIST_NAMES = 4

SESSION_HANDLE_SEPARATOR = "/"


def display_dataset_name(value: Any) -> Any:
    """
    Reduce a session-scoped dataset handle to its file name.

    Handles look like "<session token>/<file name>". The token is noise to
    a reader, so only the file name is shown.

    The split is done here rather than imported, to keep this module free
    of the heavy analysis imports.

    Args:
        value:
            Argument value, which may or may not be a handle.

    Returns:
        The file name when the value looks like a handle, otherwise the
        value unchanged.
    """

    if not isinstance(value, str):
        return value

    if SESSION_HANDLE_SEPARATOR not in value:
        return value

    token, _, remainder = value.partition(SESSION_HANDLE_SEPARATOR)

    # Session tokens are 32 character hex strings. Anything else is a real
    # path and should be left alone.
    if len(token) == 32 and all(
        character in "0123456789abcdef"
        for character in token.lower()
    ):
        return remainder

    return value


def sanitise_handles(payload: Any) -> Any:
    """
    Replace session-scoped dataset handles with plain file names.

    Tool results echo the handle back under keys such as file_name. The
    session token in it is noise to a reader and has no business being
    retained in the conversation history, so it is stripped wherever it
    appears rather than only in the displayed arguments.

    Args:
        payload:
            Any parsed tool result or argument value.

    Returns:
        The same structure with handle strings shortened.
    """

    if isinstance(payload, str):
        return display_dataset_name(payload)

    if isinstance(payload, dict):
        return {
            key: sanitise_handles(value)
            for key, value in payload.items()
        }

    if isinstance(payload, list):
        return [sanitise_handles(item) for item in payload]

    if isinstance(payload, tuple):
        return tuple(sanitise_handles(item) for item in payload)

    return payload


def format_value(value: Any) -> str:
    """
    Render one result value compactly for display.

    Args:
        value:
            Value taken from a tool result.

    Returns:
        A short human readable string.
    """

    if value is None:
        return "none"

    if isinstance(value, bool):
        return "yes" if value else "no"

    if isinstance(value, int):
        return f"{value:,}"

    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return f"{int(value):,}"

        return f"{round(value, 4):,}".rstrip("0").rstrip(".")

    if isinstance(value, str):
        text = value.strip()

        if len(text) > MAX_TEXT_LENGTH:
            return text[:MAX_TEXT_LENGTH].rstrip() + "..."

        return text

    if isinstance(value, (list, tuple)):
        if not value:
            return "none"

        if all(isinstance(item, str) for item in value):
            shown = ", ".join(value[:MAX_LIST_NAMES])

            if len(value) > MAX_LIST_NAMES:
                return f"{len(value)} ({shown}, ...)"

            return f"{len(value)} ({shown})"

        return f"{len(value)} entries"

    if isinstance(value, dict):
        return f"{len(value)} entries"

    return str(value)


def parse_result(content: Any) -> Any:
    """
    Recover a tool's return value from a tool message.

    Tool results arrive as the Python representation of whatever the tool
    returned. literal_eval reads that back safely: it accepts only
    literals and never executes code.

    Args:
        content:
            The tool message content.

    Returns:
        The parsed value, or the original content when it cannot be
        parsed.
    """

    if isinstance(content, (dict, list)):
        return content

    if not isinstance(content, str):
        return content

    text = content.strip()

    if not text or text[0] not in "{[(":
        return content

    try:
        return ast.literal_eval(text)

    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return content


def summarise_result(payload: Any) -> list[str]:
    """
    Pick the figures worth showing from a tool result.

    Args:
        payload:
            Parsed tool result.

    Returns:
        Up to MAX_HIGHLIGHTS lines of "key: value".
    """

    if not isinstance(payload, dict):
        rendered = format_value(payload)

        return [rendered] if rendered else []

    ordered_keys: list[str] = []

    for key in LEADING_KEYS:
        if key in payload:
            ordered_keys.append(key)

    for key in payload:
        if key in NOISE_KEYS or key in ordered_keys:
            continue

        ordered_keys.append(key)

    highlights = []

    for key in ordered_keys:
        value = payload[key]

        # An empty list adds nothing; it usually means "none removed".
        if isinstance(value, (list, tuple, dict)) and not value:
            continue

        highlights.append(f"{key}: {format_value(value)}")

        if len(highlights) >= MAX_HIGHLIGHTS:
            break

    return highlights


def _tool_calls_of(message: Any) -> list[dict]:
    """Return the tool calls on a message, tolerating either shape."""

    calls = getattr(message, "tool_calls", None)

    if calls is None and isinstance(message, dict):
        calls = message.get("tool_calls")

    if not calls:
        return []

    normalised = []

    for call in calls:
        if isinstance(call, dict):
            normalised.append(call)

        else:
            normalised.append({
                "name": getattr(call, "name", ""),
                "args": getattr(call, "args", {}),
                "id": getattr(call, "id", ""),
            })

    return normalised


def _attribute(message: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from a message object or a plain dictionary."""

    if isinstance(message, dict):
        return message.get(name, default)

    return getattr(message, name, default)


def extract_calculation_trail(agent_response: Any) -> list[dict]:
    """
    Reconstruct the tool calls behind an agent response.

    Args:
        agent_response:
            The value returned by the agent, or the message list itself.

    Returns:
        One entry per tool call, in the order the agent made them, each
        holding the tool name, the arguments, the parsed result, the
        selected highlights and whether the call reported failure.
    """

    messages = agent_response

    if isinstance(agent_response, dict):
        messages = agent_response.get("messages", [])

    if messages is None:
        return []

    # Results are matched to their call by identifier, because a model may
    # request several tools before any of them return.
    results_by_id: dict[str, Any] = {}
    results_without_id: list[tuple[str, Any]] = []

    for message in messages:
        if type(message).__name__ != "ToolMessage" and not (
            isinstance(message, dict)
            and message.get("role") == "tool"
        ):
            continue

        call_id = _attribute(message, "tool_call_id", "")
        content = _attribute(message, "content", "")

        if call_id:
            results_by_id[call_id] = content

        else:
            results_without_id.append(
                (_attribute(message, "name", ""), content)
            )

    trail: list[dict] = []

    for message in messages:
        for call in _tool_calls_of(message):
            call_id = call.get("id", "")
            name = call.get("name", "") or "unknown tool"

            if call_id in results_by_id:
                raw_result = results_by_id[call_id]

            else:
                raw_result = None

                for index, (result_name, content) in enumerate(
                    results_without_id
                ):
                    if result_name == name:
                        raw_result = content
                        results_without_id.pop(index)
                        break

            parsed = (
                sanitise_handles(parse_result(raw_result))
                if raw_result is not None
                else None
            )

            arguments = sanitise_handles(call.get("args") or {})

            failed = (
                isinstance(parsed, dict)
                and parsed.get("success") is False
            )

            trail.append({
                "step": len(trail) + 1,
                "tool": name,
                "arguments": arguments,
                "result": parsed,
                "highlights": (
                    summarise_result(parsed)
                    if parsed is not None
                    else []
                ),
                "failed": bool(failed),
                "error": (
                    str(parsed.get("error", ""))
                    if failed
                    else ""
                ),
            })

    return trail


def describe_trail(trail: list[dict]) -> str:
    """
    Summarise a trail in one line, for an expander label.

    Args:
        trail:
            Output of extract_calculation_trail.

    Returns:
        A short description of how much work the answer rests on.
    """

    if not trail:
        return "No calculation: the model answered without running anything"

    failures = sum(1 for entry in trail if entry["failed"])

    label = (
        "1 calculation"
        if len(trail) == 1
        else f"{len(trail)} calculations"
    )

    if failures:
        return f"Show the calculation ({label}, {failures} failed)"

    return f"Show the calculation ({label})"


def trail_is_empty(trail: list[dict]) -> bool:
    """
    Report whether an answer rests on no computation at all.

    Args:
        trail:
            Output of extract_calculation_trail.

    Returns:
        True when no tool was called.
    """

    return not trail
