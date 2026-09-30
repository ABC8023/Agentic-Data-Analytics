"""
The boundary between local computation and the language model.

Every analysis in this application runs locally with pandas. The model is
only ever given the column schema and computed summaries, so uploaded
rows do not become part of a prompt.

This module holds that rule in one place, provides the scrubber that
enforces it on the way out of a tool, and is covered by a contract test
so the guarantee cannot quietly regress when a new tool is added.

What is shared with the model when a key is configured:
    column names, data types, row and column counts
    null counts, distinct counts, duplicate and constant column counts
    numeric summaries such as mean, median, quartiles and range
    category labels together with their counts and shares
    model metrics, class distributions and feature importance

What is never shared:
    individual rows or records
    row previews, samples or heads of a table
    the values of identifier-like columns, such as names, emails or
    reference codes, which are reported only as a distinct count

Two limitations, stated plainly rather than glossed over.

Category labels are aggregate level rather than row level, so an
ordinary label such as a region or a product category does reach the
model inside a frequency summary. A question like "which region is
largest" cannot be answered otherwise while the model composes the
answer. Columns that look like identifiers are excluded from this, so
a list of customer names is never enumerated.

Numeric summaries include minimums, maximums and quartiles. In a column
with very few distinct values, such a summary can coincide exactly with
an individual cell value. There is no way to summarise a numeric column
without revealing something about its contents, so this is a property of
aggregate reporting rather than a defect.

Both limitations apply to the AI analyst, which composes prose and needs
figures to do it. They do not apply to a question answered from a query
plan: nlq_model sends the column names, kinds and the labels of columns
worth filtering on, and no computed figure at all. Most questions take
that route, so the compromise above is now the exception rather than the
rule. It has not been removed, because a conversational answer cannot be
written without something to write about.
"""

import functools
import numbers
from collections.abc import Iterable
from typing import Any

# Row level content is never sent. This is not configurable.
SHARE_ROW_VALUES = False

# Ordinary category labels are sent, with the limitation described above.
SHARE_CATEGORY_LABELS = True

# Values of identifier-like columns are never enumerated, only counted.
SHARE_IDENTIFIER_LABELS = False

# Result keys that carry row level or cell level content. Anything named
# like this is stripped from a payload before it can reach the model.
ROW_LEVEL_KEYS = frozenset({
    "records",
    "rows",
    "row",
    "head",
    "preview",
    "sample",
    "samples",
    "sample_values",
    "sample_rows",
    "cell_values",
    "values_preview",
    "raw",
    "raw_rows",
    "data",
})

REDACTION_NOTE = (
    "Row level content was withheld. This assistant only receives the "
    "column schema and computed summaries."
)


def is_scalar(value: Any) -> bool:
    """
    Report whether a value is a single scalar rather than a collection.

    The forbidden key names are ambiguous on their own. "rows" may be a
    count, which is a harmless aggregate, or a list of records, which is
    not. Shape decides: a scalar under a forbidden name is kept, a
    collection is removed.

    Args:
        value:
            Value to inspect.

    Returns:
        True for None, strings, bytes, booleans and any number, including
        numpy scalar types.
    """

    return (
        value is None
        or isinstance(value, (str, bytes, bool, numbers.Number))
    )


def describe_policy() -> str:
    """
    Return a short statement of the data handling rule for the interface.

    Returns:
        A sentence suitable for display next to the chat input.
    """

    return (
        "Analysis runs locally with pandas. Questions about figures, "
        "periods, segments and shares are answered here, and a model "
        "sees only the column names and labels needed to interpret the "
        "wording, never a figure. Open-ended questions go to the AI "
        "analyst, which additionally receives computed summaries such as "
        "category counts and numeric ranges. Individual rows are never "
        "sent either way, and the values of identifier-like columns such "
        "as names or reference codes are never listed."
    )


def scrub_model_payload(
    payload: Any
) -> tuple[Any, list[str]]:
    """
    Remove row level content from a value on its way to the model.

    Nested dictionaries and lists are walked, so a forbidden key is
    caught wherever a tool happens to place it.

    Args:
        payload:
            Value a tool is about to return.

    Returns:
        A tuple of the cleaned value and the sorted names of the keys
        that were removed.
    """

    removed: set = set()

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = {}

            for key, item in value.items():
                if (
                    isinstance(key, str)
                    and key.lower() in ROW_LEVEL_KEYS
                    and not is_scalar(item)
                ):
                    removed.add(key)
                    continue

                cleaned[key] = walk(item)

            if removed and "privacy_note" not in cleaned:
                cleaned["privacy_note"] = REDACTION_NOTE

            return cleaned

        if isinstance(value, list):
            return [walk(item) for item in value]

        if isinstance(value, tuple):
            return tuple(walk(item) for item in value)

        return value

    cleaned_payload = walk(payload)

    return cleaned_payload, sorted(removed)


def model_safe(function):
    """
    Decorate a tool so its result cannot carry row level content.

    Applied beneath the LangChain tool decorator, this is the single
    enforcement point for the rule. A new tool added later inherits the
    guarantee by being decorated, and the contract test fails loudly if
    it is not.

    Args:
        function:
            The tool implementation to wrap.

    Returns:
        The wrapped function, with its name, docstring and signature
        preserved so tool schema inference still works.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)

        cleaned, _ = scrub_model_payload(result)

        return cleaned

    wrapper.__model_safe__ = True

    return wrapper


def find_row_level_keys(payload: Any) -> list[str]:
    """
    Report any row level keys present in a payload.

    Used by the contract test to assert that no registered tool leaks
    row content.

    Args:
        payload:
            Value to inspect.

    Returns:
        Sorted names of offending keys. Empty when the payload is clean.
    """

    found: set = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if (
                    isinstance(key, str)
                    and key.lower() in ROW_LEVEL_KEYS
                    and not is_scalar(item)
                ):
                    found.add(key)

                walk(item)

        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(payload)

    return sorted(found)


def contains_any_value(
    payload: Any,
    forbidden_values: Iterable[Any]
) -> list[str]:
    """
    Report which of a set of values appear anywhere in a payload.

    Lets a test assert that specific cell values from a fixture never
    show up in what a tool returns.

    Args:
        payload:
            Value to inspect.

        forbidden_values:
            Values that must not appear.

    Returns:
        The forbidden values that were found, as strings.
    """

    targets = {str(value) for value in forbidden_values}
    seen: set = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key) in targets:
                    seen.add(str(key))

                walk(item)

        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

        else:
            if str(value) in targets:
                seen.add(str(value))

    walk(payload)

    return sorted(seen)


def policy_summary() -> dict[str, Any]:
    """
    Return the policy as data, for display or for a test to assert on.

    Returns:
        The current switch values and the forbidden key list.
    """

    return {
        "share_row_values": SHARE_ROW_VALUES,
        "share_category_labels": SHARE_CATEGORY_LABELS,
        "row_level_keys": sorted(ROW_LEVEL_KEYS),
        "statement": describe_policy(),
    }
