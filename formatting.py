"""
How numbers and periods are written down.

Every figure the interface shows passes through here, so a total is
formatted the same way in a headline, a chart label and an exported
brief. Keeping it in one module is what stops "1,234.0" appearing next to
"1.2k" on the same screen.

Standard library and pandas only, so it can be tested directly.
"""

from typing import Any

import pandas as pd

# Thresholds for compact notation. Below the first, plain notation reads
# better; a reader does not want "847" written as "0.8k".
COMPACT_STEPS = (
    (1_000_000_000_000, "T"),
    (1_000_000_000, "B"),
    (1_000_000, "M"),
    (1_000, "k"),
)

# How a period start is written, per grain.
PERIOD_FORMATS = {
    "day": "%d %b %Y",
    "week": "week of %d %b %Y",
    "month": "%b %Y",
    "quarter": None,
    "year": "%Y",
}

MISSING_TEXT = "not available"


def is_missing(value: Any) -> bool:
    """
    Report whether a value should be shown as unavailable.

    Args:
        value:
            Value to test.

    Returns:
        True for None and for any pandas or numpy null.
    """

    if value is None:
        return True

    try:
        return bool(pd.isna(value))

    except (TypeError, ValueError):
        return False


def format_number(value: Any, decimals: int | None = None) -> str:
    """
    Write a number in full, with thousands separators.

    Args:
        value:
            Number to format.

        decimals:
            Fixed number of decimal places. When omitted, the scale of the
            value decides: large numbers are shown whole, small ones keep
            enough precision to stay meaningful.

    Returns:
        The formatted number, or a placeholder when it is missing.
    """

    if is_missing(value):
        return MISSING_TEXT

    try:
        number = float(value)

    except (TypeError, ValueError):
        return str(value)

    if decimals is None:
        if number == int(number):
            decimals = 0

        else:
            magnitude = abs(number)

            if magnitude >= 100:
                decimals = 0

            elif magnitude >= 10:
                decimals = 1

            elif magnitude >= 1:
                decimals = 2

            else:
                decimals = 4

    return f"{number:,.{decimals}f}"


def format_compact(value: Any, decimals: int = 1) -> str:
    """
    Write a number briefly, for axis labels and tight cards.

    Args:
        value:
            Number to format.

        decimals:
            Decimal places to keep on the scaled value.

    Returns:
        A short form such as "1.2M", or the plain number below a thousand.
    """

    if is_missing(value):
        return MISSING_TEXT

    try:
        number = float(value)

    except (TypeError, ValueError):
        return str(value)

    sign = "-" if number < 0 else ""
    magnitude = abs(number)

    for step, suffix in COMPACT_STEPS:
        if magnitude >= step:
            scaled = magnitude / step

            rendered = f"{scaled:,.{decimals}f}".rstrip("0").rstrip(".")

            return f"{sign}{rendered}{suffix}"

    return format_number(number)


def format_percent(
    value: Any,
    decimals: int = 1,
    already_scaled: bool = False
) -> str:
    """
    Write a proportion as a percentage.

    Args:
        value:
            The proportion, for example 0.184.

        decimals:
            Decimal places to keep.

        already_scaled:
            Set when the value is already expressed out of 100.

    Returns:
        A string such as "18.4%".
    """

    if is_missing(value):
        return MISSING_TEXT

    try:
        number = float(value)

    except (TypeError, ValueError):
        return str(value)

    if not already_scaled:
        number *= 100

    return f"{number:,.{decimals}f}%"


def format_signed(value: Any, decimals: int | None = None) -> str:
    """
    Write a change with an explicit sign, so direction is unmistakable.

    Args:
        value:
            The change.

        decimals:
            Passed through to format_number.

    Returns:
        A string such as "+1,204" or "-38.5".
    """

    if is_missing(value):
        return MISSING_TEXT

    try:
        number = float(value)

    except (TypeError, ValueError):
        return str(value)

    rendered = format_number(abs(number), decimals)

    if number > 0:
        return f"+{rendered}"

    if number < 0:
        return f"-{rendered}"

    return rendered


def format_signed_percent(value: Any, decimals: int = 1) -> str:
    """
    Write a proportional change with an explicit sign.

    Args:
        value:
            The change as a proportion, for example -0.072.

        decimals:
            Decimal places to keep.

    Returns:
        A string such as "-7.2%".
    """

    if is_missing(value):
        return MISSING_TEXT

    try:
        number = float(value) * 100

    except (TypeError, ValueError):
        return str(value)

    sign = "+" if number > 0 else ("-" if number < 0 else "")

    return f"{sign}{abs(number):,.{decimals}f}%"


def format_period(value: Any, grain: str = "month") -> str:
    """
    Write the start of a period the way a reader expects to see it.

    Args:
        value:
            Timestamp marking the start of the period.

        grain:
            One of day, week, month, quarter or year.

    Returns:
        A label such as "Aug 2026" or "Q3 2026".
    """

    if is_missing(value):
        return MISSING_TEXT

    timestamp = pd.Timestamp(value)

    if grain == "quarter":
        return f"Q{timestamp.quarter} {timestamp.year}"

    pattern = PERIOD_FORMATS.get(grain, PERIOD_FORMATS["month"])

    if pattern is None:
        pattern = PERIOD_FORMATS["month"]

    return timestamp.strftime(pattern)


def format_period_range(start: Any, end: Any, grain: str = "month") -> str:
    """
    Write the span a series covers.

    Args:
        start:
            First period start.

        end:
            Last period start.

        grain:
            Period grain, used for both labels.

    Returns:
        A string such as "Jan 2024 to Jul 2026", or a single label when
        the range holds only one period.
    """

    first = format_period(start, grain)
    last = format_period(end, grain)

    if first == last:
        return first

    return f"{first} to {last}"


def pluralise(count: int, singular: str, plural: str | None = None) -> str:
    """
    Write a count with a correctly pluralised noun.

    Args:
        count:
            How many.

        singular:
            Noun for one.

        plural:
            Noun for other counts. Defaults to the singular plus "s".

    Returns:
        A string such as "1 period" or "7 periods".
    """

    word = singular if abs(count) == 1 else (plural or f"{singular}s")

    return f"{count:,} {word}"


def describe_grain(grain: str) -> str:
    """
    Name a grain as an adjective, for prose.

    Args:
        grain:
            One of day, week, month, quarter or year.

    Returns:
        A word such as "monthly".
    """

    return {
        "day": "daily",
        "week": "weekly",
        "month": "monthly",
        "quarter": "quarterly",
        "year": "yearly",
    }.get(grain, grain)
