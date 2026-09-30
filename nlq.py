"""
Questions in English, answered by arithmetic.

The design commitment here is that a language model never produces a
figure. It may only produce a plan, and the plan is a small typed
structure with a fixed vocabulary: an intent, a measure, an aggregation, a
timeline, a segment, some filters, a period window. Every plan is checked
against the actual dataset before anything runs, and the arithmetic is
done in pandas by this module.

That boundary is the whole point. A model asked "what was revenue last
quarter" will happily answer with a number, and that number is a plausible
guess drawn from the shape of the conversation. It cannot be checked, and
it is wrong often enough to be dangerous while being right often enough to
be trusted. Asking the model for a plan instead moves it to the job it is
actually good at, which is reading intent out of ambiguous English, and
leaves the counting to code.

Local rules try first, because most real questions are simple and a
round-trip to a model to compute a sum is both slow and pointless. The
model is a fallback for the questions the rules cannot parse, and any
answer that went that route is labelled, because a plan inferred by a
model is a weaker claim than one matched by a rule.

This module holds the plan and its validator. The parser, the executor and
the model fallback build on it.
"""

from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

import aggregation
import schema
import segments

# What a question is asking for. The intent decides the shape of the
# answer, and therefore which other fields the plan must carry.
INTENTS = (
    # One figure. "What was total revenue last quarter."
    "aggregate",
    # A figure per period. "Show revenue by month."
    "timeseries",
    # A figure per part, ranked. "Which region sells most."
    "breakdown",
    # Two period windows against each other. "Is this month worse."
    "compare",
    # The largest or smallest period. "Which month was worst."
    "extreme",
    # Direction over the whole span. "Is revenue growing."
    "trend",
    # One part as a proportion of the total. "How much is north."
    "share",
)

# Intents that cannot be answered without a timeline.
INTENTS_NEEDING_DATE = frozenset({
    "timeseries",
    "compare",
    "extreme",
    "trend",
})

# Intents that cannot be answered without something to split by.
INTENTS_NEEDING_SEGMENT = frozenset({"breakdown", "share"})

# How many values each filter operator expects. None means one or more.
OPERATOR_ARITY = {
    "equals": 1,
    "not_equals": 1,
    "in": None,
    "not_in": None,
    "contains": 1,
    "greater_than": 1,
    "greater_or_equal": 1,
    "less_than": 1,
    "less_or_equal": 1,
    "between": 2,
    "is_missing": 0,
    "is_present": 0,
}

FILTER_OPERATORS = tuple(OPERATOR_ARITY)

# Operators that only mean something against a number.
NUMERIC_OPERATORS = frozenset({
    "greater_than",
    "greater_or_equal",
    "less_than",
    "less_or_equal",
    "between",
})

# Operators that only mean something against text.
TEXT_OPERATORS = frozenset({"contains"})

# How a period window is expressed.
WINDOW_KINDS = (
    # Everything in the dataset.
    "all",
    # The most recent count periods of a grain.
    "last_periods",
    # An explicit date range, inclusive of both ends.
    "between_dates",
    # A single named period, given by the date it starts.
    "period",
)

DIRECTIONS = ("highest", "lowest")

# Where the plan came from. Rules are a stronger claim than a model guess,
# and the interface says which one ran.
PLAN_SOURCES = ("rules", "model")

# Ranking beyond this many parts stops being a ranking and becomes a
# table, which the breakdown view already provides.
MAX_TOP_N = 50

# A measure has to be mostly readable as numbers. Below this the column is
# text that happens to contain digits. Shared with schema, so a column the
# role detector calls a measure is one a question can total.
#
# Above the threshold, any value that will not parse is reported as a row
# count rather than a share. A share rounds to "0%" for one bad row in a
# thousand, which reads as nothing being wrong while a figure quietly
# excludes a row.
MEASURE_PARSE_RATE = schema.NUMERIC_COERCION_RATE

# Matches aggregation.usable_date_columns, so a column the timeline
# accepts is a column a question can ask about.
DATE_PARSE_RATE = 0.8


@dataclass(frozen=True)
class Filter:
    """
    One condition narrowing the rows a question is about.

    Attributes:
        column:
            Column the condition applies to.

        operator:
            One of FILTER_OPERATORS.

        values:
            Comparison values, as many as the operator expects.
    """

    column: str
    operator: str
    values: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Render as plain data, for storage or a model exchange."""

        return {
            "column": self.column,
            "operator": self.operator,
            "values": list(self.values),
        }

    def describe(self) -> str:
        """Write the condition as a phrase, for the calculation trail."""

        if self.operator == "is_missing":
            return f"'{self.column}' is blank"

        if self.operator == "is_present":
            return f"'{self.column}' is filled in"

        rendered = ", ".join(str(value) for value in self.values)

        phrases = {
            "equals": f"'{self.column}' is {rendered}",
            "not_equals": f"'{self.column}' is not {rendered}",
            "in": f"'{self.column}' is one of {rendered}",
            "not_in": f"'{self.column}' is none of {rendered}",
            "contains": f"'{self.column}' contains {rendered}",
            "greater_than": f"'{self.column}' is above {rendered}",
            "greater_or_equal": f"'{self.column}' is at least {rendered}",
            "less_than": f"'{self.column}' is below {rendered}",
            "less_or_equal": f"'{self.column}' is at most {rendered}",
        }

        if self.operator == "between" and len(self.values) == 2:
            return (
                f"'{self.column}' is between {self.values[0]} and "
                f"{self.values[1]}"
            )

        return phrases.get(
            self.operator,
            f"'{self.column}' {self.operator} {rendered}"
        )


@dataclass(frozen=True)
class Window:
    """
    The stretch of time a question is about.

    A relative window is deliberately left symbolic. "The last three
    months" is resolved against the most recent date in the dataset, not
    against today, and that resolution happens at execution time where the
    data is. A dataset that ends in 2023 asked for the last three months
    should report its final quarter, not an empty result because the
    calendar has moved on.

    Attributes:
        kind:
            One of WINDOW_KINDS.

        count:
            How many periods, for "last_periods".

        grain:
            Which periods, for "last_periods" and "period".

        start:
            ISO date. Required for "period"; optional for
            "between_dates", where omitting it means "from the beginning".

        end:
            ISO date, for "between_dates". Omitting it means "to the end".

        offset:
            How many periods further back to shift a relative window. Zero
            is the most recent. One is the period before that, which is
            what makes "this month against last month" expressible without
            a second vocabulary.
    """

    kind: str = "all"
    count: int | None = None
    grain: str | None = None
    start: str | None = None
    end: str | None = None
    offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Render as plain data."""

        return {
            "kind": self.kind,
            "count": self.count,
            "grain": self.grain,
            "start": self.start,
            "end": self.end,
            "offset": self.offset,
        }

    def describe(self) -> str:
        """Write the window as a phrase."""

        if self.kind == "all":
            return "the whole dataset"

        if self.kind == "last_periods":
            noun = self.grain or "period"
            plural = "" if self.count == 1 else "s"

            if not self.offset:
                return f"the last {self.count} {noun}{plural}"

            if self.count == 1:
                return (
                    f"the {noun} {self.offset} before the most recent"
                )

            return (
                f"the {self.count} {noun}{plural} ending "
                f"{self.offset} {noun}{plural} before the most recent"
            )

        if self.kind == "between_dates":
            if self.start and self.end:
                return f"{self.start} to {self.end}"

            if self.start:
                return f"{self.start} onwards"

            return f"up to {self.end}"

        if self.kind == "period":
            return f"the {self.grain or 'period'} starting {self.start}"

        return self.kind


EVERYTHING = Window()


@dataclass(frozen=True)
class QueryPlan:
    """
    Everything needed to answer one question, and nothing else.

    A plan is deliberately narrow. It cannot express arbitrary
    computation, which is what makes it safe to accept one from a language
    model: the worst a bad plan can do is ask a wrong but harmless
    question of the data.

    Attributes:
        intent:
            One of INTENTS.

        measure:
            Column being measured, or None to count rows.

        aggregation:
            One of aggregation.AGGREGATIONS.

        date:
            Column carrying the timeline.

        grain:
            Period grain, or None to detect one.

        segment:
            Column to split by.

        segment_value:
            The single part a "share" question is about.

        filters:
            Conditions narrowing the rows.

        window:
            The stretch of time in question.

        comparison:
            The second window, for "compare".

        top_n:
            How many parts to rank.

        direction:
            Whether "extreme" and "breakdown" want the top or the bottom.

        question:
            The English the plan came from, kept for display.

        source:
            "rules" or "model".
    """

    intent: str
    measure: str | None = None
    aggregation: str = "sum"
    date: str | None = None
    grain: str | None = None
    segment: str | None = None
    segment_value: str | None = None
    filters: tuple[Filter, ...] = ()
    window: Window = EVERYTHING
    comparison: Window | None = None
    top_n: int | None = None
    direction: str = "highest"
    question: str = ""
    source: str = "rules"

    def to_dict(self) -> dict[str, Any]:
        """Render as plain data, for storage or a model exchange."""

        return {
            "intent": self.intent,
            "measure": self.measure,
            "aggregation": self.aggregation,
            "date": self.date,
            "grain": self.grain,
            "segment": self.segment,
            "segment_value": self.segment_value,
            "filters": [entry.to_dict() for entry in self.filters],
            "window": self.window.to_dict(),
            "comparison": (
                self.comparison.to_dict()
                if self.comparison
                else None
            ),
            "top_n": self.top_n,
            "direction": self.direction,
            "question": self.question,
            "source": self.source,
        }

    @property
    def measure_label(self) -> str:
        """Name the measure the way a sentence would."""

        return self.measure or "Rows"

    def describe(self) -> str:
        """
        Write the plan as a sentence, so a reader can check the reading.

        The interface shows this above the answer. A wrong answer to the
        right question and a right answer to the wrong question look
        identical until the question is stated.
        """

        verb = {
            "aggregate": "Calculating",
            "timeseries": "Charting",
            "breakdown": "Ranking",
            "compare": "Comparing",
            "extreme": "Finding the "
                       f"{'highest' if self.direction == 'highest' else 'lowest'}"
                       " period of",
            "trend": "Testing the direction of",
            "share": "Measuring the share of",
        }.get(self.intent, "Reading")

        if self.measure:
            subject = f"{self.aggregation} of '{self.measure}'"

        else:
            subject = "the number of rows"

        parts = [f"{verb} {subject}"]

        if self.intent in {"breakdown", "share"} and self.segment:
            if self.intent == "share":
                parts.append(
                    f"for '{self.segment_value}' within '{self.segment}'"
                )

            else:
                parts.append(f"by '{self.segment}'")

        elif self.segment:
            parts.append(f"split by '{self.segment}'")

        if self.intent == "timeseries" and self.date:
            parts.append(
                f"per {self.grain or 'detected'} period on '{self.date}'"
            )

        if self.window.kind != "all":
            parts.append(f"over {self.window.describe()}")

        if self.comparison:
            parts.append(f"against {self.comparison.describe()}")

        if self.filters:
            parts.append(
                "where "
                + " and ".join(entry.describe() for entry in self.filters)
            )

        if self.top_n:
            parts.append(f"keeping the top {self.top_n}")

        return ", ".join(parts) + "."


def filter_from_dict(
    payload: Any,
    position: int
) -> tuple[Filter | None, list[str]]:
    """
    Read one filter from plain data, refusing anything unexpected.

    Args:
        payload:
            Candidate filter.

        position:
            Index in the list, so an error can point at the right one.

    Returns:
        The filter and any errors. A filter is only returned when there
        are no errors.
    """

    label = f"Filter {position + 1}"

    if not isinstance(payload, dict):
        return None, [f"{label} is not an object."]

    allowed = {"column", "operator", "values"}
    unexpected = sorted(set(payload) - allowed)

    if unexpected:
        return None, [
            f"{label} has unexpected field(s): {', '.join(unexpected)}."
        ]

    column = payload.get("column")
    operator = payload.get("operator")

    errors = []

    if not isinstance(column, str) or not column:
        errors.append(f"{label} has no column name.")

    if operator not in OPERATOR_ARITY:
        errors.append(
            f"{label} uses an unknown operator "
            f"'{operator}'. Supported: {', '.join(FILTER_OPERATORS)}."
        )

    raw_values = payload.get("values", [])

    if raw_values is None:
        raw_values = []

    if isinstance(raw_values, (str, bytes)) or not isinstance(
        raw_values,
        (list, tuple)
    ):
        raw_values = [raw_values]

    if errors:
        return None, errors

    expected = OPERATOR_ARITY[operator]
    values = tuple(raw_values)

    if expected == 0 and values:
        errors.append(
            f"{label} uses '{operator}', which takes no comparison "
            "value."
        )

    elif expected is None and not values:
        errors.append(
            f"{label} uses '{operator}', which needs at least one value."
        )

    elif isinstance(expected, int) and expected > 0 and (
        len(values) != expected
    ):
        errors.append(
            f"{label} uses '{operator}', which needs exactly "
            f"{expected} value(s), but {len(values)} were given."
        )

    if errors:
        return None, errors

    return Filter(column=column, operator=operator, values=values), []


def window_from_dict(
    payload: Any,
    label: str = "Window"
) -> tuple[Window | None, list[str]]:
    """
    Read a period window from plain data, refusing anything unexpected.

    Args:
        payload:
            Candidate window, or None for the whole dataset.

        label:
            Name used in error messages.

    Returns:
        The window and any errors.
    """

    if payload is None:
        return EVERYTHING, []

    if not isinstance(payload, dict):
        return None, [f"{label} is not an object."]

    allowed = {"kind", "count", "grain", "start", "end", "offset"}
    unexpected = sorted(set(payload) - allowed)

    if unexpected:
        return None, [
            f"{label} has unexpected field(s): {', '.join(unexpected)}."
        ]

    kind = payload.get("kind", "all")

    if kind not in WINDOW_KINDS:
        return None, [(
            f"{label} has an unknown kind '{kind}'. Supported: "
            f"{', '.join(WINDOW_KINDS)}."
        )]

    errors = []
    count = payload.get("count")

    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int):
            errors.append(f"{label} count must be a whole number.")

        elif count < 1:
            errors.append(f"{label} count must be at least 1.")

    grain = payload.get("grain")

    if grain is not None and grain not in aggregation.PERIOD_GRAINS:
        errors.append(
            f"{label} grain '{grain}' is not a period. Supported: "
            f"{', '.join(aggregation.PERIOD_GRAINS)}."
        )

    start = payload.get("start")
    end = payload.get("end")

    for name, value in (("start", start), ("end", end)):
        if value is not None and not isinstance(value, str):
            errors.append(f"{label} {name} must be a date written as text.")

    if kind == "last_periods":
        if count is None:
            errors.append(
                f"{label} asks for the last periods without saying how "
                "many."
            )

        if grain is None:
            errors.append(
                f"{label} asks for the last periods without saying which "
                "period."
            )

    # One end is enough: "since March" is a real question, and the missing
    # end is the end of the data.
    if kind == "between_dates" and not (start or end):
        errors.append(f"{label} needs a start date, an end date or both.")

    if kind == "period" and not start:
        errors.append(f"{label} needs the date the period starts.")

    offset = payload.get("offset", 0)

    if offset is None:
        offset = 0

    if isinstance(offset, bool) or not isinstance(offset, int):
        errors.append(f"{label} offset must be a whole number.")

    elif offset < 0:
        errors.append(f"{label} offset cannot be negative.")

    elif offset and kind != "last_periods":
        errors.append(
            f"{label} shifts by {offset} period(s), which only means "
            "something for a window counted back from the most recent."
        )

    if errors:
        return None, errors

    return Window(
        kind=kind,
        count=count,
        grain=grain,
        start=start,
        end=end,
        offset=offset
    ), []


def plan_from_dict(payload: Any) -> tuple[QueryPlan | None, list[str]]:
    """
    Read a plan from plain data, refusing anything unexpected.

    This is the boundary model output crosses, so it is strict on purpose:
    unknown fields are refused rather than ignored, and every value is
    checked against its vocabulary. Silently dropping a field a model
    thought it was setting would produce an answer to a question nobody
    asked.

    Args:
        payload:
            Candidate plan.

    Returns:
        The plan and any errors. A plan is only returned when there are no
        errors. Note that a plan returned here is well formed, not
        necessarily answerable: validate_plan checks it against the data.
    """

    if not isinstance(payload, dict):
        return None, ["The plan is not an object."]

    allowed = {
        "intent",
        "measure",
        "aggregation",
        "date",
        "grain",
        "segment",
        "segment_value",
        "filters",
        "window",
        "comparison",
        "top_n",
        "direction",
        "question",
        "source",
    }

    unexpected = sorted(set(payload) - allowed)

    if unexpected:
        return None, [(
            "The plan has unexpected field(s): "
            f"{', '.join(unexpected)}. Supported: "
            f"{', '.join(sorted(allowed))}."
        )]

    errors: list[str] = []

    intent = payload.get("intent")

    if intent not in INTENTS:
        errors.append(
            f"'{intent}' is not a supported intent. Supported: "
            f"{', '.join(INTENTS)}."
        )

    aggregation_name = payload.get("aggregation", "sum")

    if aggregation_name not in aggregation.AGGREGATIONS:
        errors.append(
            f"'{aggregation_name}' is not a supported aggregation. "
            f"Supported: {', '.join(aggregation.AGGREGATIONS)}."
        )

    grain = payload.get("grain")

    if grain is not None and grain not in aggregation.PERIOD_GRAINS:
        errors.append(
            f"'{grain}' is not a period. Supported: "
            f"{', '.join(aggregation.PERIOD_GRAINS)}."
        )

    direction = payload.get("direction", "highest")

    if direction not in DIRECTIONS:
        errors.append(
            f"'{direction}' is not a direction. Supported: "
            f"{', '.join(DIRECTIONS)}."
        )

    source = payload.get("source", "rules")

    if source not in PLAN_SOURCES:
        errors.append(
            f"'{source}' is not a plan source. Supported: "
            f"{', '.join(PLAN_SOURCES)}."
        )

    for name in ("measure", "date", "segment", "segment_value", "question"):
        value = payload.get(name)

        if value is not None and not isinstance(value, str):
            errors.append(f"'{name}' must be text or empty.")

    top_n = payload.get("top_n")

    if top_n is not None:
        if isinstance(top_n, bool) or not isinstance(top_n, int):
            errors.append("top_n must be a whole number.")

        elif top_n < 1:
            errors.append("top_n must be at least 1.")

        elif top_n > MAX_TOP_N:
            errors.append(f"top_n cannot be more than {MAX_TOP_N}.")

    raw_filters = payload.get("filters") or []

    if not isinstance(raw_filters, (list, tuple)):
        errors.append("filters must be a list.")
        raw_filters = []

    filters = []

    for position, entry in enumerate(raw_filters):
        parsed, filter_errors = filter_from_dict(entry, position)

        if filter_errors:
            errors.extend(filter_errors)

        else:
            filters.append(parsed)

    window, window_errors = window_from_dict(
        payload.get("window"),
        "The window"
    )
    errors.extend(window_errors)

    comparison = None

    if payload.get("comparison") is not None:
        comparison, comparison_errors = window_from_dict(
            payload.get("comparison"),
            "The comparison window"
        )
        errors.extend(comparison_errors)

    if errors:
        return None, errors

    return QueryPlan(
        intent=intent,
        measure=payload.get("measure"),
        aggregation=aggregation_name,
        date=payload.get("date"),
        grain=grain,
        segment=payload.get("segment"),
        segment_value=payload.get("segment_value"),
        filters=tuple(filters),
        window=window or EVERYTHING,
        comparison=comparison,
        top_n=top_n,
        direction=direction,
        question=payload.get("question") or "",
        source=source
    ), []


def _numeric_parse_rate(series: pd.Series) -> float:
    """
    Share of a column's filled-in values that read as a number.

    Delegates to schema so the parser, the validator and role detection
    cannot drift apart on what counts as measurable.
    """

    return schema.numeric_share(series)


def _date_parse_rate(series: pd.Series) -> float:
    """Share of a column that reads as a date."""

    if not len(series):
        return 0.0

    return float(aggregation.parse_dates(series).notna().mean())


@dataclass
class Validation:
    """
    The result of checking a plan against a dataset.

    Attributes:
        valid:
            Whether the plan can be executed.

        errors:
            Why not, each naming the column or field at fault.

        warnings:
            Things that will not stop execution but change how the answer
            should be read.

        plan:
            The plan, with anything the validator could fill in resolved.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    plan: QueryPlan | None = None


def validate_plan(
    plan: QueryPlan,
    frame: pd.DataFrame
) -> Validation:
    """
    Check a plan against the dataset it would run against.

    Well formed is not the same as answerable. A plan can name a real
    intent and a real aggregation and still ask for the total of a column
    of surnames, or a breakdown by a column holding one value per row.
    Every refusal names the column, so the interface can say what to fix
    rather than only that something is wrong.

    Args:
        plan:
            A plan from the parser or from plan_from_dict.

        frame:
            The dataset the plan would run against.

    Returns:
        A Validation carrying the refusals and the things that change how
        the answer should be read.
    """

    errors: list[str] = []
    warnings: list[str] = []

    columns = {str(name) for name in frame.columns}

    if plan.intent not in INTENTS:
        return Validation(
            valid=False,
            errors=[f"'{plan.intent}' is not a supported intent."]
        )

    if frame.empty:
        return Validation(
            valid=False,
            errors=[(
                "The dataset has no rows, so there is nothing to "
                "calculate."
            )]
        )

    counting_rows = plan.measure is None or plan.aggregation == "count"

    # Measure.
    if plan.measure is not None:
        if plan.measure not in columns:
            errors.append(
                f"There is no column called '{plan.measure}'."
            )

        elif not counting_rows:
            rate = _numeric_parse_rate(frame[plan.measure])

            if rate < MEASURE_PARSE_RATE:
                errors.append(
                    f"'{plan.measure}' cannot be measured: only "
                    f"{rate:.0%} of its filled-in values read as numbers."
                )

            else:
                unusable = int(
                    pd.to_numeric(
                        frame[plan.measure],
                        errors="coerce"
                    ).isna().sum()
                )

                if unusable:
                    warnings.append(
                        f"{unusable:,} row(s) have no usable "
                        f"'{plan.measure}' and are excluded from every "
                        "figure below."
                    )

    elif plan.aggregation != "count":
        # A plan with no measure counts rows whatever it claims to do. Left
        # unchecked, a plan saying "sum" would return a count and describe
        # itself as a total, which is the one failure this module exists to
        # prevent.
        errors.append(
            f"'{plan.aggregation}' needs a measure column. With no "
            "measure the only honest aggregation is 'count'."
        )

    # Date.
    if plan.date is not None:
        if plan.date not in columns:
            errors.append(f"There is no column called '{plan.date}'.")

        else:
            rate = _date_parse_rate(frame[plan.date])

            if rate < DATE_PARSE_RATE:
                errors.append(
                    f"'{plan.date}' cannot carry a timeline: only "
                    f"{rate:.0%} of its values read as dates."
                )

            else:
                unreadable = int(
                    aggregation.parse_dates(frame[plan.date]).isna().sum()
                )

                if unreadable:
                    warnings.append(
                        f"{unreadable:,} row(s) have no readable "
                        f"'{plan.date}' and are excluded."
                    )

    elif plan.intent in INTENTS_NEEDING_DATE:
        errors.append(
            f"A '{plan.intent}' question needs a date column, and none "
            "was identified."
        )

    if plan.window.kind != "all" and plan.date is None:
        errors.append(
            "The question asks about a stretch of time, but no date "
            "column was identified."
        )

    if plan.comparison is not None and plan.intent != "compare":
        warnings.append(
            "A comparison window was given for a question that does not "
            "compare periods, and it will be ignored."
        )

    if plan.intent == "compare" and plan.comparison is None:
        errors.append(
            "A comparison needs two periods, and only one was given."
        )

    # Segment.
    if plan.segment is not None and plan.intent not in INTENTS_NEEDING_SEGMENT:
        # Only a breakdown or a share splits by a segment. Any other intent
        # would compute its figure and quietly drop the split, which the
        # live evaluation caught: "which territory brought in the most"
        # was answered with a quarter. The message says what to use, so a
        # model's retry can correct it.
        errors.append(
            f"A '{plan.intent}' plan does not split by '{plan.segment}'. "
            "To rank or compare the parts of a column, use intent "
            "'breakdown' (with top_n 1 for the single highest or lowest "
            "part), or 'share' for one part's share of the total. To "
            "narrow to one part, use a filter instead."
        )

    elif plan.segment is not None:
        if plan.segment not in columns:
            errors.append(f"There is no column called '{plan.segment}'.")

        else:
            suitability = segments.describe_segment_suitability(
                frame,
                plan.segment
            )

            if not suitability["usable"]:
                errors.append(suitability["reason"])

    elif plan.intent in INTENTS_NEEDING_SEGMENT:
        errors.append(
            f"A '{plan.intent}' question needs something to split by, "
            "and no column was identified."
        )

    if plan.intent == "share":
        if not plan.segment_value:
            errors.append(
                "A share question needs to name which part it is about."
            )

        elif plan.segment in columns:
            labels = set(
                segments.segment_labels(frame[plan.segment]).astype(str)
            )

            if str(plan.segment_value) not in labels:
                errors.append(
                    f"'{plan.segment_value}' is not a value of "
                    f"'{plan.segment}'."
                )

        # A share of a total is a proportion. A share of an average is
        # nothing at all, and printing one would invite a false reading.
        if plan.aggregation not in segments.ADDITIVE_AGGREGATIONS:
            errors.append(
                f"A share cannot be taken of a '{plan.aggregation}', "
                "because the parts of an average do not add up to the "
                "whole. Ask for the "
                f"{plan.aggregation} of that part instead."
            )

    # Filters.
    for entry in plan.filters:
        if entry.column not in columns:
            errors.append(
                f"There is no column called '{entry.column}' to filter "
                "on."
            )

            continue

        if entry.operator in NUMERIC_OPERATORS:
            if _numeric_parse_rate(frame[entry.column]) < (
                MEASURE_PARSE_RATE
            ):
                errors.append(
                    f"'{entry.column}' is not numeric, so it cannot be "
                    f"compared with {entry.operator.replace('_', ' ')}."
                )

            for value in entry.values:
                try:
                    float(value)

                except (TypeError, ValueError):
                    errors.append(
                        f"'{value}' is not a number, so "
                        f"'{entry.column}' cannot be compared with it."
                    )

        if entry.operator in TEXT_OPERATORS and (
            pd.api.types.is_numeric_dtype(frame[entry.column])
        ):
            warnings.append(
                f"'{entry.column}' holds numbers, so a 'contains' test "
                "compares their written form."
            )

    if plan.top_n is not None and plan.intent not in {
        "breakdown",
        "extreme",
    }:
        warnings.append(
            "A limit was given for a question that returns one figure, "
            "and it will be ignored."
        )

    if errors:
        return Validation(valid=False, errors=errors, warnings=warnings)

    return Validation(
        valid=True,
        errors=[],
        warnings=warnings,
        plan=plan
    )


def resolve_plan_columns(
    plan: QueryPlan,
    frame: pd.DataFrame
) -> QueryPlan:
    """
    Fill in the columns a question left implicit.

    A question like "is it growing" names no measure and no date, but the
    dataset usually has an obvious candidate for each. Detection is
    reported to the reader through the plan description, so an
    assumption made here is visible rather than hidden.

    Args:
        plan:
            The plan so far.

        frame:
            The dataset.

    Returns:
        The plan with missing roles filled from detection where the intent
        needs them.
    """

    detected = schema.detect_business_schema(
        frame,
        measure=plan.measure,
        date=plan.date,
        segment=plan.segment
    )

    measure = plan.measure
    date = plan.date
    segment = plan.segment

    if measure is None and plan.aggregation != "count":
        measure = detected["measure"]

    if date is None and (
        plan.intent in INTENTS_NEEDING_DATE
        or plan.window.kind != "all"
        # A per-day rate divides by the calendar, so it needs a timeline
        # even when the intent alone would not.
        or plan.aggregation == "sum_per_day"
    ):
        date = detected["date"]

    if segment is None and plan.intent in INTENTS_NEEDING_SEGMENT:
        segment = detected["segment"]

    # Counting rows is the honest fallback when nothing can be measured,
    # rather than refusing a question the data can partly answer.
    aggregation_name = plan.aggregation

    if measure is None and aggregation_name != "count":
        aggregation_name = "count"

    return replace(
        plan,
        measure=measure,
        date=date,
        segment=segment,
        aggregation=aggregation_name
    )
