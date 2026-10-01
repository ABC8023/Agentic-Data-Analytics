"""
Running a plan, and showing the work.

Every figure a reader sees comes from this module, and every figure comes
with the steps that produced it. The trail is not decoration: prose alone
gives nobody a way to tell a computed number from a plausible one, and the
whole point of planning a question rather than asking a model to answer it
is that the arithmetic can be inspected.

Two decisions here are worth stating up front because they change answers.

A relative window is resolved against the data, not the calendar. "The
last three months" of a dataset that ends in June 2023 means April to June
2023. Resolving against today would return nothing and look like a
business collapse rather than a stale extract.

An unfinished final period is excluded from a relative window, and the
exclusion is reported. A dataset ending on the 4th of the month has four
days in its newest month, and letting "last month" mean those four days
turns a partial extract into a catastrophic decline.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import aggregation
import formatting
import nlq
import nlq_model
import nlq_rules
import segments
import timeseries

# How to step back one period of each grain.
GRAIN_STEPS = {
    "day": lambda count: pd.Timedelta(days=count),
    "week": lambda count: pd.Timedelta(weeks=count),
    "month": lambda count: pd.DateOffset(months=count),
    "quarter": lambda count: pd.DateOffset(months=3 * count),
    "year": lambda count: pd.DateOffset(years=count),
}

# Rows shown in an answer table before it stops being an answer and starts
# being an export.
MAX_ANSWER_ROWS = 50


@dataclass
class Answer:
    """
    What running a plan produced.

    Attributes:
        success:
            Whether the plan could be run.

        error:
            Why not.

        plan:
            The plan that ran.

        headline:
            The sentence answering the question.

        figures:
            The numbers behind the sentence, named.

        table:
            Supporting rows, when the answer has any.

        trail:
            The steps taken, in order, each with the arithmetic.

        notes:
            Things that change how the answer should be read.
    """

    success: bool
    plan: nlq.QueryPlan | None = None
    error: str = ""
    headline: str = ""
    figures: dict[str, Any] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    trail: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Trail:
    """
    The steps behind an answer, collected as they happen.

    Recorded during execution rather than reconstructed afterwards,
    because a reconstruction is a description of what the code was meant
    to do rather than of what it did.
    """

    def __init__(self):
        self.steps: list[dict[str, Any]] = []

    def add(self, action: str, detail: str, result: str = "") -> None:
        """
        Record one step.

        Args:
            action:
                What was done, in a few words.

            detail:
                The arithmetic or the rule applied.

            result:
                What came back.
        """

        self.steps.append({
            "step": len(self.steps) + 1,
            "action": action,
            "detail": detail,
            "result": result,
        })


def _shift_period(start: Any, grain: str, periods_back: int) -> pd.Timestamp:
    """
    Step a period start back by a whole number of periods.

    Args:
        start:
            The period start to move.

        grain:
            Which period.

        periods_back:
            How many periods to go back. Zero returns the input.

    Returns:
        The start of the earlier period, realigned to the period boundary
        so month arithmetic cannot drift.
    """

    moment = pd.Timestamp(start)

    if not periods_back:
        return moment

    step = GRAIN_STEPS.get(grain, GRAIN_STEPS["month"])
    moved = moment - step(periods_back)

    return pd.Timestamp(
        aggregation.period_starts(pd.Series([moved]), grain).iloc[0]
    )


def resolve_window(
    dates: pd.Series,
    window: nlq.Window,
    fallback_grain: str = "month"
) -> dict[str, Any]:
    """
    Turn a symbolic window into a pair of dates.

    Args:
        dates:
            The parsed timeline, used to anchor a relative window.

        window:
            The window from the plan.

        fallback_grain:
            Grain to use when the window does not name one.

    Returns:
        The inclusive start and end, a phrase describing the range, and any
        note about a period that was excluded.
    """

    usable = dates.dropna()

    if usable.empty:
        return {
            "success": False,
            "error": "No usable dates were found, so no period can be "
                     "selected.",
        }

    if window.kind == "all":
        return {
            "success": True,
            "start": None,
            "end": None,
            "description": "the whole dataset",
            "notes": [],
        }

    if window.kind == "between_dates":
        start = pd.Timestamp(window.start) if window.start else None
        end = pd.Timestamp(window.end) if window.end else None

        if end is not None:
            # An ISO date names a day, and a range ending on it includes
            # that whole day rather than only its first instant.
            end = end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

        return {
            "success": True,
            "start": start,
            "end": end,
            "description": window.describe(),
            "notes": [],
        }

    grain = window.grain or fallback_grain

    if window.kind == "period":
        start = pd.Timestamp(
            aggregation.period_starts(
                pd.Series([pd.Timestamp(window.start)]),
                grain
            ).iloc[0]
        )

        return {
            "success": True,
            "start": start,
            "end": aggregation.period_end(start, grain),
            "description": (
                f"{formatting.format_period(start, grain)}"
            ),
            "notes": [],
        }

    # A window counted back from the most recent data.
    latest = usable.max()
    newest_start = pd.Timestamp(
        aggregation.period_starts(pd.Series([latest]), grain).iloc[0]
    )

    notes: list[str] = []

    if aggregation.final_period_is_incomplete(newest_start, latest, grain):
        skipped = formatting.format_period(newest_start, grain)
        newest_start = _shift_period(newest_start, grain, 1)

        notes.append(
            f"{skipped} is still in progress and has been left out, "
            "because a part-finished period reads as a fall rather than "
            "as incomplete data."
        )

    if window.offset:
        newest_start = _shift_period(newest_start, grain, window.offset)

    count = max(1, window.count or 1)
    oldest_start = _shift_period(newest_start, grain, count - 1)

    start = oldest_start
    end = aggregation.period_end(newest_start, grain)

    if count == 1:
        description = formatting.format_period(newest_start, grain)

    else:
        description = formatting.format_period_range(
            oldest_start,
            newest_start,
            grain
        )

    return {
        "success": True,
        "start": start,
        "end": end,
        "description": description,
        "notes": notes,
    }


def _compare_text(series: pd.Series) -> pd.Series:
    """Render a column as trimmed lowercase text, for label matching."""

    return series.astype("object").map(
        lambda value: (
            None
            if value is None or (
                isinstance(value, float) and np.isnan(value)
            )
            else str(value).strip().lower()
        )
    )


def apply_filters(
    frame: pd.DataFrame,
    filters: tuple[nlq.Filter, ...],
    trail: Trail
) -> pd.DataFrame:
    """
    Narrow the rows, recording what each condition removed.

    Label comparisons are case-insensitive on the written form of the
    value, because a question types "north" where the column holds
    "North". Numeric comparisons are done as numbers, and a value that
    will not read as one is treated as not matching rather than as zero.

    Args:
        frame:
            The dataset.

        filters:
            Conditions from the plan.

        trail:
            Collects the steps.

    Returns:
        The rows that met every condition.
    """

    working = frame

    for entry in filters:
        before = int(len(working))
        column = working[entry.column]

        if entry.operator == "is_missing":
            mask = column.isna()

        elif entry.operator == "is_present":
            mask = column.notna()

        elif entry.operator in nlq.NUMERIC_OPERATORS:
            numbers = pd.to_numeric(column, errors="coerce")
            values = [float(value) for value in entry.values]

            if entry.operator == "greater_than":
                mask = numbers > values[0]

            elif entry.operator == "greater_or_equal":
                mask = numbers >= values[0]

            elif entry.operator == "less_than":
                mask = numbers < values[0]

            elif entry.operator == "less_or_equal":
                mask = numbers <= values[0]

            else:
                mask = numbers.between(values[0], values[1])

            mask = mask.fillna(False)

        elif entry.operator == "contains":
            needle = str(entry.values[0]).strip().lower()
            mask = _compare_text(column).map(
                lambda value, needle=needle: (
                    value is not None and needle in value
                )
            )

        else:
            wanted = {
                str(value).strip().lower()
                for value in entry.values
            }
            rendered = _compare_text(column)

            if entry.operator in {"equals", "in"}:
                mask = rendered.isin(wanted)

            else:
                mask = ~rendered.isin(wanted)

        working = working[mask.to_numpy(dtype=bool)]
        after = int(len(working))

        trail.add(
            "Filtered rows",
            f"Kept rows where {entry.describe()}.",
            f"{after:,} of {before:,} rows remain"
        )

    return working


def _measure_values(
    frame: pd.DataFrame,
    plan: nlq.QueryPlan
) -> pd.Series:
    """The measure as numbers, with unreadable values dropped."""

    return pd.to_numeric(frame[plan.measure], errors="coerce").dropna()


def _combine(values: pd.Series, aggregation_name: str) -> float | None:
    """Apply one aggregation to a column of numbers."""

    if values.empty:
        return None

    if aggregation_name in {"sum", "sum_per_day"}:
        return float(values.sum())

    if aggregation_name == "mean":
        return float(values.mean())

    if aggregation_name == "median":
        return float(values.median())

    if aggregation_name == "max":
        return float(values.max())

    if aggregation_name == "min":
        return float(values.min())

    return float(len(values))


def _aggregation_noun(plan: nlq.QueryPlan) -> str:
    """Name the figure a plan produces, the way a sentence would."""

    if plan.measure is None or plan.aggregation == "count":
        return "the number of rows"

    return {
        "sum": f"total {plan.measure}",
        "sum_per_day": f"{plan.measure} per day",
        "mean": f"average {plan.measure}",
        "median": f"median {plan.measure}",
        "max": f"the highest {plan.measure}",
        "min": f"the lowest {plan.measure}",
    }.get(plan.aggregation, f"{plan.aggregation} of {plan.measure}")


def _windowed(
    frame: pd.DataFrame,
    plan: nlq.QueryPlan,
    trail: Trail,
    notes: list[str]
) -> tuple[pd.DataFrame, str] | dict[str, Any]:
    """
    Cut the rows down to the window the question asked about.

    Args:
        frame:
            Rows that already passed the filters.

        plan:
            The plan.

        trail:
            Collects the steps.

        notes:
            Collects things that change how to read the answer.

    Returns:
        The rows in the window and a phrase describing it, or a failure
        dictionary when the window could not be resolved.
    """

    if plan.window.kind == "all" or plan.date is None:
        return frame, "the whole dataset"

    dates = aggregation.parse_dates(frame[plan.date])
    resolved = resolve_window(dates, plan.window, plan.grain or "month")

    if not resolved.get("success"):
        return resolved

    mask = dates.notna()

    if resolved["start"] is not None:
        mask &= dates >= resolved["start"]

    if resolved["end"] is not None:
        mask &= dates <= resolved["end"]

    selected = frame[mask.to_numpy(dtype=bool)]
    notes.extend(resolved["notes"])

    trail.add(
        "Selected the period",
        (
            f"Kept rows whose '{plan.date}' falls in "
            f"{resolved['description']}."
        ),
        f"{len(selected):,} of {len(frame):,} rows remain"
    )

    return selected, resolved["description"]


def _period_series(
    frame: pd.DataFrame,
    plan: nlq.QueryPlan,
    trail: Trail
) -> dict[str, Any]:
    """Build the period series a time-based question needs."""

    aggregated = aggregation.aggregate_periods(
        frame,
        date_column=plan.date,
        measure_column=(
            None if plan.aggregation == "count" else plan.measure
        ),
        aggregation=plan.aggregation,
        grain=plan.grain
    )

    if aggregated.get("success"):
        trail.add(
            "Grouped by period",
            (
                f"Grouped rows into "
                f"{formatting.describe_grain(aggregated['grain'])} "
                f"periods on '{plan.date}' and combined them with "
                f"{plan.aggregation}."
            ),
            f"{aggregated['period_count']:,} periods"
        )

    return aggregated


def _segment_totals(
    frame: pd.DataFrame,
    plan: nlq.QueryPlan,
    trail: Trail
) -> dict[str, Any]:
    """
    Total the measure for each part of the segment.

    Grouping is done directly rather than through the period matrix, so a
    question like "revenue by category" still works on a table with no
    timeline in it at all.

    Args:
        frame:
            Rows in the window.

        plan:
            The plan.

        trail:
            Collects the steps.

    Returns:
        The ranked parts, and the share figures when shares mean anything.
    """

    labels = segments.segment_labels(frame[plan.segment]).astype(str)

    if plan.aggregation == "count" or plan.measure is None:
        totals = labels.value_counts()

    else:
        numbers = pd.to_numeric(frame[plan.measure], errors="coerce")
        usable = numbers.notna()
        totals = numbers[usable].groupby(labels[usable]).agg(
            "sum" if plan.aggregation == "sum_per_day"
            else plan.aggregation
        )

    if totals.empty:
        return {"success": False, "error": "No rows are left to group."}

    # One row of a period matrix, so the share rules live in one place
    # rather than being restated here.
    matrix = pd.DataFrame(
        [totals.to_numpy(dtype=float)],
        columns=[str(name) for name in totals.index],
        index=pd.DatetimeIndex([pd.Timestamp("2000-01-01")])
    )

    breakdown = segments.segment_breakdown(
        matrix,
        additive=plan.aggregation in segments.ADDITIVE_AGGREGATIONS,
        top_count=plan.top_n or segments.MAX_MATRIX_SEGMENTS
    )

    trail.add(
        "Grouped by segment",
        (
            f"Grouped rows by '{plan.segment}' and combined "
            f"{_aggregation_noun(plan)} within each part."
        ),
        f"{breakdown['part_count']:,} parts"
    )

    return {"success": True, "breakdown": breakdown}


def _ordered_table(
    breakdown: dict[str, Any],
    plan: nlq.QueryPlan
) -> pd.DataFrame:
    """Rank the parts the way the question asked."""

    table = breakdown["table"].copy()
    table = table.sort_values(
        "total",
        ascending=plan.direction == "lowest"
    )

    if plan.top_n:
        table = table.head(plan.top_n)

    if not breakdown["shares_meaningful"]:
        table = table.drop(columns=["share", "cumulative_share"])

    return table.head(MAX_ANSWER_ROWS).reset_index(drop=True)


def answer_aggregate(
    frame: pd.DataFrame,
    plan: nlq.QueryPlan,
    window_text: str,
    trail: Trail,
    notes: list[str]
) -> Answer:
    """Answer a question asking for one figure."""

    if plan.aggregation == "count" or plan.measure is None:
        figure = float(len(frame))
        detail = f"Counted the rows left: {len(frame):,}."

    else:
        values = _measure_values(frame, plan)
        figure = _combine(values, plan.aggregation)
        detail = (
            f"Applied {plan.aggregation} to "
            f"{len(values):,} usable '{plan.measure}' values."
        )

    if plan.aggregation == "sum_per_day" and figure is not None:
        days = _covered_days(frame, plan)

        if days:
            figure = figure / days
            detail += f" Divided by the {days:,} days covered."

        else:
            notes.append(
                "No usable dates were found, so a per-day rate could not "
                "be worked out and the total is shown instead."
            )

    trail.add("Calculated the figure", detail, formatting.format_number(figure))

    if figure is None:
        return Answer(
            success=True,
            plan=plan,
            headline=(
                f"No rows match, so {_aggregation_noun(plan)} cannot be "
                f"calculated for {window_text}."
            ),
            figures={"value": None, "rows": int(len(frame))},
            trail=trail.steps,
            notes=notes
        )

    return Answer(
        success=True,
        plan=plan,
        headline=(
            f"{_aggregation_noun(plan).capitalize()} over "
            f"{window_text} is {formatting.format_number(figure)}, "
            f"from {formatting.pluralise(int(len(frame)), 'row')}."
        ),
        figures={"value": figure, "rows": int(len(frame))},
        trail=trail.steps,
        notes=notes
    )


def _covered_days(frame: pd.DataFrame, plan: nlq.QueryPlan) -> int:
    """How many days of calendar the rows actually span."""

    if not plan.date or plan.date not in frame.columns:
        return 0

    dates = aggregation.parse_dates(frame[plan.date]).dropna()

    if dates.empty:
        return 0

    return int((dates.max() - dates.min()).days) + 1


def answer_timeseries(
    frame: pd.DataFrame,
    plan: nlq.QueryPlan,
    window_text: str,
    trail: Trail,
    notes: list[str]
) -> Answer:
    """Answer a question asking for a figure per period."""

    aggregated = _period_series(frame, plan, trail)

    if not aggregated.get("success"):
        return Answer(
            success=False,
            plan=plan,
            error=aggregated.get("error", "The series could not be built."),
            trail=trail.steps
        )

    periods = aggregated["periods"]
    notes.extend(aggregated["notes"])

    table = pd.DataFrame({
        "Period": periods["period_label"],
        _aggregation_noun(plan).capitalize(): periods["value"],
        "Rows": periods["row_count"],
    }).tail(MAX_ANSWER_ROWS).reset_index(drop=True)

    return Answer(
        success=True,
        plan=plan,
        headline=(
            f"{_aggregation_noun(plan).capitalize()} across "
            f"{formatting.pluralise(aggregated['period_count'], 'period')} "
            f"of {formatting.describe_grain(aggregated['grain'])} data, "
            f"{formatting.format_period_range(aggregated['first_period'], aggregated['last_period'], aggregated['grain'])}."
        ),
        figures={
            "period_count": aggregated["period_count"],
            "latest": (
                float(periods["value"].iloc[-1])
                if pd.notna(periods["value"].iloc[-1])
                else None
            ),
        },
        table=table,
        trail=trail.steps,
        notes=notes
    )


def answer_breakdown(
    frame: pd.DataFrame,
    plan: nlq.QueryPlan,
    window_text: str,
    trail: Trail,
    notes: list[str]
) -> Answer:
    """Answer a question asking for a figure per part."""

    grouped = _segment_totals(frame, plan, trail)

    if not grouped.get("success"):
        return Answer(
            success=False,
            plan=plan,
            error=grouped.get("error", "The breakdown could not be built."),
            trail=trail.steps
        )

    breakdown = grouped["breakdown"]

    ranking = breakdown["table"].sort_values(
        "total",
        ascending=plan.direction == "lowest"
    )

    if ranking.empty:
        return Answer(
            success=False,
            plan=plan,
            error="No parts were left to rank.",
            trail=trail.steps
        )

    # Rows with no label are a real part of the total, but they are not a
    # part of the business. Answering "which region is worst" with
    # "(not set)" is arithmetically true and tells the reader nothing about
    # regions, so the leading labelled part is named and the substitution
    # is stated. This has to happen before any limit is applied, or a
    # question asking for one part leaves nothing to promote.
    labelled = ranking[ranking["segment"] != segments.UNLABELLED_TEXT]
    leads_unlabelled = str(
        ranking["segment"].iloc[0]
    ) == segments.UNLABELLED_TEXT

    if leads_unlabelled and not labelled.empty:
        notes.append(
            f"Rows with no '{plan.segment}' would have been the "
            f"{plan.direction} group at "
            f"{formatting.format_number(float(ranking['total'].iloc[0]))}"
            f". They are counted in the total but are not a part of "
            f"'{plan.segment}', so they are left out of the ranking."
        )

        ranking = labelled

    if plan.top_n:
        ranking = ranking.head(plan.top_n)

    if not breakdown["shares_meaningful"]:
        ranking = ranking.drop(columns=["share", "cumulative_share"])

    table = ranking.head(MAX_ANSWER_ROWS).reset_index(drop=True)
    leader = table.iloc[0]
    ranked = "highest" if plan.direction == "highest" else "lowest"

    sentence = (
        f"The {ranked} '{plan.segment}' by "
        f"{_aggregation_noun(plan)} over {window_text} is "
        f"'{leader['segment']}' at "
        f"{formatting.format_number(leader['total'])}"
    )

    if breakdown["shares_meaningful"] and pd.notna(leader.get("share")):
        sentence += (
            f", which is {formatting.format_percent(leader['share'])} of "
            "the total"
        )

    sentence += "."

    if not breakdown["shares_meaningful"]:
        notes.append(breakdown["shares_withheld_because"])

    return Answer(
        success=True,
        plan=plan,
        headline=sentence,
        figures={
            "leader": str(leader["segment"]),
            "leader_total": float(leader["total"]),
            "part_count": breakdown["part_count"],
        },
        table=table.rename(columns={
            "segment": str(plan.segment),
            "total": _aggregation_noun(plan).capitalize(),
            "share": "Share",
            "cumulative_share": "Running share",
            "rank": "Rank",
        }),
        trail=trail.steps,
        notes=notes
    )


def answer_extreme(
    frame: pd.DataFrame,
    plan: nlq.QueryPlan,
    window_text: str,
    trail: Trail,
    notes: list[str]
) -> Answer:
    """Answer a question asking which period was largest or smallest."""

    aggregated = _period_series(frame, plan, trail)

    if not aggregated.get("success"):
        return Answer(
            success=False,
            plan=plan,
            error=aggregated.get("error", "The series could not be built."),
            trail=trail.steps
        )

    periods = aggregated["periods"]
    usable = periods[periods["value"].notna()]

    if usable.empty:
        return Answer(
            success=False,
            plan=plan,
            error="No period has a figure to compare.",
            trail=trail.steps
        )

    position = (
        usable["value"].idxmin()
        if plan.direction == "lowest"
        else usable["value"].idxmax()
    )
    chosen = periods.loc[position]
    notes.extend(aggregated["notes"])

    trail.add(
        "Picked the period",
        (
            f"Took the {plan.direction} of "
            f"{len(usable):,} period figures."
        ),
        f"{chosen['period_label']}: "
        f"{formatting.format_number(chosen['value'])}"
    )

    ordered = usable.sort_values(
        "value",
        ascending=plan.direction == "lowest"
    )

    table = pd.DataFrame({
        "Period": ordered["period_label"],
        _aggregation_noun(plan).capitalize(): ordered["value"],
        "Rows": ordered["row_count"],
    }).head(plan.top_n or MAX_ANSWER_ROWS).reset_index(drop=True)

    return Answer(
        success=True,
        plan=plan,
        headline=(
            f"The {plan.direction} "
            f"{formatting.describe_grain(aggregated['grain'])[:-2]} "
            f"period by {_aggregation_noun(plan)} is "
            f"{chosen['period_label']} at "
            f"{formatting.format_number(chosen['value'])}."
        ),
        figures={
            "period": str(chosen["period_label"]),
            "value": float(chosen["value"]),
        },
        table=table,
        trail=trail.steps,
        notes=notes
    )


def answer_trend(
    frame: pd.DataFrame,
    plan: nlq.QueryPlan,
    window_text: str,
    trail: Trail,
    notes: list[str]
) -> Answer:
    """Answer a question asking which way the figures are going."""

    aggregated = _period_series(frame, plan, trail)

    if not aggregated.get("success"):
        return Answer(
            success=False,
            plan=plan,
            error=aggregated.get("error", "The series could not be built."),
            trail=trail.steps
        )

    periods = aggregated["periods"]
    grain = aggregated["grain"]

    trend = timeseries.fit_trend(
        periods["period_start"],
        periods["value"]
    )

    notes.extend(aggregated["notes"])

    if not trend.get("success"):
        return Answer(
            success=False,
            plan=plan,
            error=trend.get(
                "error",
                "There is not enough history to judge a direction."
            ),
            trail=trail.steps,
            notes=notes
        )

    trail.add(
        "Fitted a trendline",
        (
            "Fitted a Theil-Sen line across the periods, which resists "
            "being pulled by one unusual period, and read the "
            "confidence interval of its slope."
        ),
        (
            f"{trend['direction']}, "
            f"{'significant' if trend['trend_is_significant'] else 'not distinguishable from flat'}"
        )
    )

    return Answer(
        success=True,
        plan=plan,
        headline=(
            f"{_aggregation_noun(plan).capitalize()}: "
            f"{timeseries.describe_trend(trend, grain)}"
        ),
        figures={
            "direction": trend["direction"],
            "slope_per_period": trend["slope_per_period"],
            "significant": bool(trend["trend_is_significant"]),
        },
        table=pd.DataFrame({
            "Period": periods["period_label"],
            _aggregation_noun(plan).capitalize(): periods["value"],
        }).tail(MAX_ANSWER_ROWS).reset_index(drop=True),
        trail=trail.steps,
        notes=notes
    )


def answer_compare(
    frame: pd.DataFrame,
    plan: nlq.QueryPlan,
    trail: Trail,
    notes: list[str]
) -> Answer:
    """Answer a question setting one period against another."""

    dates = aggregation.parse_dates(frame[plan.date])
    grain = plan.grain or "month"

    latest = resolve_window(dates, plan.window, grain)
    earlier = resolve_window(dates, plan.comparison, grain)

    if not latest.get("success") or not earlier.get("success"):
        return Answer(
            success=False,
            plan=plan,
            error=(
                latest.get("error")
                or earlier.get("error")
                or "The periods could not be resolved."
            ),
            trail=trail.steps
        )

    notes.extend(latest["notes"])

    figures = {}

    for label, resolved in (("latest", latest), ("previous", earlier)):
        mask = dates.notna()

        if resolved["start"] is not None:
            mask &= dates >= resolved["start"]

        if resolved["end"] is not None:
            mask &= dates <= resolved["end"]

        selected = frame[mask.to_numpy(dtype=bool)]

        if plan.aggregation == "count" or plan.measure is None:
            value = float(len(selected))

        else:
            value = _combine(
                _measure_values(selected, plan),
                plan.aggregation
            )

        figures[label] = value
        figures[f"{label}_label"] = resolved["description"]
        figures[f"{label}_rows"] = int(len(selected))

        trail.add(
            f"Measured {resolved['description']}",
            (
                f"Kept the {len(selected):,} rows in that period and "
                f"applied {plan.aggregation}."
            ),
            formatting.format_number(value)
        )

    current = figures["latest"]
    previous = figures["previous"]

    if current is None or previous is None:
        return Answer(
            success=True,
            plan=plan,
            headline=(
                f"One of the two periods has no usable rows, so "
                f"{figures['latest_label']} and "
                f"{figures['previous_label']} cannot be compared."
            ),
            figures=figures,
            trail=trail.steps,
            notes=notes
        )

    change = current - previous
    relative = change / abs(previous) if previous else None

    figures["change"] = change
    figures["relative_change"] = relative

    direction = "rose" if change > 0 else ("fell" if change < 0 else "held")

    sentence = (
        f"{_aggregation_noun(plan).capitalize()} {direction} from "
        f"{formatting.format_number(previous)} in "
        f"{figures['previous_label']} to "
        f"{formatting.format_number(current)} in "
        f"{figures['latest_label']}"
    )

    if relative is not None:
        sentence += f", a change of {formatting.format_signed_percent(relative)}"

    else:
        sentence += (
            ", and the earlier figure was zero so there is no percentage "
            "to quote"
        )

    sentence += "."

    trail.add(
        "Compared the periods",
        (
            f"{formatting.format_number(current)} minus "
            f"{formatting.format_number(previous)}"
            + (
                f", divided by {formatting.format_number(abs(previous))}"
                if relative is not None
                else ""
            )
            + "."
        ),
        formatting.format_signed(change)
    )

    return Answer(
        success=True,
        plan=plan,
        headline=sentence,
        figures=figures,
        table=pd.DataFrame({
            "Period": [
                figures["previous_label"],
                figures["latest_label"],
            ],
            _aggregation_noun(plan).capitalize(): [previous, current],
            "Rows": [
                figures["previous_rows"],
                figures["latest_rows"],
            ],
        }),
        trail=trail.steps,
        notes=notes
    )


def answer_share(
    frame: pd.DataFrame,
    plan: nlq.QueryPlan,
    window_text: str,
    trail: Trail,
    notes: list[str]
) -> Answer:
    """Answer a question asking what proportion one part accounts for."""

    grouped = _segment_totals(frame, plan, trail)

    if not grouped.get("success"):
        return Answer(
            success=False,
            plan=plan,
            error=grouped.get("error", "The breakdown could not be built."),
            trail=trail.steps
        )

    breakdown = grouped["breakdown"]

    if not breakdown["shares_meaningful"]:
        return Answer(
            success=False,
            plan=plan,
            error=breakdown["shares_withheld_because"],
            trail=trail.steps,
            notes=notes
        )

    table = breakdown["table"]
    wanted = str(plan.segment_value).strip().lower()

    rows = table[
        table["segment"].astype(str).str.strip().str.lower() == wanted
    ]

    if rows.empty:
        return Answer(
            success=False,
            plan=plan,
            error=(
                f"'{plan.segment_value}' has no rows in "
                f"{window_text}, so it has no share of the total."
            ),
            trail=trail.steps,
            notes=notes
        )

    row = rows.iloc[0]

    trail.add(
        "Divided the part by the whole",
        (
            f"{formatting.format_number(row['total'])} for "
            f"'{row['segment']}' divided by "
            f"{formatting.format_number(breakdown['grand_total'])} "
            "across every part."
        ),
        formatting.format_percent(row["share"])
    )

    return Answer(
        success=True,
        plan=plan,
        headline=(
            f"'{row['segment']}' accounts for "
            f"{formatting.format_percent(row['share'])} of "
            f"{_aggregation_noun(plan)} over {window_text}, "
            f"{formatting.format_number(row['total'])} of "
            f"{formatting.format_number(breakdown['grand_total'])}."
        ),
        figures={
            "part": str(row["segment"]),
            "total": float(row["total"]),
            "share": float(row["share"]),
            "grand_total": breakdown["grand_total"],
        },
        table=_ordered_table(breakdown, plan).rename(columns={
            "segment": str(plan.segment),
            "total": _aggregation_noun(plan).capitalize(),
            "share": "Share",
            "cumulative_share": "Running share",
            "rank": "Rank",
        }),
        trail=trail.steps,
        notes=notes
    )


def execute_plan(
    frame: pd.DataFrame,
    plan: nlq.QueryPlan
) -> Answer:
    """
    Run a validated plan and return the answer with its working.

    Args:
        frame:
            The dataset.

        plan:
            A plan that has passed nlq.validate_plan. Validation is
            repeated here, because a plan reaching execution unvalidated
            is the one path that could put an unchecked model suggestion
            in front of a reader as a figure.

    Returns:
        An Answer. Failure is reported rather than raised, because every
        caller wants to show the reason.
    """

    validation = nlq.validate_plan(plan, frame)

    if not validation.valid:
        return Answer(
            success=False,
            plan=plan,
            error=" ".join(validation.errors)
        )

    trail = Trail()
    notes = list(validation.warnings)

    trail.add(
        "Read the question",
        plan.describe(),
        f"{len(frame):,} rows to work from"
    )

    working = apply_filters(frame, plan.filters, trail)

    if working.empty and plan.filters:
        return Answer(
            success=True,
            plan=plan,
            headline=(
                "No rows match those conditions, so there is nothing to "
                "measure. The conditions were: "
                + "; ".join(
                    entry.describe() for entry in plan.filters
                )
                + "."
            ),
            figures={"rows": 0},
            trail=trail.steps,
            notes=notes
        )

    if plan.intent == "compare":
        return answer_compare(working, plan, trail, notes)

    selected = _windowed(working, plan, trail, notes)

    if isinstance(selected, dict):
        return Answer(
            success=False,
            plan=plan,
            error=selected.get("error", "The period could not be read."),
            trail=trail.steps,
            notes=notes
        )

    windowed, window_text = selected

    if windowed.empty:
        return Answer(
            success=True,
            plan=plan,
            headline=(
                f"No rows fall in {window_text}, so there is nothing to "
                "measure there."
            ),
            figures={"rows": 0},
            trail=trail.steps,
            notes=notes
        )

    handlers = {
        "aggregate": answer_aggregate,
        "timeseries": answer_timeseries,
        "breakdown": answer_breakdown,
        "extreme": answer_extreme,
        "trend": answer_trend,
        "share": answer_share,
    }

    handler = handlers.get(plan.intent)

    if handler is None:
        return Answer(
            success=False,
            plan=plan,
            error=f"'{plan.intent}' cannot be executed.",
            trail=trail.steps,
            notes=notes
        )

    return handler(windowed, plan, window_text, trail, notes)


def describe_trail(trail: list[dict[str, Any]]) -> str:
    """
    Summarise the working in one line, for an expander label.

    Args:
        trail:
            Steps from an Answer.

    Returns:
        A short description of how much work the answer rests on.
    """

    if not trail:
        return "No calculation ran"

    if len(trail) == 1:
        return "Show the calculation (1 step)"

    return f"Show the calculation ({len(trail)} steps)"


def unique_notes(*groups) -> list[str]:
    """
    Join note lists in order, keeping each note once.

    The parser and the executor both validate the plan, so the same
    warning, such as rows excluded for an unusable measure, used to be
    shown twice under one answer.
    """

    return list(dict.fromkeys(
        note
        for group in groups
        for note in group
    ))


def answer_question(
    frame: pd.DataFrame,
    question: str,
    invoke=None,
    glossary=None
) -> dict[str, Any]:
    """
    Answer a question, trying the rules before the model.

    The rules run first because most questions are simple, and a
    round-trip to a model to compute a sum is slow, costly and no more
    accurate. When the rules refuse, the question goes to the model as a
    planning problem, and whatever comes back is validated and executed
    here.

    Args:
        frame:
            The dataset.

        question:
            What the reader typed.

        invoke:
            A callable taking a prompt and returning text, used for the
            fallback. Pass None to keep everything local, in which case an
            unreadable question is refused with the parser's reason.

        glossary:
            Optional glossary.Glossary. When the question goes to the
            model, the most relevant definitions go with it.

    Returns:
        The answer, how the plan was arrived at, and everything the
        interface needs to be honest about both.
    """

    parsed = nlq_rules.parse_question(question, frame)

    if parsed.understood:
        answer = execute_plan(frame, parsed.plan)

        return {
            "answer": answer,
            "planned_by": "rules",
            "plan": parsed.plan,
            "matched": parsed.matched,
            "ignored": parsed.ignored,
            "notes": unique_notes(parsed.warnings, answer.notes),
            "rules_reason": "",
            "model_replies": [],
            "not_on_topic": False,
            "model_unanswerable": False,
            "model_unreachable": False,
            "glossary_used": [],
        }

    if invoke is None:
        return {
            "answer": Answer(
                success=False,
                error=parsed.reason,
                notes=list(parsed.warnings)
            ),
            "planned_by": "nobody",
            "plan": None,
            "matched": parsed.matched,
            "ignored": parsed.ignored,
            "notes": list(parsed.warnings),
            "rules_reason": parsed.reason,
            "model_replies": [],
            "not_on_topic": False,
            "model_unanswerable": False,
            "model_unreachable": False,
            "glossary_used": [],
        }

    # Retrieval runs only on this path: a question the rules answered
    # needed no help with its wording.
    retrieved = glossary.retrieve(question) if glossary is not None else []

    planned = nlq_model.plan_with_model(
        question,
        frame,
        invoke,
        definitions=[chunk.as_prompt_text() for chunk in retrieved] or None
    )
    glossary_used = [chunk.term or chunk.id for chunk in retrieved]

    if not planned.understood:
        return {
            "answer": Answer(
                success=False,
                error=(
                    f"{parsed.reason} The model was asked to plan it "
                    f"instead, and that did not work either: "
                    f"{planned.reason}"
                ),
                notes=list(parsed.warnings)
            ),
            "planned_by": "nobody",
            "plan": None,
            "matched": parsed.matched,
            "ignored": parsed.ignored,
            "notes": list(parsed.warnings),
            "rules_reason": parsed.reason,
            "model_replies": planned.replies,
            "not_on_topic": planned.not_on_topic,
            "model_unanswerable": planned.unanswerable,
            "model_unreachable": planned.unreachable,
            "glossary_used": glossary_used,
        }

    answer = execute_plan(frame, planned.plan)

    return {
        "answer": answer,
        "planned_by": "model",
        "plan": planned.plan,
        "matched": parsed.matched,
        "ignored": parsed.ignored,
        "notes": unique_notes(
            parsed.warnings,
            planned.warnings,
            answer.notes
        ),
        "rules_reason": parsed.reason,
        "model_replies": planned.replies,
        "not_on_topic": False,
        "model_unanswerable": False,
        "model_unreachable": False,
        "glossary_used": glossary_used,
    }
