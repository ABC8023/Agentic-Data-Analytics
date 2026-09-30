"""
Splitting a total into the parts that make it up.

The whole point of a segment breakdown is that the parts add back to the
total. If they do not, the reader is being shown two different datasets
side by side and invited to draw conclusions from the difference. So the
arithmetic here is deliberately built to reconcile, and where it cannot,
it says so rather than quietly presenting numbers that do not tie up.

Three things make reconciliation harder than it looks.

Rows with no segment label. Dropping them is the tempting choice and the
wrong one: a breakdown missing a fifth of the data will not add up, and
nothing on the screen would explain why. They are kept under an explicit
"not set" label instead, and counted, so the gap is visible.

Aggregations that do not add. A total splits into parts. An average does
not: the average of the parts is not the average of the whole unless every
part has the same number of rows. Shares of total and a change waterfall
are therefore only offered for the aggregations where they are true.

Period alignment. The headline series drops an unfinished final period and
fills calendar gaps. A breakdown computed independently would do neither
and would disagree with the chart above it. The period series is passed in
and the breakdown is pinned to it.
"""

from typing import Any

import numpy as np
import pandas as pd

import aggregation
import formatting

# Aggregations where a total genuinely divides into parts, so shares and a
# change waterfall mean something. Averages and medians are excluded: the
# mean of the parts is not the mean of the whole.
ADDITIVE_AGGREGATIONS = frozenset({"sum", "count", "sum_per_day"})

# Label given to rows whose segment value is blank. Named rather than
# dropped, so the breakdown still adds up to the headline total.
UNLABELLED_TEXT = "(not set)"

# Parts shown individually in a matrix before the tail is grouped. Beyond
# roughly this many rows a heatmap stops being readable.
MAX_MATRIX_SEGMENTS = 12

# Bars shown individually in a change waterfall before the tail is grouped.
# A waterfall with forty steps communicates nothing.
MAX_WATERFALL_STEPS = 8

# A segment column with almost as many distinct values as rows is an
# identifier, and grouping by it produces one row per part.
IDENTIFIER_DISTINCT_RATIO = 0.5

# Minimum number of distinct parts worth breaking down. One part is the
# total again.
MINIMUM_SEGMENT_VALUES = 2


def segment_labels(values: pd.Series) -> pd.Series:
    """
    Turn a column into segment labels, naming the blanks.

    Args:
        values:
            The raw column.

    Returns:
        Strings, with missing and empty values replaced by an explicit
        label rather than removed.
    """

    labels = values.astype("object").where(values.notna(), None)
    labels = labels.map(
        lambda entry: (
            UNLABELLED_TEXT
            if entry is None or str(entry).strip() == ""
            else str(entry)
        )
    )

    return labels.astype("string").fillna(UNLABELLED_TEXT)


def describe_segment_suitability(
    frame: pd.DataFrame,
    segment: str
) -> dict[str, Any]:
    """
    Decide whether a column is worth breaking a total down by.

    Args:
        frame:
            The dataset.

        segment:
            Candidate column.

    Returns:
        Whether it is usable, why not when it is not, and the distinct
        count either way.
    """

    if segment not in frame.columns:
        return {
            "usable": False,
            "reason": f"There is no column called '{segment}'.",
            "distinct": 0,
        }

    labels = segment_labels(frame[segment])
    distinct = int(labels.nunique(dropna=False))
    rows = int(len(frame))

    if distinct < MINIMUM_SEGMENT_VALUES:
        return {
            "usable": False,
            "reason": (
                f"'{segment}' holds the same value in every row, so "
                "splitting by it just repeats the total."
            ),
            "distinct": distinct,
        }

    if rows and distinct / rows > IDENTIFIER_DISTINCT_RATIO:
        return {
            "usable": False,
            "reason": (
                f"'{segment}' has {distinct:,} distinct values across "
                f"{rows:,} rows, so it identifies rows rather than "
                "grouping them. Breaking the total down by it would "
                "produce one part per row."
            ),
            "distinct": distinct,
        }

    return {"usable": True, "reason": "", "distinct": distinct}


def segment_matrix(
    frame: pd.DataFrame,
    date: str,
    segment: str,
    periods: pd.DataFrame,
    measure: str | None = None,
    aggregation_name: str = "sum",
    grain: str = "month"
) -> dict[str, Any]:
    """
    Build a period-by-segment table pinned to an existing period series.

    The rows are exactly the periods of the series passed in, so the
    breakdown covers the same span as the headline chart and excludes the
    same unfinished final period.

    Args:
        frame:
            The dataset.

        date:
            Column holding the dates.

        segment:
            Column to split by.

        periods:
            The "periods" frame from aggregation.aggregate_periods, whose
            period_start column defines the rows and whose period_days
            column is reused so a per-day rate divides by the same
            denominator for every part.

        measure:
            Column being measured, or None to count rows.

        aggregation_name:
            One of aggregation.AGGREGATIONS.

        grain:
            Period grain, used for labels.

    Returns:
        The matrix, whether the parts add to the total, how far off the
        reconciliation is, and how many rows carried no segment label.
    """

    suitability = describe_segment_suitability(frame, segment)

    if not suitability["usable"]:
        return {"success": False, "error": suitability["reason"]}

    if date not in frame.columns:
        return {
            "success": False,
            "error": f"There is no column called '{date}'.",
        }

    if measure is not None and measure not in frame.columns:
        return {
            "success": False,
            "error": f"There is no column called '{measure}'.",
        }

    if aggregation_name not in aggregation.AGGREGATIONS:
        return {
            "success": False,
            "error": (
                f"'{aggregation_name}' is not a supported aggregation."
            ),
        }

    # Mirror the headline series exactly: counting rows keeps rows whose
    # measure is unreadable, because they are still rows.
    counting_rows = measure is None or aggregation_name == "count"
    additive = aggregation_name in ADDITIVE_AGGREGATIONS

    working = pd.DataFrame({
        "_date": aggregation.parse_dates(frame[date]),
        "_segment": segment_labels(frame[segment]).astype(str),
    })

    if not counting_rows:
        working["_measure"] = pd.to_numeric(
            frame[measure],
            errors="coerce"
        )

    working = working[working["_date"].notna()]

    if not counting_rows:
        working = working[working["_measure"].notna()]

    if working.empty:
        return {
            "success": False,
            "error": (
                "No rows survived with both a readable date and a usable "
                "measure, so there is nothing to break down."
            ),
        }

    working["_period"] = aggregation.period_starts(working["_date"], grain)

    wanted = pd.DatetimeIndex(periods["period_start"])
    working = working[working["_period"].isin(wanted)]

    if working.empty:
        return {
            "success": False,
            "error": (
                "No rows fall inside the periods being shown, so the "
                "breakdown would be empty."
            ),
        }

    unlabelled_rows = int(
        (working["_segment"] == UNLABELLED_TEXT).sum()
    )

    grouped = working.groupby(["_period", "_segment"], sort=True)

    if counting_rows:
        cells = grouped.size()

    elif aggregation_name == "sum_per_day":
        cells = grouped["_measure"].sum()

    else:
        cells = grouped["_measure"].agg(aggregation_name)

    matrix = cells.unstack("_segment")
    matrix = matrix.reindex(wanted)
    matrix.index.name = "period_start"
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    matrix = matrix.astype(float)

    # A part with no rows in a period contributed nothing to the total,
    # and nothing is zero. For an average it is not zero, it is unknown.
    if additive:
        matrix = matrix.fillna(0.0)

    if aggregation_name == "sum_per_day":
        days = pd.Series(
            periods["period_days"].to_numpy(dtype=float),
            index=wanted
        )
        matrix = matrix.div(days.replace(0.0, np.nan), axis=0)

    reconciliation = _check_reconciliation(matrix, periods, additive)

    labels = [
        formatting.format_period(start, grain)
        for start in matrix.index
    ]

    return {
        "success": True,
        "matrix": matrix,
        "period_labels": labels,
        "segment_column": segment,
        "segment_count": int(matrix.shape[1]),
        "additive": additive,
        "aggregation": aggregation_name,
        "grain": grain,
        "unlabelled_rows": unlabelled_rows,
        "unlabelled_share": (
            unlabelled_rows / int(len(working))
            if len(working)
            else 0.0
        ),
        "rows_used": int(len(working)),
        **reconciliation,
    }


def _check_reconciliation(
    matrix: pd.DataFrame,
    periods: pd.DataFrame,
    additive: bool
) -> dict[str, Any]:
    """
    Test whether the parts add back to the headline totals.

    Args:
        matrix:
            Period-by-segment values.

        periods:
            The headline period series.

        additive:
            Whether the aggregation is one where they should add.

    Returns:
        Whether the parts reconcile and the largest discrepancy found.
    """

    if not additive:
        return {
            "reconciles": False,
            "largest_discrepancy": None,
            "reconciliation_note": (
                "The parts are not expected to add back to the total, "
                "because an average of the parts is not the average of "
                "the whole."
            ),
        }

    totals = pd.Series(
        periods["value"].to_numpy(dtype=float),
        index=matrix.index
    )

    difference = (matrix.sum(axis=1, min_count=1) - totals).abs()
    largest = float(difference.max()) if len(difference) else 0.0

    if not np.isfinite(largest):
        largest = 0.0

    scale = float(
        np.nanmax(np.abs(totals.to_numpy(dtype=float)))
        if len(totals)
        else 1.0
    )

    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    reconciles = largest <= max(1e-9, 1e-6 * scale)

    return {
        "reconciles": bool(reconciles),
        "largest_discrepancy": largest,
        "reconciliation_note": (
            "The parts add back to each period total."
            if reconciles
            else (
                "The parts do not add back to the period totals, so the "
                "breakdown should not be read as a decomposition."
            )
        ),
    }


def segment_totals(matrix: pd.DataFrame) -> pd.Series:
    """
    Total each part across every period.

    Args:
        matrix:
            Period-by-segment values.

    Returns:
        One figure per part, largest first.
    """

    totals = matrix.sum(axis=0, min_count=1)

    return totals.sort_values(ascending=False)


def segment_breakdown(
    matrix: pd.DataFrame,
    additive: bool,
    top_count: int = 5
) -> dict[str, Any]:
    """
    Rank the parts and measure how much rests on the largest.

    Shares are only computed when they mean something. With a negative
    part in the mix a share of the total can exceed 100% or come out
    negative, which reads as a bug rather than as the offsetting figures
    it actually is, so shares are withheld and the reason recorded.

    Args:
        matrix:
            Period-by-segment values.

        additive:
            Whether the aggregation divides into parts.

        top_count:
            How many parts to name.

    Returns:
        The ranked table, the concentration figures, and whether shares
        were computable.
    """

    totals = segment_totals(matrix)
    grand_total = float(totals.sum())
    has_negative = bool((totals < 0).any())

    shares_meaningful = (
        additive
        and grand_total > 0
        and not has_negative
    )

    table = pd.DataFrame({
        "segment": totals.index.astype(str),
        "total": totals.to_numpy(dtype=float),
    })

    if shares_meaningful:
        table["share"] = table["total"] / grand_total
        table["cumulative_share"] = table["share"].cumsum()

    else:
        table["share"] = np.nan
        table["cumulative_share"] = np.nan

    table["rank"] = np.arange(1, len(table) + 1)

    top = table.head(top_count).copy()

    largest_share = (
        float(table["share"].iloc[0])
        if shares_meaningful and len(table)
        else None
    )

    # How the largest part compares with an even split. A bare share is
    # not enough to judge concentration: half the total is dominance when
    # there are ten parts and exactly even when there are two.
    even_share = 1.0 / len(table) if len(table) else None

    dominance = (
        largest_share / even_share
        if largest_share is not None and even_share
        else None
    )

    top_share = (
        float(top["share"].sum())
        if shares_meaningful and len(top)
        else None
    )

    if shares_meaningful and len(table):
        needed = int((table["cumulative_share"] < 0.8).sum()) + 1
        parts_for_80 = min(needed, len(table))

    else:
        parts_for_80 = None

    return {
        "table": table,
        "top": top,
        "part_count": int(len(table)),
        "grand_total": grand_total,
        "largest_segment": (
            str(table["segment"].iloc[0]) if len(table) else None
        ),
        "largest_share": largest_share,
        "even_share": even_share,
        "dominance": dominance,
        "top_share": top_share,
        "top_count": int(len(top)),
        "parts_for_80_percent": parts_for_80,
        "shares_meaningful": shares_meaningful,
        "shares_withheld_because": (
            ""
            if shares_meaningful
            else _explain_withheld_shares(
                additive,
                grand_total,
                has_negative
            )
        ),
    }


def _explain_withheld_shares(
    additive: bool,
    grand_total: float,
    has_negative: bool
) -> str:
    """
    Say why a share of the total was not computed.

    Args:
        additive:
            Whether the aggregation divides into parts.

        grand_total:
            Sum across the parts.

        has_negative:
            Whether any part came out below zero.

    Returns:
        A sentence.
    """

    if not additive:
        return (
            "Shares of a total only mean something when the parts add "
            "up, which averages and medians do not."
        )

    if has_negative:
        return (
            "At least one part is negative, so a share of the total "
            "could exceed 100% or come out below zero. The figures are "
            "shown without shares instead."
        )

    if grand_total == 0:
        return (
            "The parts sum to zero, so there is no total for them to be "
            "a share of."
        )

    return "Shares could not be computed for this breakdown."


def segment_waterfall(
    matrix: pd.DataFrame,
    periods: pd.DataFrame,
    additive: bool,
    grain: str = "month"
) -> dict[str, Any]:
    """
    Account for the latest period-on-period change part by part.

    Args:
        matrix:
            Period-by-segment values.

        periods:
            The headline period series.

        additive:
            Whether the aggregation divides into parts.

        grain:
            Period grain, for labels.

    Returns:
        A row per part with its change and contribution, the residual left
        over, and whether the parts fully account for the movement.
    """

    if len(matrix) < 2:
        return {
            "success": False,
            "error": (
                "There is only one period, so there is no change to "
                "account for."
            ),
        }

    latest_total = periods["value"].iloc[-1]
    previous_total = periods["value"].iloc[-2]

    if pd.isna(latest_total) or pd.isna(previous_total):
        return {
            "success": False,
            "error": (
                "One of the last two periods has no value, so the change "
                "cannot be split up."
            ),
        }

    latest = matrix.iloc[-1]
    previous = matrix.iloc[-2]

    if additive:
        latest = latest.fillna(0.0)
        previous = previous.fillna(0.0)

    change = latest - previous
    total_change = float(latest_total) - float(previous_total)

    table = pd.DataFrame({
        "segment": matrix.columns.astype(str),
        "previous": previous.to_numpy(dtype=float),
        "latest": latest.to_numpy(dtype=float),
        "change": change.to_numpy(dtype=float),
    })

    table["relative_change"] = np.where(
        table["previous"] != 0,
        table["change"] / table["previous"].abs(),
        np.nan
    )

    if additive and total_change != 0:
        table["contribution"] = table["change"] / total_change

    else:
        table["contribution"] = np.nan

    table["entered"] = (table["previous"] == 0) & (table["latest"] != 0)
    table["exited"] = (table["latest"] == 0) & (table["previous"] != 0)

    table = table.reindex(
        table["change"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    accounted = float(np.nansum(table["change"].to_numpy(dtype=float)))
    residual = total_change - accounted

    scale = max(
        abs(float(latest_total)),
        abs(float(previous_total)),
        1.0
    )

    reconciles = additive and abs(residual) <= 1e-6 * scale

    moved_up = int((table["change"] > 0).sum())
    moved_down = int((table["change"] < 0).sum())
    offsetting = bool(moved_up and moved_down)

    return {
        "success": True,
        "table": table,
        "latest_label": formatting.format_period(matrix.index[-1], grain),
        "previous_label": formatting.format_period(
            matrix.index[-2],
            grain
        ),
        "latest_total": float(latest_total),
        "previous_total": float(previous_total),
        "total_change": total_change,
        "accounted_change": accounted,
        "residual": float(residual),
        "reconciles": bool(reconciles),
        "explains_total": bool(reconciles),
        "parts_moved_up": moved_up,
        "parts_moved_down": moved_down,
        "offsetting_movements": offsetting,
        "entered_count": int(table["entered"].sum()),
        "exited_count": int(table["exited"].sum()),
        "additive": additive,
    }


def waterfall_steps(
    table: pd.DataFrame,
    limit: int = MAX_WATERFALL_STEPS
) -> dict[str, Any]:
    """
    Reduce a change table to the largest movers plus a grouped remainder.

    The remainder is a sum rather than a truncation, so the steps still
    add to the same figure as the full table. A waterfall whose bars do
    not reach the closing total is worse than no waterfall.

    Args:
        table:
            The "table" from segment_waterfall.

        limit:
            How many movers to show individually.

    Returns:
        The steps to draw and how many parts were grouped.
    """

    ordered = table.reindex(
        table["change"].abs().sort_values(ascending=False).index
    )

    columns = ["segment", "change"]

    if len(ordered) <= limit:
        return {
            "steps": ordered[columns].reset_index(drop=True),
            "grouped_count": 0,
            "other_label": "",
        }

    kept = ordered.iloc[:limit][columns]
    tail = ordered.iloc[limit:]
    label = f"All other parts ({len(tail):,})"

    steps = pd.concat(
        [
            kept,
            pd.DataFrame([{
                "segment": label,
                "change": float(tail["change"].sum()),
            }]),
        ],
        ignore_index=True
    )

    return {
        "steps": steps,
        "grouped_count": int(len(tail)),
        "other_label": label,
    }


def collapse_matrix(
    matrix: pd.DataFrame,
    limit: int = MAX_MATRIX_SEGMENTS
) -> dict[str, Any]:
    """
    Reduce a wide matrix to the largest parts plus a grouped tail.

    Args:
        matrix:
            Period-by-segment values.

        limit:
            How many parts to keep separate.

    Returns:
        The reduced matrix, ordered largest first, and how many parts were
        grouped into the tail.
    """

    totals = matrix.sum(axis=0, min_count=1)
    ordered = totals.abs().sort_values(ascending=False).index.tolist()

    if len(ordered) <= limit:
        return {
            "matrix": matrix.reindex(ordered, axis=1),
            "grouped_count": 0,
            "other_label": "",
        }

    kept = ordered[:limit]
    tail = ordered[limit:]

    reduced = matrix.reindex(kept, axis=1).copy()
    label = f"Other ({len(tail):,} parts)"
    reduced[label] = matrix.reindex(tail, axis=1).sum(
        axis=1,
        min_count=1
    )

    return {
        "matrix": reduced,
        "grouped_count": len(tail),
        "other_label": label,
    }


def segment_period_shares(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each period's row into shares of that period's total.

    Reading a heatmap of raw totals mostly shows which periods were big.
    Sharing each row out of its own total shows mix shifting instead,
    which is usually the question being asked.

    Args:
        matrix:
            Period-by-segment values.

    Returns:
        A matrix of shares, empty where a period total was zero or the
        row held a negative value that makes a share meaningless.
    """

    values = matrix.to_numpy(dtype=float)
    totals = np.nansum(values, axis=1)

    unusable = (
        (totals == 0)
        | ~np.isfinite(totals)
        | np.any(values < 0, axis=1)
    )

    safe_totals = np.where(unusable, np.nan, totals)
    shares = values / safe_totals[:, None]

    return pd.DataFrame(
        shares,
        index=matrix.index,
        columns=matrix.columns
    )


def drill_down(
    frame: pd.DataFrame,
    segment: str,
    value: str,
    next_segment: str,
    date: str,
    periods: pd.DataFrame,
    measure: str | None = None,
    aggregation_name: str = "sum",
    grain: str = "month"
) -> dict[str, Any]:
    """
    Re-run the breakdown inside one part, grouped by another dimension.

    Args:
        frame:
            The dataset.

        segment:
            Column already broken down.

        value:
            The part to look inside.

        next_segment:
            Column to group by within that part.

        date:
            Column holding the dates.

        periods:
            The headline period series, reused so the rows still line up.

        measure:
            Column being measured, or None to count rows.

        aggregation_name:
            One of aggregation.AGGREGATIONS.

        grain:
            Period grain.

    Returns:
        The same structure as segment_matrix, restricted to the chosen
        part, plus the share of rows it accounts for.
    """

    if segment not in frame.columns:
        return {
            "success": False,
            "error": f"There is no column called '{segment}'.",
        }

    labels = segment_labels(frame[segment]).astype(str)
    selected = frame[labels == str(value)]

    if selected.empty:
        return {
            "success": False,
            "error": (
                f"No rows have '{segment}' equal to '{value}'."
            ),
        }

    result = segment_matrix(
        selected,
        date=date,
        segment=next_segment,
        periods=periods,
        measure=measure,
        aggregation_name=aggregation_name,
        grain=grain
    )

    result["parent_segment"] = segment
    result["parent_value"] = str(value)
    result["parent_rows"] = int(len(selected))
    result["parent_row_share"] = (
        int(len(selected)) / int(len(frame)) if len(frame) else 0.0
    )

    if not result.get("success"):
        # Scope the refusal to the part being examined. A column can group
        # the whole dataset perfectly and still hold one value inside a
        # single part, and the unscoped message would read as though the
        # column were unusable everywhere.
        result["error"] = (
            f"Inside '{segment}' = '{value}': {result.get('error', '')}"
        )

        return result

    # Within one part the figures no longer add to the headline total, and
    # saying otherwise would be wrong however the arithmetic came out.
    result["reconciles"] = False
    result["reconciliation_note"] = (
        f"These figures cover only the rows where '{segment}' is "
        f"'{value}', so they add up to that part rather than to the "
        "headline total."
    )

    return result
