"""
The single place that decides what a period is.

Every figure that moves over time in this application is built from
aggregate_periods, so a trend, an anomaly band, a forecast and a movement
comparison can never disagree about which rows belong to which month.

Two adjustments are made that most tools skip, because skipping them
produces charts that lie:

An in-progress final period is excluded. A month that is four days old
holds four days of sales, so leaving it in draws a cliff at the right hand
edge and invites the reader to conclude the business collapsed.

Periods with no rows are filled with zero when zero is the truthful
answer. If nothing sold in March, the total for March is zero, and
skipping the period would draw a straight line through it as though the
gap never happened. Zero is only truthful for a total or a count, so an
average or a median leaves the gap empty instead of inventing a value.

Both adjustments are reported rather than applied silently.
"""

from typing import Any

import pandas as pd

import formatting

PERIOD_GRAINS = ("day", "week", "month", "quarter", "year")

# Frequency aliases used to build the period index for each grain.
GRAIN_FREQUENCY = {
    "day": "D",
    "week": "W-MON",
    "month": "MS",
    "quarter": "QS",
    "year": "YS",
}

# Aliases used with pandas Period, which needs the anchored form.
GRAIN_PERIOD_ALIAS = {
    "day": "D",
    "week": "W-MON",
    "month": "M",
    "quarter": "Q",
    "year": "Y",
}

# Aggregations offered for a measure. Only a total and a count can be
# zero-filled, because zero is a real answer for those and a fiction for
# the rest.
AGGREGATIONS = (
    "sum",
    "sum_per_day",
    "mean",
    "median",
    "count",
    "min",
    "max",
)

ZERO_FILLABLE = frozenset({"sum", "count", "sum_per_day"})

# Aggregations that grow with how much time a period covers. A total for
# February is smaller than one for January for no reason other than the
# calendar, which manufactures a fall that no anomaly detector should be
# asked to interpret.
EXPOSURE_SENSITIVE = frozenset({"sum", "count"})

# Ratio of longest to shortest period beyond which unequal exposure is
# worth warning about. Months alone reach 31/28, about 1.107.
EXPOSURE_WARNING_RATIO = 1.05

# Grain chosen from how much calendar the data covers. A fortnight of data
# reads best daily; five years reads best quarterly.
GRAIN_THRESHOLDS = (
    (31, "day"),
    (120, "week"),
    (800, "month"),
    (2200, "quarter"),
)

# Below this many usable periods a trend or forecast is not worth drawing.
MINIMUM_PERIODS_FOR_TREND = 4


def parse_dates(values: Any) -> pd.Series:
    """
    Convert a column to timestamps, discarding what cannot be read.

    Args:
        values:
            The column to convert.

    Returns:
        A datetime series with unparseable entries as missing.
    """

    series = pd.Series(values)

    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    return pd.to_datetime(series, errors="coerce", format="mixed")


def detect_grain(dates: pd.Series) -> str:
    """
    Choose a period grain from the span the data covers.

    Args:
        dates:
            Parsed datetime series.

    Returns:
        One of PERIOD_GRAINS.
    """

    usable = pd.Series(dates).dropna()

    if usable.empty:
        return "month"

    span_days = int(
        (usable.max() - usable.min()) / pd.Timedelta(days=1)
    )

    for threshold, grain in GRAIN_THRESHOLDS:
        if span_days <= threshold:
            return grain

    return "year"


def period_starts(dates: pd.Series, grain: str) -> pd.Series:
    """
    Map timestamps to the first moment of the period containing them.

    Args:
        dates:
            Parsed datetime series.

        grain:
            One of PERIOD_GRAINS.

    Returns:
        A datetime series of period starts.
    """

    alias = GRAIN_PERIOD_ALIAS.get(grain, "M")

    return pd.Series(dates).dt.to_period(alias).dt.start_time


def period_end(start: Any, grain: str) -> pd.Timestamp:
    """
    Return the last moment belonging to a period.

    Args:
        start:
            Start of the period.

        grain:
            One of PERIOD_GRAINS.

    Returns:
        The end of the period.
    """

    alias = GRAIN_PERIOD_ALIAS.get(grain, "M")

    return pd.Period(pd.Timestamp(start), freq=alias).end_time


def final_period_is_incomplete(
    last_period_start: Any,
    last_observation: Any,
    grain: str
) -> bool:
    """
    Report whether the newest period is still being filled.

    Dates are compared at day resolution, because a daily series whose
    timestamps sit at midnight would otherwise look permanently
    unfinished.

    Args:
        last_period_start:
            Start of the newest period present.

        last_observation:
            The newest timestamp in the data.

        grain:
            One of PERIOD_GRAINS.

    Returns:
        True when the data stops before the period does.
    """

    closes = period_end(last_period_start, grain).normalize()
    observed = pd.Timestamp(last_observation).normalize()

    return observed < closes


def _observed_days(
    start: Any,
    grain: str,
    first_observation: Any,
    last_observation: Any
) -> int:
    """
    Count the calendar days of a period that the data actually covers.

    A period at either end of the series is usually partial, and a total
    over a partial period is not comparable with a full one.

    Args:
        start:
            Period start.

        grain:
            Period grain.

        first_observation:
            Earliest date in the data.

        last_observation:
            Latest date in the data.

    Returns:
        Days covered, at least one.
    """

    period_first = max(
        pd.Timestamp(start).normalize(),
        pd.Timestamp(first_observation).normalize()
    )
    period_last = min(
        period_end(start, grain).normalize(),
        pd.Timestamp(last_observation).normalize()
    )

    covered = (period_last - period_first) / pd.Timedelta(days=1) + 1

    return max(int(covered), 1)


def aggregate_periods(
    frame: pd.DataFrame,
    date_column: str,
    measure_column: str | None = None,
    aggregation: str = "sum",
    grain: str | None = None,
    drop_incomplete_final: bool = True,
    fill_gaps: bool = True
) -> dict[str, Any]:
    """
    Build a period series from a table of rows.

    Args:
        frame:
            The dataset.

        date_column:
            Column holding the dates.

        measure_column:
            Column to aggregate. Omit to count rows per period.

        aggregation:
            One of AGGREGATIONS.

        grain:
            Period grain. Detected from the data when omitted.

        drop_incomplete_final:
            Whether to exclude a final period the data does not fill.

        fill_gaps:
            Whether to insert missing periods. Filled with zero for a
            total or a count, and left empty for anything else.

    Returns:
        The period series under "periods" together with the grain, the row
        accounting, and the notes describing every adjustment made.
    """

    if date_column not in frame.columns:
        return {
            "success": False,
            "error": f"There is no column called '{date_column}'."
        }

    counting_rows = (
        measure_column is None
        or aggregation == "count"
    )



    if (
        measure_column is not None
        and measure_column not in frame.columns
    ):
        return {
            "success": False,
            "error": f"There is no column called '{measure_column}'."
        }

    if aggregation not in AGGREGATIONS:
        return {
            "success": False,
            "error": (
                f"'{aggregation}' is not a supported aggregation. "
                f"Choose one of: {', '.join(AGGREGATIONS)}."
            )
        }

    total_rows = int(len(frame))

    working = pd.DataFrame({
        "_date": parse_dates(frame[date_column])
    })

    if not counting_rows:
        working["_measure"] = pd.to_numeric(
            frame[measure_column],
            errors="coerce"
        )

    unreadable_dates = int(working["_date"].isna().sum())
    working = working[working["_date"].notna()]

    if working.empty:
        return {
            "success": False,
            "error": (
                f"No usable dates were found in '{date_column}'. "
                f"{unreadable_dates:,} of {total_rows:,} values could "
                "not be read as a date."
            )
        }

    unusable_measure = 0

    if not counting_rows:
        unusable_measure = int(working["_measure"].isna().sum())
        working = working[working["_measure"].notna()]

        if working.empty:
            return {
                "success": False,
                "error": (
                    f"No usable numbers were found in "
                    f"'{measure_column}'."
                )
            }

    resolved_grain = grain or detect_grain(working["_date"])

    if resolved_grain not in GRAIN_FREQUENCY:
        return {
            "success": False,
            "error": (
                f"'{resolved_grain}' is not a supported period. "
                f"Choose one of: {', '.join(PERIOD_GRAINS)}."
            )
        }

    last_observation = working["_date"].max()
    working["_period"] = period_starts(working["_date"], resolved_grain)

    grouped = working.groupby("_period", sort=True)

    if counting_rows:
        values = grouped.size().rename("value")
        row_counts = values.copy()

    elif aggregation == "sum_per_day":
        values = grouped["_measure"].sum().rename("value")
        row_counts = grouped.size().rename("row_count")

    else:
        values = grouped["_measure"].agg(aggregation).rename("value")
        row_counts = grouped.size().rename("row_count")

    periods = pd.DataFrame({"value": values})
    periods["row_count"] = row_counts
    periods = periods.reset_index().rename(
        columns={"_period": "period_start"}
    )

    first_observation = working["_date"].min()

    notes: list[str] = []

    if unreadable_dates:
        notes.append(
            f"{unreadable_dates:,} row(s) were left out because their "
            f"'{date_column}' value could not be read as a date."
        )

    if unusable_measure:
        notes.append(
            f"{unusable_measure:,} row(s) were left out because their "
            f"'{measure_column}' value was not a number."
        )

    # Insert the periods the calendar expects but the data never
    # mentions, so a gap is drawn as a gap.
    zero_filled = 0
    gaps_left_empty = 0

    if fill_gaps and len(periods) > 1:
        complete_index = pd.date_range(
            start=periods["period_start"].min(),
            end=periods["period_start"].max(),
            freq=GRAIN_FREQUENCY[resolved_grain]
        )

        missing_count = len(complete_index) - len(periods)

        if missing_count > 0:
            periods = (
                periods
                .set_index("period_start")
                .reindex(complete_index)
                .rename_axis("period_start")
                .reset_index()
            )

            periods["row_count"] = (
                periods["row_count"].fillna(0).astype(int)
            )

            if aggregation in ZERO_FILLABLE:
                periods["value"] = periods["value"].fillna(0)
                zero_filled = int(missing_count)

                notes.append(
                    f"{zero_filled:,} period(s) had no rows and are "
                    "shown as zero, which is the true total for a "
                    "period with nothing in it."
                )

            else:
                gaps_left_empty = int(missing_count)

                notes.append(
                    f"{gaps_left_empty:,} period(s) had no rows and are "
                    f"left empty, because a period with nothing in it "
                    f"has no {aggregation}."
                )

    # Drop the period the data has not finished filling.
    excluded_period_label = ""
    excluded_final = False
    excluded_rows = 0

    if drop_incomplete_final and len(periods) >= 1:
        last_start = periods["period_start"].iloc[-1]

        if final_period_is_incomplete(
            last_start,
            last_observation,
            resolved_grain
        ):
            if len(periods) > 2:
                excluded_rows = int(periods["row_count"].iloc[-1])
                excluded_period_label = formatting.format_period(
                    last_start,
                    resolved_grain
                )
                periods = periods.iloc[:-1].copy()
                excluded_final = True

                notes.append(
                    f"{excluded_period_label} is still in progress and "
                    f"has been left out. It held {excluded_rows:,} "
                    "row(s) so far, which would read as a fall rather "
                    "than a partial period."
                )

            else:
                notes.append(
                    "The newest period is still in progress. It has "
                    "been kept because removing it would leave too "
                    "little history to show."
                )

    periods = periods.reset_index(drop=True)

    # How much calendar each period actually covers, clipped to the range
    # the data spans. Computed after gap filling and after the incomplete
    # final period is dropped, so every remaining row gets a real figure.
    periods["period_days"] = [
        _observed_days(
            start,
            resolved_grain,
            first_observation,
            last_observation
        )
        for start in periods["period_start"]
    ]

    if aggregation == "sum_per_day":
        periods["value"] = (
            periods["value"] / periods["period_days"]
        )

    periods["period_label"] = [
        formatting.format_period(start, resolved_grain)
        for start in periods["period_start"]
    ]

    # A total over periods of different lengths carries a saw-tooth that
    # has nothing to do with the business. February is short, not weak.
    exposure_ratio = 1.0
    unequal_exposure = False

    if len(periods) > 1:
        shortest = int(periods["period_days"].min())
        longest = int(periods["period_days"].max())

        if shortest > 0:
            exposure_ratio = longest / shortest

        unequal_exposure = (
            exposure_ratio > EXPOSURE_WARNING_RATIO
        )

    if unequal_exposure and aggregation in EXPOSURE_SENSITIVE:
        notes.append(
            f"These periods cover different numbers of days, from "
            f"{int(periods['period_days'].min())} to "
            f"{int(periods['period_days'].max())}. A total rises and "
            f"falls with the calendar as well as with the business, so "
            f"a short period can look like a decline. Switch the "
            f"aggregation to 'sum_per_day' to compare rates instead."
        )

    return {
        "success": True,
        "grain": resolved_grain,
        "aggregation": aggregation,
        "measure_column": measure_column if not counting_rows else None,
        "date_column": date_column,
        "counting_rows": counting_rows,
        "periods": periods,
        "period_count": int(len(periods)),
        "total_rows": total_rows,
        "rows_used": int(periods["row_count"].sum()),
        "unreadable_dates": unreadable_dates,
        "unusable_measure": unusable_measure,
        "zero_filled_periods": zero_filled,
        "gaps_left_empty": gaps_left_empty,
        "exposure_ratio": round(float(exposure_ratio), 4),
        "unequal_period_lengths": bool(unequal_exposure),
        "excluded_final_period": excluded_final,
        "excluded_period_label": excluded_period_label,
        "excluded_period_rows": excluded_rows,
        "first_period": (
            periods["period_start"].iloc[0]
            if len(periods)
            else None
        ),
        "last_period": (
            periods["period_start"].iloc[-1]
            if len(periods)
            else None
        ),
        "enough_for_trend": (
            int(periods["value"].notna().sum())
            >= MINIMUM_PERIODS_FOR_TREND
        ),
        "notes": notes,
    }


def usable_date_columns(
    frame: pd.DataFrame,
    minimum_success_rate: float = 0.8,
    minimum_distinct_periods: int = 3
) -> list[str]:
    """
    Find the columns that could drive a time series.

    A column qualifies when most of its values read as dates and they
    spread across enough distinct points to plot.

    Args:
        frame:
            The dataset.

        minimum_success_rate:
            Share of non-missing values that must parse as a date.

        minimum_distinct_periods:
            Distinct dates needed before a column is worth offering.

    Returns:
        Column names, in the order they appear in the dataset.
    """

    candidates = []

    for column in frame.columns:
        series = frame[column]

        if pd.api.types.is_numeric_dtype(series) and not (
            pd.api.types.is_datetime64_any_dtype(series)
        ):
            # A bare number is not treated as a date. Epoch seconds and
            # year numbers are too easily confused with measures.
            continue

        non_missing = series.dropna()

        if non_missing.empty:
            continue

        sample = non_missing.head(200)
        converted = pd.to_datetime(
            sample,
            errors="coerce",
            format="mixed"
        )

        if converted.notna().mean() < minimum_success_rate:
            continue

        if converted.dropna().nunique() < minimum_distinct_periods:
            continue

        candidates.append(str(column))

    return candidates


def usable_measure_columns(
    frame: pd.DataFrame,
    exclude: tuple[str, ...] = ()
) -> list[str]:
    """
    Find the numeric columns worth aggregating over time.

    Args:
        frame:
            The dataset.

        exclude:
            Columns to leave out, such as the chosen date column.

    Returns:
        Column names, in the order they appear in the dataset.
    """

    measures = []

    for column in frame.columns:
        if str(column) in exclude:
            continue

        series = frame[column]

        if not pd.api.types.is_numeric_dtype(series):
            continue

        if pd.api.types.is_bool_dtype(series):
            continue

        if series.dropna().nunique() <= 1:
            continue

        measures.append(str(column))

    return measures
