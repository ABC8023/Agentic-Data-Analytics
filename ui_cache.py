"""
Memoised profiling and analysis for the Streamlit reruns.
"""


import pandas as pd
import streamlit as st

import aggregation
import anomalies
import business_insights
import forecasting
import schema
import segments
import timeseries
from dataset_store import (
    get_or_load_dataframe,
)
from ml import (
    analyse_tool,
)
from profiling import data_time_column, get_data_quality_report, stat_overview

# ──────────────────────────────────────────────────────────────
#  MEMOISED PROFILING
#
#  Streamlit reruns this whole script on every widget interaction, so
#  without memoisation the dataset was fully re-profiled each time a
#  slider moved. These wrappers are keyed on the session-scoped dataset
#  handle plus a version counter that is bumped whenever the active
#  dataset changes, so results are reused until the data really differs.
#  The handle carries the session token, so no session can read another
#  session's cached results.
# ──────────────────────────────────────────────────────────────

PROFILE_CACHE_ENTRIES = 64


@st.cache_data(show_spinner=False, max_entries=PROFILE_CACHE_ENTRIES)
def cached_overview_metrics(dataset_key, data_version):
    """Row count, column count and deep memory usage in megabytes."""
    frame = get_or_load_dataframe(dataset_key)

    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "memory_mb": float(
            frame.memory_usage(deep=True).sum()
            / (1024 ** 2)
        )
    }


@st.cache_data(show_spinner=False, max_entries=PROFILE_CACHE_ENTRIES)
def cached_quality_report(dataset_key, data_version):
    """Missing values, duplicates and constant columns."""
    return get_data_quality_report.invoke({
        "name": dataset_key
    })


@st.cache_data(show_spinner=False, max_entries=PROFILE_CACHE_ENTRIES)
def cached_column_information(dataset_key, data_version):
    """Per-column dtype, missing count and distinct count."""
    frame = get_or_load_dataframe(dataset_key)

    return pd.DataFrame({
        "Column": frame.columns,
        "Data type": frame.dtypes.astype(str).values,
        "Missing values": frame.isnull().sum().values,
        "Unique values": [
            frame[column].nunique(dropna=True)
            for column in frame.columns
        ]
    })


@st.cache_data(show_spinner=False, max_entries=PROFILE_CACHE_ENTRIES)
def cached_statistics(dataset_key, data_version, top_categories):
    """
    Numerical describe output and categorical frequency summaries.

    Calls the plain function rather than the agent tool, so the interface
    still shows the most common values of every column. The masking that
    protects identifier columns applies only to what the model receives.
    """
    return stat_overview(
        name=dataset_key,
        n_cate=top_categories,
        mask_identifier_labels=False
    )


@st.cache_data(show_spinner=False, max_entries=PROFILE_CACHE_ENTRIES)
def cached_cleaning_metrics(dataset_key, data_version):
    """Row count, total missing values and duplicate row count."""
    frame = get_or_load_dataframe(dataset_key)

    return {
        "rows": int(len(frame)),
        "missing_values": int(
            frame.isnull().sum().sum()
        ),
        "duplicate_rows": int(
            frame.duplicated().sum()
        )
    }


@st.cache_data(show_spinner=False, max_entries=PROFILE_CACHE_ENTRIES)
def cached_target_analysis(
    dataset_key,
    data_version,
    target_column
):
    """Target column profile and classification/regression recommendation."""
    return analyse_tool(
        name=dataset_key,
        target_column=target_column
    )


@st.cache_data(show_spinner=False, max_entries=PROFILE_CACHE_ENTRIES)
def cached_date_like_columns(dataset_key, data_version):
    """Columns that parse as dates, used by the time-series chart."""
    frame = get_or_load_dataframe(dataset_key)

    return data_time_column(frame)


@st.cache_data(show_spinner=False, max_entries=PROFILE_CACHE_ENTRIES)
def cached_date_columns(dataset_key, data_version):
    """Columns that could drive a timeline, in dataset order."""
    frame = get_or_load_dataframe(dataset_key)

    return aggregation.usable_date_columns(frame)


@st.cache_data(show_spinner=False, max_entries=PROFILE_CACHE_ENTRIES)
def cached_measure_columns(dataset_key, data_version, exclude):
    """Numeric columns worth aggregating over time."""
    frame = get_or_load_dataframe(dataset_key)

    return aggregation.usable_measure_columns(frame, exclude=exclude)


@st.cache_data(show_spinner=False, max_entries=PROFILE_CACHE_ENTRIES)
def cached_time_series(
    dataset_key,
    data_version,
    date_column,
    measure_column,
    aggregation_name,
    grain,
    horizon
):
    """
    Build the whole timeline analysis in one memoised step.

    Aggregation, trend, anomaly detection, projection and backtest are
    computed together because each depends on the one before, and because
    Streamlit reruns the script on every widget interaction.

    Returns:
        A dictionary holding the period series and each analysis, or the
        aggregation failure when the series could not be built.
    """

    frame = get_or_load_dataframe(dataset_key)

    aggregated = aggregation.aggregate_periods(
        frame,
        date_column=date_column,
        measure_column=measure_column,
        aggregation=aggregation_name,
        grain=grain
    )

    if not aggregated.get("success"):
        return {"aggregated": aggregated}

    periods = aggregated["periods"]
    resolved_grain = aggregated["grain"]

    trend = timeseries.fit_trend(
        periods["period_start"],
        periods["value"]
    )

    detected = anomalies.detect_anomalies(
        periods,
        grain=resolved_grain,
        trend=trend if trend.get("success") else None
    )

    projection = forecasting.baseline_forecast(
        periods,
        grain=resolved_grain,
        horizon=horizon
    )

    verification = forecasting.backtest(
        periods,
        grain=resolved_grain,
        horizon=horizon
    )

    return {
        "aggregated": aggregated,
        "trend": trend,
        "anomalies": detected,
        "forecast": projection,
        "backtest": verification,
    }



@st.cache_data(show_spinner=False, max_entries=PROFILE_CACHE_ENTRIES)
def cached_business_schema(
    dataset_key,
    data_version,
    measure,
    date,
    segment
):
    """
    Work out which column is the measure, the date and the segment.

    The three overrides are part of the cache key, so changing one in the
    interface recomputes without discarding the other combinations a
    reader has already looked at.
    """

    frame = get_or_load_dataframe(dataset_key)

    return schema.detect_business_schema(
        frame,
        measure=measure,
        date=date,
        segment=segment
    )


@st.cache_data(show_spinner=False, max_entries=PROFILE_CACHE_ENTRIES)
def cached_business_analysis(
    dataset_key,
    data_version,
    measure,
    date,
    segment,
    aggregation_name,
    grain,
    horizon,
    schema_summary
):
    """
    Run the whole deterministic analysis in one memoised step.

    Streamlit reruns the script on every widget interaction, and this call
    aggregates, fits a trend, detects anomalies, projects, backtests and
    breaks the total down by segment. Recomputing all of it because
    somebody opened an expander would make the tab unusable.
    """

    frame = get_or_load_dataframe(dataset_key)

    return business_insights.analyse_business(
        frame,
        measure=measure,
        date=date,
        segment=segment,
        aggregation_name=aggregation_name,
        grain=grain,
        horizon=horizon,
        schema_summary=schema_summary
    )


@st.cache_data(show_spinner=False, max_entries=PROFILE_CACHE_ENTRIES)
def cached_drill_down(
    dataset_key,
    data_version,
    segment,
    value,
    next_segment,
    date,
    measure,
    aggregation_name,
    grain,
    period_starts,
    period_values,
    period_days
):
    """
    Break one part of a segment down by another dimension.

    The period series is passed as tuples rather than a frame because
    Streamlit hashes every argument, and a DataFrame argument would be
    rehashed in full on each rerun.
    """

    frame = get_or_load_dataframe(dataset_key)

    periods = pd.DataFrame({
        "period_start": pd.to_datetime(list(period_starts)),
        "value": list(period_values),
        "period_days": list(period_days),
    })

    return segments.drill_down(
        frame,
        segment=segment,
        value=value,
        next_segment=next_segment,
        date=date,
        periods=periods,
        measure=measure,
        aggregation_name=aggregation_name,
        grain=grain
    )


