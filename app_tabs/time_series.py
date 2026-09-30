"""
The time series tab.
"""


import pandas as pd
import streamlit as st

import aggregation
import anomalies
import forecasting
import formatting
import timeseries
from ui_cache import (
    cached_measure_columns,
    cached_time_series,
)
from ui_figures import (
    build_timeline_figure,
)
from ui_theme import (
    chart,
    lede,
    section,
    styled,
    table,
)


def render(*, data_version, dataset_key, date_columns):

    section("Movement over time")

    lede(
        "Choose a date and a measure. The chart shows the trend, "
        "marks periods that fall outside the expected band, and "
        "projects the next few periods with a test of whether "
        "that projection is worth anything."
    )

    control_one, control_two, control_three = st.columns(3)

    with control_one:
        chosen_date_column = st.selectbox(
            "Date column",
            options=date_columns,
            key=f"{dataset_key}_ts_date"
        )

    measure_options = cached_measure_columns(
        dataset_key,
        data_version,
        (chosen_date_column,)
    )

    # A measure comes first so the tab opens on something worth
    # looking at. Counting rows stays available, and is the only
    # option for a dataset with dates but no numeric column.
    count_label = "Number of rows"
    measure_choices = [*measure_options, count_label]

    with control_two:
        chosen_measure_label = st.selectbox(
            "Measure",
            options=measure_choices,
            key=f"{dataset_key}_ts_measure"
        )

    counting_rows = chosen_measure_label == count_label
    chosen_measure = (
        None if counting_rows else chosen_measure_label
    )

    with control_three:
        if counting_rows:
            chosen_aggregation = "count"

            st.selectbox(
                "Aggregation",
                options=["count"],
                key=f"{dataset_key}_ts_aggregation_count",
                disabled=True,
                help=(
                    "Rows are counted, so there is nothing to "
                    "choose."
                )
            )

        else:
            chosen_aggregation = st.selectbox(
                "Aggregation",
                options=[
                    name
                    for name in aggregation.AGGREGATIONS
                    if name != "count"
                ],
                key=f"{dataset_key}_ts_aggregation"
            )

    grain_one, grain_two = st.columns(2)

    with grain_one:
        grain_choice = st.selectbox(
            "Period",
            options=["Automatic", *aggregation.PERIOD_GRAINS],
            key=f"{dataset_key}_ts_grain",
            help=(
                "Automatic picks a period from how much calendar "
                "the data covers."
            )
        )

    with grain_two:
        chosen_horizon = st.slider(
            "Periods to project",
            min_value=1,
            max_value=12,
            value=forecasting.DEFAULT_HORIZON,
            key=f"{dataset_key}_ts_horizon"
        )

    requested_grain = (
        None if grain_choice == "Automatic" else grain_choice
    )

    with st.spinner("Building the timeline..."):
        analysis = cached_time_series(
            dataset_key,
            data_version,
            chosen_date_column,
            chosen_measure,
            chosen_aggregation,
            requested_grain,
            chosen_horizon
        )

    aggregated = analysis["aggregated"]

    if not aggregated.get("success"):
        st.error(aggregated.get(
            "error",
            "The timeline could not be built."
        ))

    else:
        periods = aggregated["periods"]
        resolved_grain = aggregated["grain"]
        trend = analysis["trend"]
        detected = analysis["anomalies"]
        projection = analysis["forecast"]
        verification = analysis["backtest"]

        summary_one, summary_two, summary_three, summary_four = (
            st.columns(4)
        )

        with summary_one:
            st.metric(
                "Periods",
                f"{aggregated['period_count']:,}",
                help=formatting.format_period_range(
                    aggregated["first_period"],
                    aggregated["last_period"],
                    resolved_grain
                )
            )

        with summary_two:
            st.metric(
                "Latest period",
                formatting.format_compact(
                    periods["value"].iloc[-1]
                ),
                help=periods["period_label"].iloc[-1]
            )

        with summary_three:
            if len(periods) >= 2:
                latest = periods["value"].iloc[-1]
                previous = periods["value"].iloc[-2]

                change = (
                    None
                    if not (
                        pd.notna(latest) and pd.notna(previous)
                    )
                    else latest - previous
                )

                st.metric(
                    "Change on previous",
                    formatting.format_signed(change),
                    help=(
                        "The most recent complete period "
                        "against the one before it."
                    )
                )

            else:
                st.metric("Change on previous", "not available")

        with summary_four:
            st.metric(
                "Direction",
                (
                    trend.get("direction", "unknown").capitalize()
                    if trend.get("success")
                    else "not available"
                ),
                help=timeseries.describe_trend(
                    trend,
                    resolved_grain
                )
            )

        figure = build_timeline_figure(
            periods=periods,
            grain=resolved_grain,
            trend=trend,
            detected=detected,
            projection=projection,
            measure_label=(
                "Rows" if counting_rows else chosen_measure_label
            )
        )

        chart(
            styled(figure),
            width="stretch",
            key=f"{dataset_key}_timeline_chart"
        )

        # Every adjustment made to the timeline is stated, so a
        # reader can tell a real fall from a bookkeeping artefact.
        for note in aggregated["notes"]:
            st.caption(note)

        if trend.get("success"):
            st.caption(
                timeseries.describe_trend(trend, resolved_grain)
            )

        st.divider()

        section("Unusual periods")

        if not detected.get("success"):
            st.info(detected.get(
                "error",
                "No anomaly test was possible."
            ))

        else:
            st.write(anomalies.describe_anomalies(detected))

            if detected["anomaly_count"]:
                table(
                    pd.DataFrame([
                        {
                            "Period": entry["period_label"],
                            "Value": entry["value"],
                            "Expected": round(
                                entry["expected"], 3
                            ),
                            "Difference": round(
                                entry["deviation"], 3
                            ),
                            "Direction": entry["direction"],
                            "Deviations from trend": round(
                                abs(entry["score"]), 2
                            ),
                        }
                        for entry in detected["anomalies"]
                    ]),
                    width="stretch",
                    hide_index=True
                )

            for note in detected["notes"]:
                st.caption(note)

            st.caption(
                f"Band width: "
                f"{detected['multiplier']:.2f} deviations. "
                + anomalies.describe_false_alarm_rate()
            )

        st.divider()

        section("Projection")

        if not projection.get("success"):
            st.info(projection.get(
                "error",
                "No projection was possible."
            ))

        else:
            if verification.get("success"):
                if verification["beat_naive"]:
                    st.success(verification["verdict"])

                else:
                    # The most important sentence on the tab.
                    st.warning(verification["verdict"])

                st.caption(
                    f"Tested over "
                    f"{formatting.pluralise(verification['fold_count'], 'run')}"
                    f", each projecting "
                    f"{formatting.pluralise(verification['horizon'], 'period')}"
                    f" it had not seen. Average error "
                    f"{formatting.format_number(verification['model_error'])} "
                    f"against "
                    f"{formatting.format_number(verification['naive_error'])} "
                    f"for assuming no change."
                )

            else:
                st.info(verification.get(
                    "error",
                    "The projection could not be tested."
                ))

            table(
                pd.DataFrame({
                    "Period": projection["forecast"][
                        "period_label"
                    ],
                    "Projection": projection["forecast"][
                        "forecast"
                    ].round(3),
                    "Low": projection["forecast"][
                        "lower"
                    ].round(3),
                    "High": projection["forecast"][
                        "upper"
                    ].round(3),
                }),
                width="stretch",
                hide_index=True
            )

            if projection["seasonality_applied"]:
                st.caption(
                    "A month-of-year adjustment was applied, "
                    "estimated from the repeated calendar months "
                    "in the history."
                )

            for note in projection["notes"]:
                st.caption(note)

