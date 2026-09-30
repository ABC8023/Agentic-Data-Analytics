"""
The dashboard tab.
"""

import os

import pandas as pd
import streamlit as st

import aggregation
import forecasting
import formatting
import schema
import segments
from ui_cache import (
    cached_business_analysis,
    cached_business_schema,
    cached_drill_down,
)
from ui_components import (
    render_evidence_ledger,
    render_recommendations,
)
from ui_figures import (
    build_segment_heatmap,
    build_timeline_figure,
    build_waterfall_figure,
)
from ui_theme import (
    chart,
    lede,
    safe,
    section,
    styled,
    table,
)


def render(*, data_version, dataset_key, date_columns, name):

    section("What the numbers say")

    lede(
        "Every figure below is calculated from the loaded rows "
        "and shown with the arithmetic that produced it. The "
        "suggestions at the end are investigations the numbers "
        "point to, not causes: a table of totals can show what "
        "moved, never why."
    )

    # Detect first with nothing overridden, to get the candidate
    # lists and the defaults the controls open on.
    suggested = cached_business_schema(
        dataset_key,
        data_version,
        None,
        None,
        None
    )

    role_one, role_two, role_three = st.columns(3)

    date_choices = suggested["date_candidates"] or date_columns

    with role_one:
        chosen_bi_date = st.selectbox(
            "Timeline",
            options=date_choices,
            key=f"{dataset_key}_bi_date",
            help=suggested["date_reason"]
        )

    bi_count_label = "Number of rows"
    measure_choices = [
        *suggested["measure_candidates"],
        bi_count_label,
    ]

    with role_two:
        chosen_bi_measure_label = st.selectbox(
            "Measure",
            options=measure_choices,
            key=f"{dataset_key}_bi_measure",
            help=(
                suggested["measure_reason"]
                or "No column reads as a measure."
            )
        )

    bi_counting_rows = (
        chosen_bi_measure_label == bi_count_label
    )
    chosen_bi_measure = (
        None if bi_counting_rows else chosen_bi_measure_label
    )

    no_segment_label = "No breakdown"
    segment_choices = [
        *suggested["segment_candidates"],
        no_segment_label,
    ]

    with role_three:
        chosen_segment_label = st.selectbox(
            "Break down by",
            options=segment_choices,
            key=f"{dataset_key}_bi_segment",
            help=(
                suggested["segment_reason"]
                or "No column reads as a segment."
            )
        )

    chosen_segment = (
        None
        if chosen_segment_label == no_segment_label
        else chosen_segment_label
    )

    option_one, option_two, option_three = st.columns(3)

    with option_one:
        if bi_counting_rows:
            chosen_bi_aggregation = "count"

            st.selectbox(
                "Aggregation",
                options=["count"],
                key=f"{dataset_key}_bi_aggregation_count",
                disabled=True,
                help=(
                    "Rows are counted, so there is nothing to "
                    "choose."
                )
            )

        else:
            chosen_bi_aggregation = st.selectbox(
                "Aggregation",
                options=[
                    name
                    for name in aggregation.AGGREGATIONS
                    if name != "count"
                ],
                key=f"{dataset_key}_bi_aggregation"
            )

    with option_two:
        bi_grain_choice = st.selectbox(
            "Period",
            options=["Automatic", *aggregation.PERIOD_GRAINS],
            key=f"{dataset_key}_bi_grain",
            help=(
                "Automatic picks a period from how much calendar "
                "the data covers."
            )
        )

    with option_three:
        chosen_bi_horizon = st.slider(
            "Periods to project",
            min_value=1,
            max_value=12,
            value=forecasting.DEFAULT_HORIZON,
            key=f"{dataset_key}_bi_horizon"
        )

    requested_bi_grain = (
        None
        if bi_grain_choice == "Automatic"
        else bi_grain_choice
    )

    resolved_schema = cached_business_schema(
        dataset_key,
        data_version,
        chosen_bi_measure,
        chosen_bi_date,
        chosen_segment
    )

    schema_summary = schema.describe_business_schema(
        resolved_schema
    )

    with st.spinner("Working through the data..."):
        analysis = cached_business_analysis(
            dataset_key,
            data_version,
            chosen_bi_measure,
            chosen_bi_date,
            chosen_segment,
            chosen_bi_aggregation,
            requested_bi_grain,
            chosen_bi_horizon,
            schema_summary
        )

    if not analysis.get("success"):
        st.error(analysis.get(
            "error",
            "The analysis could not be built."
        ))

    else:
        bi_periods = analysis["periods"]
        bi_grain = analysis["grain"]
        bi_measure_label = analysis["measure_label"]

        st.markdown(
            f"<div class='headline'>{safe(analysis['headline'])}"
            f"<span class='sub'>{safe(schema_summary)}</span></div>",
            unsafe_allow_html=True
        )

        kpi_columns = st.columns(len(analysis["kpis"]))

        for column, kpi in zip(
            kpi_columns,
            analysis["kpis"],
            strict=True
        ):
            with column:
                st.metric(
                    kpi["label"],
                    kpi["value"],
                    help=kpi["note"]
                )

                st.caption(kpi["note"])

        st.divider()

        section("Evidence")

        lede(
            "Each finding is shown with the calculation behind "
            "it, so a figure can be checked rather than trusted."
        )

        render_evidence_ledger(
            analysis["evidence"],
            f"{dataset_key}_evidence"
        )

        st.divider()

        section("What to look at next")

        render_recommendations(
            analysis["recommendations"],
            analysis["evidence"]
        )

        st.divider()

        section("Movement over time")

        timeline_figure = build_timeline_figure(
            periods=bi_periods,
            grain=bi_grain,
            trend=analysis["trend"],
            detected=analysis["anomalies"],
            projection=analysis["forecast"],
            measure_label=bi_measure_label
        )

        chart(
            styled(timeline_figure),
            width="stretch",
            key=f"{dataset_key}_bi_timeline"
        )

        for note in analysis["notes"]:
            st.caption(note)

        verification = analysis["backtest"]

        if verification.get("success"):
            if verification["beat_naive"]:
                st.success(verification["verdict"])

            else:
                st.warning(verification["verdict"])

        if chosen_segment is None:
            st.info(
                "Choose a column under 'Break down by' to see "
                "which parts of the business moved."
            )

        elif analysis["segment_matrix"] is None or not (
            analysis["segment_matrix"].get("success")
        ):
            st.warning(
                analysis["segment_error"]
                or "The breakdown could not be built."
            )

        else:
            matrix_result = analysis["segment_matrix"]
            segment_matrix = matrix_result["matrix"]
            breakdown = analysis["segment_breakdown"]
            movement = analysis["segment_movement"]

            st.divider()

            section(f"What moved, by {chosen_segment}")

            if movement is None:
                st.info(
                    "There is only one period, so there is no "
                    "change to account for."
                )

            else:
                chart(
                    styled(build_waterfall_figure(
                        movement,
                        bi_measure_label
                    )),
                    width="stretch",
                    key=f"{dataset_key}_bi_waterfall"
                )

                if movement["reconciles"]:
                    st.caption(
                        "The bars add exactly to the change "
                        "between the two totals, so nothing is "
                        "unaccounted for."
                    )

                else:
                    st.warning(
                        f"The parts account for "
                        f"{formatting.format_signed(movement['accounted_change'])}"
                        f" of a "
                        f"{formatting.format_signed(movement['total_change'])}"
                        f" change, leaving "
                        f"{formatting.format_signed(movement['residual'])}"
                        " unexplained. Read the bars as "
                        "indicative rather than as a "
                        "decomposition."
                    )

                table(
                    pd.DataFrame({
                        "Part": movement["table"]["segment"],
                        movement["previous_label"]: (
                            movement["table"]["previous"]
                            .round(3)
                        ),
                        movement["latest_label"]: (
                            movement["table"]["latest"].round(3)
                        ),
                        "Change": (
                            movement["table"]["change"].round(3)
                        ),
                        "Share of change": (
                            movement["table"]["contribution"]
                        ),
                    }),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Share of change": (
                            st.column_config.NumberColumn(
                                format="percent",
                                help=(
                                    "Each part's change divided "
                                    "by the overall change. Can "
                                    "exceed 100% when parts move "
                                    "in opposite directions."
                                )
                            )
                        ),
                    }
                )

            st.divider()

            section(f"Parts of {chosen_segment} over time")

            share_view = st.toggle(
                "Show each period as shares",
                value=matrix_result["additive"],
                key=f"{dataset_key}_bi_shares",
                disabled=not matrix_result["additive"],
                help=(
                    "Shares show the mix moving. Raw values "
                    "mostly show which periods were large."
                )
            )

            chart(
                styled(build_segment_heatmap(
                    segment_matrix,
                    as_shares=(
                        share_view and matrix_result["additive"]
                    ),
                    grain=bi_grain,
                    measure_label=bi_measure_label
                )),
                width="stretch",
                key=f"{dataset_key}_bi_heatmap"
            )

            st.caption(matrix_result["reconciliation_note"])

            if segment_matrix.shape[1] > segments.MAX_MATRIX_SEGMENTS:
                st.caption(
                    f"'{chosen_segment}' has "
                    f"{segment_matrix.shape[1]:,} parts. The "
                    f"largest {segments.MAX_MATRIX_SEGMENTS} are "
                    "shown separately and the rest are summed "
                    "into one row."
                )

            breakdown_table = breakdown["table"].copy()
            breakdown_table = breakdown_table.rename(columns={
                "segment": "Part",
                "total": bi_measure_label,
                "share": "Share",
                "cumulative_share": "Running share",
                "rank": "Rank",
            })

            if not breakdown["shares_meaningful"]:
                breakdown_table = breakdown_table.drop(
                    columns=["Share", "Running share"]
                )

                st.caption(
                    breakdown["shares_withheld_because"]
                )

            table(
                breakdown_table,
                width="stretch",
                hide_index=True,
                column_config={
                    "Share": st.column_config.NumberColumn(
                        format="percent"
                    ),
                    "Running share": (
                        st.column_config.NumberColumn(
                            format="percent"
                        )
                    ),
                }
            )

            inner_options = [
                name
                for name in resolved_schema["segment_candidates"]
                if name != chosen_segment
            ]

            if inner_options and breakdown["part_count"]:
                st.divider()

                section("Look inside one part")

                lede(
                    "Pick a part and a second dimension to "
                    "regroup it by. These figures cover only the "
                    "rows in that part, so they add up to the "
                    "part rather than to the headline total."
                )

                inside_one, inside_two = st.columns(2)

                with inside_one:
                    chosen_part = st.selectbox(
                        f"Part of {chosen_segment}",
                        options=list(
                            breakdown["table"]["segment"]
                        ),
                        key=f"{dataset_key}_bi_part"
                    )

                with inside_two:
                    chosen_inner = st.selectbox(
                        "Regroup by",
                        options=inner_options,
                        key=f"{dataset_key}_bi_inner"
                    )

                inner = cached_drill_down(
                    dataset_key,
                    data_version,
                    chosen_segment,
                    chosen_part,
                    chosen_inner,
                    chosen_bi_date,
                    chosen_bi_measure,
                    chosen_bi_aggregation,
                    bi_grain,
                    tuple(
                        str(value)
                        for value in bi_periods["period_start"]
                    ),
                    tuple(
                        float(value)
                        for value in bi_periods["value"]
                    ),
                    tuple(
                        int(value)
                        for value in bi_periods["period_days"]
                    )
                )

                if not inner.get("success"):
                    st.warning(inner.get(
                        "error",
                        "This part could not be broken down."
                    ))

                else:
                    st.caption(
                        f"'{chosen_part}' holds "
                        f"{formatting.pluralise(inner['parent_rows'], 'row')}"
                        f", {formatting.format_percent(inner['parent_row_share'])}"
                        " of the dataset."
                    )

                    chart(
                        styled(build_segment_heatmap(
                            inner["matrix"],
                            as_shares=False,
                            grain=bi_grain,
                            measure_label=bi_measure_label
                        )),
                        width="stretch",
                        key=f"{dataset_key}_bi_inner_heatmap"
                    )

                    inner_breakdown = segments.segment_breakdown(
                        inner["matrix"],
                        additive=inner["additive"]
                    )

                    table(
                        pd.DataFrame({
                            "Part": (
                                inner_breakdown["table"]
                                ["segment"]
                            ),
                            bi_measure_label: (
                                inner_breakdown["table"]["total"]
                                .round(3)
                            ),
                        }),
                        width="stretch",
                        hide_index=True
                    )

                    st.caption(inner["reconciliation_note"])

        st.divider()

        section("Take it with you")

        st.download_button(
            "Download the brief",
            data=analysis["brief"],
            file_name=(
                f"{os.path.splitext(name)[0]}-brief.md"
            ),
            mime="text/markdown",
            key=f"{dataset_key}_bi_brief"
        )

        with st.expander("How the columns were read"):
            st.write(schema_summary)

            for reason in (
                resolved_schema["measure_reason"],
                resolved_schema["date_reason"],
                resolved_schema["segment_reason"],
            ):
                if reason:
                    st.caption(reason)

            for note in resolved_schema["notes"]:
                st.caption(note)

            if resolved_schema["identifier_columns"]:
                st.caption(
                    "Treated as identifiers and kept out of the "
                    "analysis: "
                    + ", ".join(
                        resolved_schema["identifier_columns"]
                    )
                    + "."
                )

