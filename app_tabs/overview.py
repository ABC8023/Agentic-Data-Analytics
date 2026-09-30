"""
The overview tab.
"""


import streamlit as st

from ui_cache import (
    cached_column_information,
    cached_overview_metrics,
    cached_quality_report,
)
from ui_theme import (
    section,
    table,
)


def render(*, data_version, dataframe, dataset_key):

    overview_metrics = cached_overview_metrics(
        dataset_key,
        data_version
    )

    column1, column2, column3 = st.columns(3)

    with column1:
        st.metric(
            label="Rows",
            value=f"{overview_metrics['rows']:,}"
        )

    with column2:
        st.metric(
            label="Columns",
            value=overview_metrics["columns"]
        )

    with column3:
        st.metric(
            label="Memory usage",
            value=f"{overview_metrics['memory_mb']:.2f} MB"
        )

    quality_report = cached_quality_report(
        dataset_key,
        data_version
    )

    if quality_report.get("success"):
        section("Data-quality report")

        quality_column1, quality_column2, quality_column3 = (
            st.columns(3)
        )

        with quality_column1:
            st.metric(
                "Missing values",
                quality_report["total_missing_values"]
            )

        with quality_column2:
            st.metric(
                "Duplicate rows",
                quality_report["duplicate_rows"]
            )

        with quality_column3:
            st.metric(
                "Constant columns",
                len(
                    quality_report["constant_columns"]
                )
            )

    else:
        st.error(
            quality_report.get(
                "error",
                "Unable to generate the quality report."
            )
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    maximum_preview_rows = min(
                50,
                len(dataframe)
            )

    default_preview_rows = min(
                10,
                maximum_preview_rows
            )

    preview_rows = st.slider(
                "Number of rows to display",
                min_value=1,
                max_value=maximum_preview_rows,
                value=default_preview_rows
            )

    st.dataframe(
                dataframe.head(preview_rows),
                width="stretch",
                hide_index=True
            )

    section("Column information")

    column_information = cached_column_information(
        dataset_key,
        data_version
    )

    table(
        column_information,
        width="stretch",
        hide_index=True
                        )

