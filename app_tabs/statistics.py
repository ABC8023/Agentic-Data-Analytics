"""
The statistics tab.
"""


import pandas as pd
import streamlit as st

from ui_cache import (
    cached_statistics,
)
from ui_theme import (
    section,
    table,
)


def render(*, data_version, dataset_key):

    statistics_report = cached_statistics(
        dataset_key,
        data_version,
        5
    )

    if statistics_report.get("success"):

        numeric_statistics = statistics_report.get(
            "numeric_statistics",
            {}
        )

        if numeric_statistics:
            section("Numerical statistics")

            numeric_statistics_df = pd.DataFrame(
                numeric_statistics
            )

            table(
                numeric_statistics_df,
                width="stretch"
            )

        categorical_statistics = statistics_report.get(
            "categorical_statistics",
            {}
        )

        if categorical_statistics:
            section("Categorical statistics")

            for column_name, column_summary in categorical_statistics.items():

                with st.expander(
                    f"Column: {column_name}"
                ):
                    st.metric(
                        "Unique values",
                        column_summary["unique_values"]
                    )

                    category_df = pd.DataFrame(
                        column_summary["most_common_values"]
                    )

                    table(
                        category_df,
                        width="stretch",
                        hide_index=True
                    )

        if not numeric_statistics and not categorical_statistics:
            st.info(
                "No supported numerical or categorical columns "
                "were found."
            )

    else:
        st.error(
            statistics_report.get(
                "error",
                "Unable to calculate dataset statistics."
            )
        )

