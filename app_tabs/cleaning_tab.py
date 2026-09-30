"""
The cleaning tab.
"""


import streamlit as st

from cleaning import clean_dataset
from dataset_store import (
    register_dataset,
)
from ui_cache import (
    cached_cleaning_metrics,
)
from ui_theme import (
    lede,
    section,
)


def render(*, data_version, dataframe, dataset_key, name):

    section("Data cleaning")

    lede(
        "Configure cleaning operations, preview the result, "
        "and apply the cleaned dataset to the application."
    )

    cleaning_metrics = cached_cleaning_metrics(
        dataset_key,
        data_version
    )

    cleaning_metric1, cleaning_metric2, cleaning_metric3 = (
        st.columns(3)
    )

    with cleaning_metric1:
        st.metric(
            "Current rows",
            cleaning_metrics["rows"]
        )

    with cleaning_metric2:
        st.metric(
            "Missing values",
            cleaning_metrics["missing_values"]
        )

    with cleaning_metric3:
        st.metric(
            "Duplicate rows",
            cleaning_metrics["duplicate_rows"]
        )

    st.divider()

    remove_duplicates = st.checkbox(
        "Remove duplicate rows",
        value=True,
        key="clean_remove_duplicates"
    )

    columns_to_drop = st.multiselect(
        "Columns to remove",
        options=dataframe.columns.tolist(),
        key="clean_columns_to_drop"
    )

    numerical_strategy_labels = {
        "Leave missing values unchanged": "leave",
        "Fill using median": "median",
        "Fill using mean": "mean",
        "Fill using zero": "zero",
        "Remove rows containing missing numerical values": (
            "drop_rows"
        )
    }

    selected_numerical_strategy = st.selectbox(
        "Numerical missing-value strategy",
        options=list(
            numerical_strategy_labels.keys()
        ),
        index=1,
        key="clean_numeric_strategy"
    )

    numeric_missing_strategy = (
        numerical_strategy_labels[
            selected_numerical_strategy
        ]
    )

    categorical_strategy_labels = {
        "Leave missing values unchanged": "leave",
        "Fill using most frequent value": (
            "most_frequent"
        ),
        "Fill using the label 'Missing'": (
            "missing_label"
        ),
        "Remove rows containing missing categorical values": (
            "drop_rows"
        )
    }

    selected_categorical_strategy = st.selectbox(
        "Categorical missing-value strategy",
        options=list(
            categorical_strategy_labels.keys()
        ),
        index=1,
        key="clean_categorical_strategy"
    )

    categorical_missing_strategy = (
        categorical_strategy_labels[
            selected_categorical_strategy
        ]
    )

    preview_column, restore_column = st.columns(2)

    with preview_column:

        if st.button(
            "Preview cleaned dataset",
            type="primary",
            key="preview_cleaned_dataset"
        ):
            with st.spinner(
                "Applying cleaning operations..."
            ):
                cleaning_result = clean_dataset(
                    name=dataset_key,
                    remove_duplicates=remove_duplicates,
                    drop_columns=columns_to_drop,
                    numeric_missing_strategy=(
                        numeric_missing_strategy
                    ),
                    categorical_missing_strategy=(
                        categorical_missing_strategy
                    )
                )

                st.session_state.cleaning_result = (
                    cleaning_result
                )

    with restore_column:

        restore_original = st.button(
            "Restore original uploaded dataset",
            key="restore_original_dataset"
        )

        if restore_original:

            original_dataframe = (
                st.session_state
                .original_dataframe
            )

            if original_dataframe is None:
                st.error(
                    "The original dataset is not available."
                )

            else:
                restored_dataframe = (
                    original_dataframe.copy()
                )

                st.session_state.dataframe = (
                    restored_dataframe
                )

                register_dataset(
                    dataset_key,
                    restored_dataframe
                )

                st.session_state.data_version += 1

                st.session_state.cleaning_result = None
                st.session_state.recommendation_result = None
                st.session_state.ml_result = None
                st.session_state.ml_target = None
                st.session_state.ml_task = None
                st.session_state.last_ml_target = None
                st.session_state.feature_importance_result = None
                st.session_state.agent_chat_messages = []

                st.rerun()

    cleaning_result = (
        st.session_state.cleaning_result
    )

    if cleaning_result is None:

        st.info(
            "Select the cleaning options and click "
            "'Preview cleaned dataset'."
        )

    elif not cleaning_result.get("success"):

        st.error(
            cleaning_result.get(
                "error",
                "The dataset could not be cleaned."
            )
        )

    else:

        cleaned_dataframe = cleaning_result[
            "cleaned_dataframe"
        ]

        st.divider()

        section("Cleaning summary")

        summary_column1, summary_column2, summary_column3 = (
            st.columns(3)
        )

        with summary_column1:
            st.metric(
                "Rows",
                cleaning_result["cleaned_rows"],
                delta=(
                    cleaning_result["cleaned_rows"]
                    - cleaning_result["original_rows"]
                )
            )

        with summary_column2:
            st.metric(
                "Columns",
                cleaning_result["cleaned_columns"],
                delta=(
                    cleaning_result["cleaned_columns"]
                    - cleaning_result["original_columns"]
                )
            )

        with summary_column3:
            st.metric(
                "Remaining missing values",
                cleaning_result[
                    "remaining_missing_values"
                ],
                delta=(
                    cleaning_result[
                        "remaining_missing_values"
                    ]
                    - cleaning_result[
                        "original_missing_values"
                    ]
                )
            )

        st.write(
            f"Duplicate rows removed: "
            f"**{cleaning_result['duplicates_removed']}**"
        )

        st.write(
            f"Rows removed because of missing values: "
            f"**{cleaning_result['rows_removed_for_missing_values']}**"
        )

        if cleaning_result["columns_removed"]:
            st.write(
                "Columns removed: "
                f"**{', '.join(cleaning_result['columns_removed'])}**"
            )

        for warning in cleaning_result.get(
            "warnings",
            []
        ):
            st.warning(warning)

        section("Cleaned dataset preview")

        st.dataframe(
            cleaned_dataframe.head(50),
            width="stretch",
            hide_index=True
        )

        cleaned_csv = (
            cleaned_dataframe
            .to_csv(index=False)
            .encode("utf-8")
        )

        action_column1, action_column2 = (
            st.columns(2)
        )

        with action_column1:

            st.download_button(
                label="Download cleaned CSV",
                data=cleaned_csv,
                file_name=(
                    f"cleaned_{name}"
                ),
                mime="text/csv",
                key="download_cleaned_dataset"
            )

        with action_column2:

            if st.button(
                "Use cleaned dataset for analysis",
                type="primary",
                key="apply_cleaned_dataset"
            ):

                applied_dataframe = (
                    cleaned_dataframe.copy()
                )

                st.session_state.dataframe = (
                    applied_dataframe
                )

                register_dataset(
                    dataset_key,
                    applied_dataframe
                )

                st.session_state.data_version += 1

                st.session_state.cleaning_result = None
                st.session_state.recommendation_result = None
                st.session_state.ml_result = None
                st.session_state.ml_target = None
                st.session_state.ml_task = None
                st.session_state.last_ml_target = None
                st.session_state.feature_importance_result = None
                st.session_state.agent_chat_messages = []

                st.rerun()

