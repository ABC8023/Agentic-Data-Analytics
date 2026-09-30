"""
The visualisation tab.
"""


import plotly.express as px
import streamlit as st

from charts import create_plotly_figure, recommend_visualisations
from ui_cache import (
    cached_date_like_columns,
)
from ui_theme import (
    chart,
    section,
    styled,
)


def render(*, data_version, dataframe, dataset_key):

    section("Custom chart builder")

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    all_columns = dataframe.columns.tolist()

    chart_type = st.selectbox(
        "Select chart type",
        options=[
            "Histogram",
            "Box plot",
            "Bar chart",
            "Scatter plot",
            "Correlation heatmap",
            "Time-series line chart",
        ]
    )

    if chart_type == "Histogram":

        if not numeric_columns:
            st.warning(
                "This dataset has no numerical columns."
            )

        else:
            selected_column = st.selectbox(
                "Select a numerical column",
                options=numeric_columns,
                key="histogram_column"
            )

            number_of_bins = st.slider(
                "Number of bins",
                min_value=5,
                max_value=100,
                value=30
            )

            figure = px.histogram(
                dataframe,
                x=selected_column,
                nbins=number_of_bins,
                title=f"Distribution of {selected_column}"
            )

            chart(
                styled(figure),
                width="stretch",
                key="manual_histogram"
            )


    elif chart_type == "Box plot":

        if not numeric_columns:
            st.warning(
                "This dataset has no numerical columns."
            )

        else:
            selected_column = st.selectbox(
                "Select a numerical column",
                options=numeric_columns,
                key="box_column"
            )

            figure = px.box(
                dataframe,
                y=selected_column,
                points="outliers",
                title=f"Outlier analysis for {selected_column}"
            )

            chart(
                styled(figure),
                width="stretch",
                key="manual_box_plot"
            )


    elif chart_type == "Bar chart":

        categorical_columns = dataframe.select_dtypes(
        include=["object", "str", "category", "bool"]
        ).columns.tolist()

        low_cardinality_numeric_columns = [
            column
            for column in numeric_columns
            if dataframe[column].nunique(dropna=True) <= 20
        ]

        bar_chart_columns = list(dict.fromkeys(
            categorical_columns
            + low_cardinality_numeric_columns
        ))

        if not bar_chart_columns:
            st.warning(
                "No suitable categorical columns were found."
            )

        else:
            selected_column = st.selectbox(
                "Select a categorical column",
                options=bar_chart_columns,
                key=f"{dataset_key}_bar_column"
            )

            maximum_categories = st.slider(
                "Maximum categories",
                min_value=5,
                max_value=30,
                value=15,
                key=f"{dataset_key}_maximum_categories"
            )

            category_series = (
                dataframe[selected_column]
                .astype("object")
                .fillna("Missing")
                .astype(str)
            )

            category_counts = (
                category_series
                .value_counts()
                .head(maximum_categories)
                .rename_axis(selected_column)
                .reset_index(name="count")
            )

            figure = px.bar(
                category_counts,
                x=selected_column,
                y="count",
                title=f"Frequency of {selected_column}"
            )

            chart(
                styled(figure),
                width="stretch",
                key=f"{dataset_key}_manual_bar_chart"
            )

    elif chart_type == "Scatter plot":

        if len(numeric_columns) < 2:
            st.warning(
                "A scatter plot requires at least two numerical columns."
            )

        else:
            x_column = st.selectbox(
                "Select the X-axis",
                options=numeric_columns,
                key="scatter_x"
            )

            y_options = [
                column
                for column in numeric_columns
                if column != x_column
            ]

            y_column = st.selectbox(
                "Select the Y-axis",
                options=y_options,
                key="scatter_y"
            )

            figure = px.scatter(
                dataframe,
                x=x_column,
                y=y_column,
                title=f"{y_column} versus {x_column}",
                hover_data=all_columns[:5]
            )

            chart(
                styled(figure),
                width="stretch",
                key="manual_scatter_plot"
            )

    elif chart_type == "Correlation heatmap":

        if len(numeric_columns) < 2:
            st.warning(
                "A correlation heatmap requires at least two "
                "numerical columns."
            )

        else:
            selected_columns = st.multiselect(
                "Select numerical columns",
                options=numeric_columns,
                default=numeric_columns[:8],
                key="heatmap_columns"
            )

            if len(selected_columns) < 2:
                st.info(
                    "Select at least two numerical columns."
                )

            else:
                # Reuse the shared renderer so the manual builder and
                # the recommendations cannot drift apart.
                try:
                    figure = create_plotly_figure(
                        name=dataset_key,
                        chart_specification={
                            "chart_type": "correlation_heatmap",
                            "title": (
                                "Correlation between "
                                "selected columns"
                            ),
                            "x": selected_columns,
                            "y": selected_columns
                        }
                    )

                    chart(
                        styled(figure),
                        width="stretch",
                        key="manual_correlation_heatmap"
                    )

                except Exception as error:
                    st.warning(
                        f"Could not create the heatmap: {error}"
                    )

    elif chart_type == "Time-series line chart":

        date_like_columns = cached_date_like_columns(
            dataset_key,
            data_version
        )

        if not date_like_columns:
            st.warning(
                "No date or timestamp columns were detected in "
                "this dataset."
            )

        elif not numeric_columns:
            st.warning(
                "This dataset has no numerical columns to plot "
                "over time."
            )

        else:
            x_column = st.selectbox(
                "Select the date column",
                options=date_like_columns,
                key="line_x"
            )

            y_column = st.selectbox(
                "Select the numerical column",
                options=numeric_columns,
                key="line_y"
            )

            try:
                figure = create_plotly_figure(
                    name=dataset_key,
                    chart_specification={
                        "chart_type": "line",
                        "title": (
                            f"{y_column} over {x_column}"
                        ),
                        "x": x_column,
                        "y": y_column
                    }
                )

                chart(
                    styled(figure),
                    width="stretch",
                    key="manual_line_chart"
                )

                st.caption(
                    "Values are averaged per date before plotting."
                )

            except Exception as error:
                st.warning(
                    f"Could not create the line chart: {error}"
                )

    section("Recommended visualisations")

    maximum_recommendations = st.slider(
        "Maximum number of recommended charts",
        min_value=1,
        max_value=15,
        value=8,
        key=f"{dataset_key}_maximum_recommendations",
        help=(
            "Each chart type gets a turn before any type is "
            "suggested twice, so a low limit still covers a "
            "range of views."
        )
    )

    if st.button(
        "Generate recommended visualisations",
        type="primary",
        key=f"{dataset_key}_generate_recommended_visualisations"
    ):
        with st.spinner("Analysing the dataset and selecting charts..."):
            st.session_state.recommendation_result = (
                recommend_visualisations.invoke({
                    "name": dataset_key,
                    "max_recommendation": maximum_recommendations
                })
            )

    recommendation_result = st.session_state.recommendation_result

    if recommendation_result is None:
        st.info(
            "Generate recommendations to see suitable charts "
            "for this dataset."
        )

    elif not recommendation_result.get("success"):
        st.error(
            recommendation_result.get(
                "error",
                "Unable to recommend visualisations."
            )
        )

    else:
        recommendations = recommendation_result.get(
            "recommendations",
            []
        )

        if not recommendations:
            st.info(
                "No suitable visualisations were found."
            )

        else:
            for chart_number, recommendation in enumerate(
                recommendations,
                start=1
            ):
                try:
                    figure = create_plotly_figure(
                        name=dataset_key,
                        chart_specification=recommendation
                    )

                    chart(
                        styled(figure),
                        width="stretch",
                        key=(
                            f"{dataset_key}_recommended_"
                            f"chart_{chart_number}"
                        )
                    )

                    st.caption(
                        recommendation.get(
                            "reason",
                            "Recommended from the dataset structure."
                        )
                    )

                except Exception as error:
                    st.warning(
                        f"Could not create "
                        f"'{recommendation.get('title', 'chart')}': "
                        f"{error}"
                    )

