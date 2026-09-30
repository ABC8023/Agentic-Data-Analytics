"""
Chart recommendations and Plotly figure construction.
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.tools import tool

from dataset_store import get_or_load_dataframe
from privacy import model_safe
from profiling import data_time_column


@tool
@model_safe
def recommend_visualisations(name : str , max_recommendation :int =8)-> dict[str,Any]:
    """
    Recommend suitable visualisations for a CSV dataset.

    Use this tool when the user asks which charts would be useful,
    requests visual analysis, or asks how the dataset should be explored.

    The tool recommends charts but does not create the final Plotly figures.

    Args:
        name:
            Name or path of the CSV dataset.

        max_recommendation:
            Maximum number of chart recommendations to return.
            Must be between 1 and 15.

    Returns:
        A structured report containing suitable chart specifications,
        relevant columns, titles and explanations.
    """

    if max_recommendation < 1 or max_recommendation > 15:
        return{
            "success":False,
            "error":(
                "maximum recommendation must be between 1 and 15"
            )

        }
    try:
             df=get_or_load_dataframe(name)
    except(ValueError , FileNotFoundError) as error:
             return{
                 "success" : False,
                 "error": str(error)
             }
    numeric_column=df.select_dtypes(
        include=np.number
        ).columns.to_list()
    
    categorical_columns= df.select_dtypes(
             include=["object","str","category","bool"]
         ).columns.to_list()

    date_timecolumn=data_time_column(df)

    categorical_columns = [
        column
        for column in categorical_columns
        if column not in date_timecolumn
    ]

    # Candidates are grouped by chart family. Selection then takes one
    # candidate per family in turn, so a dataset with many numerical or
    # categorical columns cannot use up the whole budget on histograms and
    # bar charts and crowd out the heatmap and time-series chart.
    candidates_by_family: dict[str, list[dict[str, Any]]] = {
        "histogram": [],
        "bar": [],
        "line": [],
        "scatter": [],
        "correlation_heatmap": [],
        "box": []
    }

    for column in numeric_column[:3]:
        candidates_by_family["histogram"].append({
            "chart_type": "histogram",
            "title": f"Distribution of {column}",
            "x": column,
            "y": None,
            "reason": (
                f"A histogram shows the distribution, spread and "
                f"possible skewness of '{column}'."
            )
        })

    for column in numeric_column[:2]:
        candidates_by_family["box"].append({
            "chart_type": "box",
            "title": f"Outlier analysis for {column}",
            "x": None,
            "y": column,
            "reason": (
                f"A box plot helps identify the median, quartiles "
                f"and possible outliers in '{column}'."
            )
        })

    for column in categorical_columns:
        unique_values = df[column].nunique(dropna=True)

        if 2 <= unique_values <= 20:
            candidates_by_family["bar"].append({
                "chart_type": "bar",
                "title": f"Frequency of {column}",
                "x": column,
                "y": "count",
                "reason": (
                    f"A bar chart clearly compares the frequency "
                    f"of the {unique_values} categories in '{column}'."
                )
            })

    # A scatter plot needs two numerical columns.
    if len(numeric_column) >= 2:
        candidates_by_family["scatter"].append({
            "chart_type": "scatter",
            "title": (
                f"{numeric_column[0]} versus {numeric_column[1]}"
            ),
            "x": numeric_column[0],
            "y": numeric_column[1],
            "reason": (
                "A scatter plot helps reveal relationships, clusters "
                "and unusual observations between two numerical columns."
            )
        })

    if len(numeric_column) >= 2:
        candidates_by_family["correlation_heatmap"].append({
            "chart_type": "correlation_heatmap",
            "title": "Correlation between numerical columns",
            "x": numeric_column,
            "y": numeric_column,
            "reason": (
                "A correlation heatmap helps identify positive and "
                "negative linear relationships between numerical columns."
            )
        })

    # A time-series line chart needs a date column and a measurement.
    if date_timecolumn and numeric_column:
        candidates_by_family["line"].append({
            "chart_type": "line",
            "title": (
                f"{numeric_column[0]} over {date_timecolumn[0]}"
            ),
            "x": date_timecolumn[0],
            "y": numeric_column[0],
            "reason": (
                "A line chart helps show how a numerical measurement "
                "changes over time."
            )
        })

    # Families are visited in this order, which decides who wins when the
    # budget is smaller than the number of applicable families.
    family_order = [
        "histogram",
        "bar",
        "line",
        "scatter",
        "correlation_heatmap",
        "box"
    ]

    deepest_family = max(
        (
            len(candidates_by_family[family])
            for family in family_order
        ),
        default=0
    )

    recommendations = []

    for candidate_index in range(deepest_family):

        for family in family_order:
            family_candidates = candidates_by_family[family]

            if candidate_index >= len(family_candidates):
                continue

            recommendations.append(
                family_candidates[candidate_index]
            )

            if len(recommendations) >= max_recommendation:
                break

        if len(recommendations) >= max_recommendation:
            break

    return {
        "success": True,
        "file_name": name,
        "numeric_columns": numeric_column,
        "categorical_columns": categorical_columns,
        "datetime_columns": date_timecolumn,
        "recommendation_count": len(recommendations),
        "recommended_chart_types": sorted({
            recommendation["chart_type"]
            for recommendation in recommendations
        }),
        "recommendations": recommendations
    }




def validate_chart_column(
    df: pd.DataFrame,
    column_name: str,
    argument_name: str
) -> None:
    """
    Confirm that a requested chart column exists.

    Args:
        df:
            Dataset used to create the chart.

        column_name:
            Column requested by the chart specification.

        argument_name:
            Name of the chart argument, such as x or y.

    Raises:
        ValueError:
            If the column name is missing or does not exist.
    """

    if not column_name:
        raise ValueError(
            f"The chart specification does not contain "
            f"a valid '{argument_name}' column."
        )

    if column_name not in df.columns:
        raise ValueError(
            f"Column '{column_name}' does not exist in the dataset."
        )
def create_plotly_figure(
    name: str,
    chart_specification: dict[str, Any]
) -> go.Figure:
    """
    Create an interactive Plotly figure from a chart specification.

    Args:
        file_name:
            Name or path of the CSV dataset.

        chart_specification:
            Dictionary describing the chart type, columns and title.

    Returns:
        A Plotly Figure object.

    Raises:
        ValueError:
            If the chart type or requested columns are invalid.
    """

    df = get_or_load_dataframe(name)

    chart_type = chart_specification.get("chart_type")
    title = chart_specification.get(
        "title",
        "Dataset visualisation"
    )

    allowed_chart_types = {
        "histogram",
        "box",
        "bar",
        "scatter",
        "correlation_heatmap",
        "line"
    }

    if chart_type not in allowed_chart_types:
        raise ValueError(
            f"Unsupported chart type: '{chart_type}'. "
            f"Allowed types are: "
            f"{sorted(allowed_chart_types)}"
        )

    if chart_type == "histogram":
        x_column = chart_specification.get("x")

        validate_chart_column(
            df,
            x_column,
            "x"
        )

        if not pd.api.types.is_numeric_dtype(df[x_column]):
            raise ValueError(
                f"Histogram column '{x_column}' must be numerical."
            )

        figure = px.histogram(
            df,
            x=x_column,
            title=title
        )

    elif chart_type == "box":
        y_column = chart_specification.get("y")

        validate_chart_column(
            df,
            y_column,
            "y"
        )

        if not pd.api.types.is_numeric_dtype(df[y_column]):
            raise ValueError(
                f"Box-plot column '{y_column}' must be numerical."
            )

        figure = px.box(
            df,
            y=y_column,
            points="outliers",
            title=title
        )

    elif chart_type == "bar":
        x_column = chart_specification.get("x")

        validate_chart_column(
            df,
            x_column,
            "x"
        )

        category_counts = (
            df[x_column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(20)
            .rename_axis(x_column)
            .reset_index(name="count")
        )

        figure = px.bar(
            category_counts,
            x=x_column,
            y="count",
            title=title
        )

    elif chart_type == "scatter":
        x_column = chart_specification.get("x")
        y_column = chart_specification.get("y")

        validate_chart_column(
            df,
            x_column,
            "x"
        )

        validate_chart_column(
            df,
            y_column,
            "y"
        )

        if not pd.api.types.is_numeric_dtype(df[x_column]):
            raise ValueError(
                f"Scatter-plot X column '{x_column}' "
                f"must be numerical."
            )

        if not pd.api.types.is_numeric_dtype(df[y_column]):
            raise ValueError(
                f"Scatter-plot Y column '{y_column}' "
                f"must be numerical."
            )

        figure = px.scatter(
            df,
            x=x_column,
            y=y_column,
            title=title
        )

    elif chart_type == "correlation_heatmap":
        requested_columns = chart_specification.get(
            "x",
            []
        )

        if not isinstance(requested_columns, list):
            raise ValueError(
                "The correlation heatmap requires a list "
                "of numerical columns."
            )

        valid_numeric_columns = [
            column
            for column in requested_columns
            if (
                column in df.columns
                and pd.api.types.is_numeric_dtype(df[column])
            )
        ]

        if len(valid_numeric_columns) < 2:
            raise ValueError(
                "A correlation heatmap requires at least "
                "two valid numerical columns."
            )

        correlation_matrix = (
            df[valid_numeric_columns]
            .corr()
            .round(2)
        )

        # Correlation is bounded at -1 and +1, so the colour scale is
        # pinned to that range. Without zmin/zmax the scale stretches to
        # whatever the data happens to contain and a correlation of zero
        # stops reading as the neutral midpoint.
        figure = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            aspect="auto",
            title=title,
            zmin=-1,
            zmax=1,
            color_continuous_midpoint=0,
            labels={
                "color": "Correlation"
            }
        )

    elif chart_type == "line":
        x_column = chart_specification.get("x")
        y_column = chart_specification.get("y")

        validate_chart_column(
            df,
            x_column,
            "x"
        )

        validate_chart_column(
            df,
            y_column,
            "y"
        )

        line_data = df[
            [x_column, y_column]
        ].copy()

        line_data[x_column] = pd.to_datetime(
            line_data[x_column],
            errors="coerce"
        )

        line_data[y_column] = pd.to_numeric(
            line_data[y_column],
            errors="coerce"
        )

        line_data = line_data.dropna(
            subset=[x_column, y_column]
        )

        if line_data.empty:
            raise ValueError(
                "No valid date and numerical values remain "
                "for the line chart."
            )

        line_data = (
            line_data
            .groupby(
                x_column,
                as_index=False
            )[y_column]
            .mean()
            .sort_values(x_column)
        )

        figure = px.line(
            line_data,
            x=x_column,
            y=y_column,
            markers=True,
            title=title
        )

    figure.update_layout(
        title=title,
        xaxis_title=figure.layout.xaxis.title.text,
        yaxis_title=figure.layout.yaxis.title.text
    )

    return figure

