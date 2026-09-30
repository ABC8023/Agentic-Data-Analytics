"""
Plotly figures for the time-series and dashboard tabs.
"""


import plotly.express as px
import plotly.graph_objects as go

import formatting
import segments
from ui_theme import CHART_ROLES, CHART_SEQUENTIAL


def build_timeline_figure(
    periods,
    grain,
    trend,
    detected,
    projection,
    measure_label
):
    """
    Draw the period series with its band, flags and projection.

    Layers are added back to front so the observed line is never hidden:
    the expected band, then the trendline, then the observed values, then
    the flagged periods, then the projection.

    Args:
        periods:
            The period series.

        grain:
            Period grain, for axis labels.

        trend:
            Output of timeseries.fit_trend.

        detected:
            Output of anomalies.detect_anomalies.

        projection:
            Output of forecasting.baseline_forecast.

        measure_label:
            Name for the value axis.

    Returns:
        A Plotly figure.
    """

    figure = go.Figure()

    band_fill = CHART_ROLES["band_fill"]
    band_line = CHART_ROLES["band_line"]

    if detected.get("success") and detected.get("scale", 0) > 0:
        band = detected["band"]

        figure.add_trace(go.Scatter(
            x=band["period_start"],
            y=band["upper"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            name="Expected upper"
        ))
        figure.add_trace(go.Scatter(
            x=band["period_start"],
            y=band["lower"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=band_fill,
            hoverinfo="skip",
            name="Expected range"
        ))

    if trend.get("success"):
        figure.add_trace(go.Scatter(
            x=periods["period_start"],
            y=trend["fitted"],
            mode="lines",
            line={"dash": "dash", "width": 1.5, "color": band_line},
            name="Trend"
        ))

    figure.add_trace(go.Scatter(
        x=periods["period_start"],
        y=periods["value"],
        mode="lines+markers",
        # Coloured explicitly: Plotly assigns palette colours by trace
        # position, and the band traces before this one would push the
        # observed series off the accent.
        line={"width": 2.2, "color": CHART_ROLES["observed"]},
        marker={"size": 6, "color": CHART_ROLES["observed"]},
        name=measure_label,
        connectgaps=False
    ))

    if detected.get("success") and detected.get("anomalies"):
        figure.add_trace(go.Scatter(
            x=[entry["period_start"] for entry in detected["anomalies"]],
            y=[entry["value"] for entry in detected["anomalies"]],
            mode="markers",
            marker={
                "size": 13,
                "symbol": "circle-open",
                "line": {"width": 2.5},
                "color": CHART_ROLES["flag"],
            },
            name="Outside the band"
        ))

    if projection.get("success"):
        forecast = projection["forecast"]

        # Join the projection to the last observed point so the line does
        # not appear to start from nowhere.
        bridge_x = [periods["period_start"].iloc[-1]]
        bridge_y = [periods["value"].iloc[-1]]

        if projection.get("scale", 0) > 0:
            figure.add_trace(go.Scatter(
                x=forecast["period_start"],
                y=forecast["upper"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                name="Projection upper"
            ))
            figure.add_trace(go.Scatter(
                x=forecast["period_start"],
                y=forecast["lower"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=CHART_ROLES["projection_fill"],
                hoverinfo="skip",
                name="Projection range"
            ))

        figure.add_trace(go.Scatter(
            x=bridge_x + list(forecast["period_start"]),
            y=bridge_y + list(forecast["forecast"]),
            mode="lines+markers",
            line={"dash": "dot", "width": 2, "color": CHART_ROLES["projection"]},
            marker={"size": 5},
            name="Projection"
        ))

    figure.update_layout(
        title=(
            f"{measure_label}, "
            f"{formatting.describe_grain(grain)}"
        ),
        xaxis_title="Period",
        yaxis_title=measure_label,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        }
    )

    return figure

def build_waterfall_figure(waterfall, measure_label):
    """
    Draw the latest period-on-period change, part by part.

    The bars run from the previous total to the latest one, so the shape
    of the chart is itself the reconciliation: if the closing bar does not
    land on the observed figure, the parts do not explain the movement.
    """

    reduced = segments.waterfall_steps(waterfall["table"])
    steps = reduced["steps"]

    figure = go.Figure(go.Waterfall(
        orientation="v",
        measure=(
            ["absolute"]
            + ["relative"] * len(steps)
            + ["total"]
        ),
        x=(
            [waterfall["previous_label"]]
            + [str(name) for name in steps["segment"]]
            + [waterfall["latest_label"]]
        ),
        y=(
            [waterfall["previous_total"]]
            + [float(value) for value in steps["change"]]
            + [None]
        ),
        text=(
            [formatting.format_compact(waterfall["previous_total"])]
            + [
                formatting.format_signed(value)
                for value in steps["change"]
            ]
            + [formatting.format_compact(waterfall["latest_total"])]
        ),
        textposition="outside",
        connector={"line": {"color": CHART_ROLES["connector"]}},
        increasing={"marker": {"color": CHART_ROLES["increase"]}},
        decreasing={"marker": {"color": CHART_ROLES["decrease"]}},
        totals={"marker": {"color": CHART_ROLES["total"]}}
    ))

    figure.update_layout(
        title=(
            f"{measure_label}: {waterfall['previous_label']} to "
            f"{waterfall['latest_label']}"
        ),
        yaxis_title=measure_label,
        showlegend=False
    )

    return figure


def build_segment_heatmap(matrix, as_shares, grain, measure_label):
    """
    Draw the period-by-segment table.

    Raw totals mostly show which periods were large. Shares of each
    period's own total show the mix moving, which is usually the question
    a reader has in mind when they ask for a breakdown by segment.
    """

    reduced = segments.collapse_matrix(matrix)
    values = reduced["matrix"]

    if as_shares:
        values = segments.segment_period_shares(values)

    labels = [
        formatting.format_period(start, grain)
        for start in values.index
    ]

    figure = px.imshow(
        values.transpose(),
        x=labels,
        y=[str(name) for name in values.columns],
        aspect="auto",
        color_continuous_scale=CHART_SEQUENTIAL,
        labels={
            "x": "Period",
            "y": "Part",
            "color": (
                "Share of period" if as_shares else measure_label
            ),
        }
    )

    figure.update_layout(
        title=(
            "Share of each period by part"
            if as_shares
            else f"{measure_label} by part"
        )
    )

    if as_shares:
        figure.update_coloraxes(
            colorbar={"tickformat": ".0%"}
        )

    return figure


