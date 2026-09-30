"""
A deliberately modest forecast, and an honest report on whether it works.

The projection is the robust trend carried forward, with a month-of-year
adjustment when there is enough history to estimate one. Nothing more
elaborate. A short business series does not contain enough information to
justify a heavier model, and a heavier model mostly buys a more confident
looking line rather than a more accurate one.

What matters more than the line is the backtest. The forecast is compared
against the laziest possible alternative, assuming nothing changes, on
data it was not fitted to. When it fails to beat that, the report says so
in those words. A forecast that cannot beat "same as last period" is not
information, and presenting it as though it were is the failure mode this
module exists to prevent.

The band is an approximation. It widens with the horizon because
uncertainty compounds, but it is built from the observed spread rather
than from a distributional model of the process, so it should be read as
an indication of scale rather than a calibrated interval.
"""

from typing import Any

import numpy as np
import pandas as pd

import aggregation
import formatting
import timeseries

# Below this a projection is guesswork dressed as arithmetic.
MINIMUM_PERIODS_FOR_FORECAST = 6

# Two full cycles before a seasonal shape is estimated, so a single
# unusual December does not become "what Decembers do".
MINIMUM_PERIODS_FOR_SEASONALITY = 24

# Minimum observations of a given month before its adjustment is trusted.
MINIMUM_OBSERVATIONS_PER_SEASON = 2

# Only a monthly series gets a month-of-year adjustment. Quarterly and
# yearly series rarely have the length, and daily seasonality is a
# day-of-week question this does not attempt.
SEASONAL_GRAINS = frozenset({"month"})

# Multiplier on the observed spread for the band. Roughly a 95 per cent
# interval under normal noise, stated as approximate in the docstring.
BAND_MULTIPLIER = 1.96

DEFAULT_HORIZON = 3
MAX_HORIZON = 24
DEFAULT_BACKTEST_FOLDS = 3


def seasonal_adjustments(
    period_starts: Any,
    residuals: Any,
    grain: str
) -> dict[str, Any]:
    """
    Estimate how much each month sits above or below the trend.

    Args:
        period_starts:
            Period start timestamps.

        residuals:
            Observed value minus the trendline, per period.

        grain:
            Period grain. Only a monthly series is adjusted.

    Returns:
        The adjustment per calendar month, how many observations each
        rests on, and whether seasonality was applied at all.
    """

    starts = pd.to_datetime(pd.Series(period_starts)).reset_index(drop=True)
    values = pd.Series(residuals, dtype="float64").reset_index(drop=True)

    usable = values.notna()
    observed_periods = int(usable.sum())

    if grain not in SEASONAL_GRAINS:
        return {
            "applied": False,
            "reason": (
                f"A month-of-year adjustment is only estimated for a "
                f"monthly series, and this one is {formatting.describe_grain(grain)}."
            ),
            "adjustments": {},
            "observations": {},
        }

    if observed_periods < MINIMUM_PERIODS_FOR_SEASONALITY:
        return {
            "applied": False,
            "reason": (
                f"A month-of-year adjustment needs at least "
                f"{MINIMUM_PERIODS_FOR_SEASONALITY} months of history to "
                f"separate the season from the noise. This series has "
                f"{observed_periods}."
            ),
            "adjustments": {},
            "observations": {},
        }

    frame = pd.DataFrame({
        "month": starts.dt.month,
        "residual": values,
    }).dropna()

    adjustments: dict[int, float] = {}
    observations: dict[int, int] = {}

    for month, group in frame.groupby("month"):
        observations[int(month)] = int(len(group))

        if len(group) >= MINIMUM_OBSERVATIONS_PER_SEASON:
            # A median, so one freak month does not define the season.
            adjustments[int(month)] = float(group["residual"].median())

        else:
            adjustments[int(month)] = 0.0

    thin = [
        month
        for month, count in observations.items()
        if count < MINIMUM_OBSERVATIONS_PER_SEASON
    ]

    return {
        "applied": True,
        "reason": "",
        "adjustments": adjustments,
        "observations": observations,
        "months_without_enough_history": sorted(thin),
    }


def _future_period_starts(
    last_start: Any,
    grain: str,
    horizon: int
) -> list[pd.Timestamp]:
    """
    List the period starts that follow the end of a series.

    Args:
        last_start:
            Start of the final observed period.

        grain:
            Period grain.

        horizon:
            How many periods ahead.

    Returns:
        Future period starts, in order.
    """

    frequency = aggregation.GRAIN_FREQUENCY.get(grain, "MS")

    future = pd.date_range(
        start=pd.Timestamp(last_start),
        periods=horizon + 1,
        freq=frequency
    )

    return list(future[1:])


def baseline_forecast(
    periods: pd.DataFrame,
    grain: str = "month",
    horizon: int = DEFAULT_HORIZON,
    use_seasonality: bool = True
) -> dict[str, Any]:
    """
    Project a period series forward.

    Args:
        periods:
            Frame holding period_start and value, as built by
            aggregation.aggregate_periods.

        grain:
            Period grain.

        horizon:
            Periods ahead to project.

        use_seasonality:
            Whether to attempt a month-of-year adjustment.

    Returns:
        The projection under "forecast", the trend it rests on, the
        seasonal report, and notes describing every limitation applied.
    """

    if "period_start" not in periods or "value" not in periods:
        return {
            "success": False,
            "error": "The period series is missing period_start or value."
        }

    if horizon < 1 or horizon > MAX_HORIZON:
        return {
            "success": False,
            "error": (
                f"The horizon must be between 1 and {MAX_HORIZON} "
                f"periods."
            )
        }

    starts = pd.to_datetime(periods["period_start"]).reset_index(drop=True)
    values = pd.Series(
        periods["value"], dtype="float64"
    ).reset_index(drop=True)

    observed_periods = int(values.notna().sum())

    if observed_periods < MINIMUM_PERIODS_FOR_FORECAST:
        return {
            "success": False,
            "error": (
                f"A projection needs at least "
                f"{MINIMUM_PERIODS_FOR_FORECAST} periods with data. "
                f"This series has {observed_periods}."
            ),
        }

    trend = timeseries.fit_trend(starts, values)

    if not trend.get("success"):
        return {
            "success": False,
            "error": trend.get("error", "No trend could be fitted."),
        }

    notes: list[str] = []

    seasonal = (
        seasonal_adjustments(starts, trend["residuals"], grain)
        if use_seasonality
        else {
            "applied": False,
            "reason": "A seasonal adjustment was switched off.",
            "adjustments": {},
            "observations": {},
        }
    )

    if not seasonal["applied"] and seasonal["reason"]:
        notes.append(seasonal["reason"])

    if seasonal.get("months_without_enough_history"):
        notes.append(
            "Some calendar months appear only once, so no seasonal "
            "adjustment was applied to them."
        )

    future_starts = _future_period_starts(
        starts.iloc[-1],
        grain,
        horizon
    )

    origin = starts.iloc[0]
    scale = float(trend["scale"] or 0.0)

    projected = []

    for step, start in enumerate(future_starts, start=1):
        days_ahead = (start - origin) / pd.Timedelta(days=1)

        level = trend["intercept"] + trend["slope_per_day"] * days_ahead

        adjustment = 0.0

        if seasonal["applied"]:
            adjustment = seasonal["adjustments"].get(
                int(pd.Timestamp(start).month),
                0.0
            )

        centre = float(level + adjustment)

        # Uncertainty compounds with distance, and a short history is
        # weaker evidence than a long one.
        widening = float(
            np.sqrt(1.0 + step / max(observed_periods, 1))
        )
        half_width = BAND_MULTIPLIER * scale * widening

        projected.append({
            "period_start": pd.Timestamp(start),
            "period_label": formatting.format_period(start, grain),
            "steps_ahead": step,
            "forecast": centre,
            "lower": float(centre - half_width),
            "upper": float(centre + half_width),
        })

    forecast = pd.DataFrame(projected)

    if scale <= 0:
        notes.append(
            "Every period sits exactly on the trendline, so the "
            "projection carries no uncertainty band."
        )

    return {
        "success": True,
        "forecast": forecast,
        "horizon": int(horizon),
        "grain": grain,
        "trend": trend,
        "seasonal": seasonal,
        "seasonality_applied": bool(seasonal["applied"]),
        "observed_periods": observed_periods,
        "scale": scale,
        "notes": notes,
    }


def _forecast_values_only(
    starts: pd.Series,
    values: pd.Series,
    grain: str,
    horizon: int,
    use_seasonality: bool
) -> np.ndarray | None:
    """
    Produce just the projected numbers, for use inside the backtest.

    Args:
        starts:
            Period starts of the training window.

        values:
            Values of the training window.

        grain:
            Period grain.

        horizon:
            Periods ahead.

        use_seasonality:
            Whether to attempt a seasonal adjustment.

    Returns:
        The projected values, or None when no projection was possible.
    """

    result = baseline_forecast(
        pd.DataFrame({"period_start": starts, "value": values}),
        grain=grain,
        horizon=horizon,
        use_seasonality=use_seasonality
    )

    if not result.get("success"):
        return None

    return result["forecast"]["forecast"].to_numpy(dtype=float)


def backtest(
    periods: pd.DataFrame,
    grain: str = "month",
    horizon: int = DEFAULT_HORIZON,
    folds: int = DEFAULT_BACKTEST_FOLDS,
    use_seasonality: bool = True
) -> dict[str, Any]:
    """
    Test the projection against assuming no change.

    The series is cut back, the forecast is made from the earlier part
    only, and the result is compared with the actual values and with
    simply repeating the last observed value.

    Args:
        periods:
            The period series.

        grain:
            Period grain.

        horizon:
            Periods ahead per fold.

        folds:
            How many times to repeat the test, each ending earlier.

        use_seasonality:
            Whether the projection may use a seasonal adjustment.

    Returns:
        The error of the projection, the error of assuming no change,
        whether the projection won, and a plain sentence saying so.
    """

    starts = pd.to_datetime(periods["period_start"]).reset_index(drop=True)
    values = pd.Series(
        periods["value"], dtype="float64"
    ).reset_index(drop=True)

    total_periods = int(len(values))
    completed = []

    for fold in range(folds):
        cut = total_periods - horizon * (fold + 1)

        if cut < MINIMUM_PERIODS_FOR_FORECAST:
            break

        train_starts = starts.iloc[:cut]
        train_values = values.iloc[:cut]
        actual = values.iloc[cut:cut + horizon].to_numpy(dtype=float)

        if not np.isfinite(actual).any():
            continue

        predicted = _forecast_values_only(
            train_starts,
            train_values,
            grain,
            horizon,
            use_seasonality
        )

        if predicted is None:
            continue

        last_observed = train_values.dropna()

        if last_observed.empty:
            continue

        naive = np.full(horizon, float(last_observed.iloc[-1]))

        comparable = np.isfinite(actual)

        if not comparable.any():
            continue

        model_error = float(
            np.mean(np.abs(predicted[comparable] - actual[comparable]))
        )
        naive_error = float(
            np.mean(np.abs(naive[comparable] - actual[comparable]))
        )

        completed.append({
            "fold": len(completed) + 1,
            "trained_on": int(cut),
            "tested_on": int(comparable.sum()),
            "first_tested_period": formatting.format_period(
                starts.iloc[cut],
                grain
            ),
            "model_error": model_error,
            "naive_error": naive_error,
        })

    if not completed:
        return {
            "success": False,
            "error": (
                "There is not enough history to test the projection. A "
                f"test needs at least "
                f"{MINIMUM_PERIODS_FOR_FORECAST + horizon} periods for "
                f"a {horizon} period horizon, and this series has "
                f"{total_periods}."
            ),
            "folds": [],
        }

    model_error = float(
        np.mean([fold["model_error"] for fold in completed])
    )
    naive_error = float(
        np.mean([fold["naive_error"] for fold in completed])
    )

    beat_naive = model_error < naive_error

    skill = (
        float(1.0 - model_error / naive_error)
        if naive_error > 0
        else None
    )

    if beat_naive and skill is not None:
        verdict = (
            f"The projection was more accurate than assuming no change, "
            f"by {formatting.format_percent(skill)} on average across "
            f"{formatting.pluralise(len(completed), 'test')}."
        )

    elif beat_naive:
        verdict = (
            "The projection was more accurate than assuming no change."
        )

    else:
        verdict = (
            "The projection was no better than assuming no change. "
            "Treat the line as a rough reference rather than a "
            "prediction, and prefer the last observed value if you need "
            "a single number."
        )

    return {
        "success": True,
        "folds": completed,
        "fold_count": len(completed),
        "horizon": int(horizon),
        "model_error": model_error,
        "naive_error": naive_error,
        "beat_naive": beat_naive,
        "skill": skill,
        "verdict": verdict,
    }
