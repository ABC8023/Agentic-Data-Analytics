"""
The trendline, and how far the data sits from it.

The trend is fitted with the Theil-Sen estimator rather than least
squares. Least squares is pulled around by a single unusual period, which
matters here because finding unusual periods is the next thing the
application does: a line dragged towards an outlier hides the very point
it should expose.

Spread is measured with the median absolute deviation for the same
reason. A standard deviation is inflated by the outliers, so a band built
from it grows wide enough to swallow them.

Positions are calendar days from the first period, not row numbers, so a
series with a missing month is fitted on the real spacing.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Below this many observed periods a line says more about noise than
# about direction.
MINIMUM_POINTS = 4

# Scales the median absolute deviation to a standard deviation for
# normally distributed data.
MAD_TO_SIGMA = 1.4826


def calendar_positions(period_starts: Any) -> np.ndarray:
    """
    Express period starts as days from the first period.

    Args:
        period_starts:
            Sequence of period start timestamps.

    Returns:
        Days elapsed, as floats.
    """

    stamps = pd.to_datetime(pd.Series(period_starts))

    if stamps.empty:
        return np.array([], dtype=float)

    origin = stamps.iloc[0]

    return (
        (stamps - origin) / pd.Timedelta(days=1)
    ).to_numpy(dtype=float)


def robust_scale(residuals: Any) -> float:
    """
    Estimate spread in a way a few unusual periods cannot inflate.

    Args:
        residuals:
            Differences between observed values and the trendline.

    Returns:
        A standard-deviation-equivalent spread. Zero when the residuals
        carry no variation at all.
    """

    values = np.asarray(
        [value for value in np.ravel(residuals) if np.isfinite(value)],
        dtype=float
    )

    if values.size == 0:
        return 0.0

    median = float(np.median(values))
    absolute_deviation = float(np.median(np.abs(values - median)))

    scale = absolute_deviation * MAD_TO_SIGMA

    if scale > 0:
        return scale

    # Every residual sat at the median, which happens with a handful of
    # points or a mostly flat series. Fall back rather than report a band
    # of zero width, which would flag every period.
    fallback = float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    return fallback if np.isfinite(fallback) else 0.0


def fit_trend(
    period_starts: Any,
    values: Any
) -> dict[str, Any]:
    """
    Fit a robust straight line through a period series.

    Args:
        period_starts:
            Period start timestamps.

        values:
            Aggregated value per period. Missing entries are skipped when
            fitting but still receive a fitted value.

    Returns:
        The slope in value per day and per period, the fitted line, the
        residuals, the robust spread, a direction, and whether the slope's
        confidence interval excludes zero.
    """

    starts = pd.to_datetime(pd.Series(period_starts)).reset_index(drop=True)
    observed = pd.Series(values, dtype="float64").reset_index(drop=True)

    if len(starts) != len(observed):
        return {
            "success": False,
            "error": "Periods and values are different lengths."
        }

    positions = calendar_positions(starts)
    usable = np.isfinite(observed.to_numpy(dtype=float))
    points_used = int(usable.sum())

    if points_used < MINIMUM_POINTS:
        return {
            "success": False,
            "error": (
                f"A trend needs at least {MINIMUM_POINTS} periods with "
                f"data. This series has {points_used}."
            ),
            "points_used": points_used,
        }

    x_used = positions[usable]
    y_used = observed.to_numpy(dtype=float)[usable]

    if np.unique(x_used).size < 2:
        return {
            "success": False,
            "error": "Every observation falls on the same date.",
            "points_used": points_used,
        }

    slope, intercept, low_slope, high_slope = stats.theilslopes(
        y_used,
        x_used
    )

    fitted = intercept + slope * positions
    residuals = observed.to_numpy(dtype=float) - fitted
    scale = robust_scale(residuals)

    # A period is however many days the grain actually spans, taken from
    # the data rather than assumed, so months of different lengths and a
    # missing period both come out right.
    spacings = np.diff(positions)
    days_per_period = (
        float(np.median(spacings))
        if spacings.size
        else 0.0
    )

    span_days = float(positions[-1] - positions[0])
    total_change = float(slope * span_days)

    level = float(np.median(y_used))
    reference = abs(level) if level else 0.0

    # The confidence interval on the slope decides direction. Calling a
    # series rising when the interval spans zero would be asserting more
    # than the data supports.
    interval_excludes_zero = bool(
        np.isfinite(low_slope)
        and np.isfinite(high_slope)
        and (low_slope > 0 or high_slope < 0)
    )

    if not interval_excludes_zero:
        direction = "flat"

    elif slope > 0:
        direction = "rising"

    else:
        direction = "falling"

    return {
        "success": True,
        "slope_per_day": float(slope),
        "slope_per_period": float(slope * days_per_period),
        "intercept": float(intercept),
        "slope_low": float(low_slope),
        "slope_high": float(high_slope),
        "trend_is_significant": interval_excludes_zero,
        "direction": direction,
        "fitted": fitted,
        "residuals": residuals,
        "scale": scale,
        "points_used": points_used,
        "days_per_period": days_per_period,
        "span_days": span_days,
        "total_change": total_change,
        "level": level,
        "relative_change_per_period": (
            float(slope * days_per_period / reference)
            if reference
            else None
        ),
        "relative_total_change": (
            float(total_change / reference)
            if reference
            else None
        ),
    }


def standardised_residuals(trend: dict[str, Any]) -> np.ndarray:
    """
    Express each period's distance from the line in units of spread.

    Args:
        trend:
            Output of fit_trend.

    Returns:
        Residuals divided by the robust spread. All zeros when the spread
        is zero, since nothing can be unusual in a series with no
        variation.
    """

    residuals = np.asarray(trend.get("residuals", []), dtype=float)
    scale = float(trend.get("scale", 0.0) or 0.0)

    if residuals.size == 0 or scale <= 0:
        return np.zeros(residuals.shape, dtype=float)

    return residuals / scale


def describe_trend(
    trend: dict[str, Any],
    grain: str = "month"
) -> str:
    """
    State the direction of a series in one sentence.

    Args:
        trend:
            Output of fit_trend.

        grain:
            Period grain, used to name the unit of change.

    Returns:
        A sentence, or an explanation of why no trend was fitted.
    """

    import formatting

    if not trend.get("success"):
        return trend.get("error", "No trend could be fitted.")

    per_period = trend["slope_per_period"]
    relative = trend.get("relative_change_per_period")

    if trend["direction"] == "flat":
        return (
            "No clear direction. The change per "
            f"{grain} is too small next to the period-to-period "
            "variation to call it a trend."
        )

    movement = "rising" if per_period > 0 else "falling"

    sentence = (
        f"{movement.capitalize()} by about "
        f"{formatting.format_number(abs(per_period))} per {grain}"
    )

    if relative is not None:
        sentence += (
            f", around {formatting.format_percent(abs(relative))} of the "
            "typical level"
        )

    return sentence + "."
