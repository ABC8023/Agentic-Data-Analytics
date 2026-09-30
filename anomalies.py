"""
Which periods sit outside the band, and how often that happens by chance.

A period is flagged when its distance from the robust trendline exceeds a
multiple of the robust spread. The multiple is not taken from a normal
table: it is measured by simulation, because the estimator used here is
biased at small sample sizes and because the question is "does this series
contain an unusual period", which grows easier to fail by chance the more
periods there are.

The multipliers below were produced by tools/calibrate_anomaly_threshold.py
and target a family-wise rate of one in twenty. A genuinely stable series
raises a false flag in about five per cent of analyses, not in five per
cent of its periods.

The shape of the table is worth understanding. It falls steeply from eight
to sixty periods, because the median absolute deviation understates spread
in small samples and the multiplier has to compensate. It then flattens
and creeps back up, because a longer series offers more opportunities for
one period to look extreme. Both effects are real, and neither is visible
if the threshold is simply set to three.

Below eight periods no detection is offered at all. The simulated
multiplier there exceeds sixteen, which no real period would ever reach,
so a band would be theatre rather than a test.
"""

from typing import Any

import numpy as np
import pandas as pd

import formatting
import timeseries

# Family-wise false-alarm rate the table targets.
FALSE_ALARM_RATE = 0.05

# Produced by tools/calibrate_anomaly_threshold.py with 8,000 trials per
# length, seed 20260811. Re-run that script to reproduce or to retarget.
THRESHOLD_TABLE = {
    8: 6.857,
    9: 6.476,
    10: 5.740,
    12: 5.119,
    14: 4.769,
    16: 4.516,
    18: 4.408,
    20: 4.331,
    24: 4.183,
    30: 3.950,
    36: 3.939,
    48: 3.849,
    60: 3.782,
    84: 3.794,
    120: 3.787,
    180: 3.817,
    240: 3.903,
}

# Detection is refused below this many observed periods.
MINIMUM_PERIODS_FOR_ANOMALIES = min(THRESHOLD_TABLE)

_CALIBRATED_LENGTHS = np.array(sorted(THRESHOLD_TABLE), dtype=float)
_CALIBRATED_MULTIPLIERS = np.array(
    [THRESHOLD_TABLE[int(length)] for length in _CALIBRATED_LENGTHS],
    dtype=float
)


def threshold_for(point_count: int) -> float:
    """
    Return the band multiplier for a series of this length.

    Interpolation is done on the logarithm of the length, because the
    calibrated lengths are spaced geometrically and the curve is much
    closer to straight in that space.

    Args:
        point_count:
            Number of observed periods.

    Returns:
        The multiplier. Clamped to the ends of the calibrated range.
    """

    if point_count <= _CALIBRATED_LENGTHS[0]:
        return float(_CALIBRATED_MULTIPLIERS[0])

    if point_count >= _CALIBRATED_LENGTHS[-1]:
        return float(_CALIBRATED_MULTIPLIERS[-1])

    return float(
        np.interp(
            np.log(point_count),
            np.log(_CALIBRATED_LENGTHS),
            _CALIBRATED_MULTIPLIERS
        )
    )


def describe_false_alarm_rate() -> str:
    """
    State the calibration in words a reader can act on.

    Returns:
        A sentence describing the expected false-alarm behaviour.
    """

    one_in = int(round(1.0 / FALSE_ALARM_RATE))

    return (
        f"The band is calibrated so that a stable series raises a false "
        f"flag in roughly one analysis in {one_in}."
    )


def detect_anomalies(
    periods: pd.DataFrame,
    grain: str = "month",
    trend: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Find the periods that sit outside the calibrated band.

    Args:
        periods:
            Frame holding period_start, period_label and value.

        grain:
            Period grain, used for labels.

        trend:
            A fitted trend to reuse. Fitted here when not supplied.

    Returns:
        The band, the flagged periods, the multiplier used, and notes
        describing anything that limited the test.
    """

    if "period_start" not in periods or "value" not in periods:
        return {
            "success": False,
            "error": "The period series is missing period_start or value."
        }

    starts = pd.to_datetime(periods["period_start"]).reset_index(drop=True)
    values = pd.Series(
        periods["value"], dtype="float64"
    ).reset_index(drop=True)

    observed = int(values.notna().sum())

    if observed < MINIMUM_PERIODS_FOR_ANOMALIES:
        return {
            "success": False,
            "error": (
                f"Looking for unusual periods needs at least "
                f"{MINIMUM_PERIODS_FOR_ANOMALIES} periods with data. "
                f"This series has {observed}. With fewer than that, the "
                f"spread cannot be estimated well enough to say what "
                f"counts as unusual."
            ),
            "observed_periods": observed,
        }

    fitted_trend = trend if trend and trend.get("success") else (
        timeseries.fit_trend(starts, values)
    )

    if not fitted_trend.get("success"):
        return {
            "success": False,
            "error": fitted_trend.get(
                "error",
                "No trendline could be fitted."
            ),
            "observed_periods": observed,
        }

    scale = float(fitted_trend["scale"] or 0.0)
    multiplier = threshold_for(observed)
    fitted = np.asarray(fitted_trend["fitted"], dtype=float)

    half_width = multiplier * scale

    band = pd.DataFrame({
        "period_start": starts,
        "period_label": [
            formatting.format_period(start, grain)
            for start in starts
        ],
        "value": values,
        "expected": fitted,
        "lower": fitted - half_width,
        "upper": fitted + half_width,
    })

    notes: list[str] = []

    if scale <= 0:
        notes.append(
            "Every period sits exactly on the trendline, so there is "
            "nothing for the band to measure and no period can be "
            "unusual."
        )

        return {
            "success": True,
            "band": band,
            "anomalies": [],
            "anomaly_count": 0,
            "multiplier": multiplier,
            "scale": scale,
            "observed_periods": observed,
            "trend": fitted_trend,
            "false_alarm_rate": FALSE_ALARM_RATE,
            "notes": notes,
        }

    scores = timeseries.standardised_residuals(fitted_trend)

    anomalies = []

    for index in range(len(band)):
        value = values.iloc[index]

        if not np.isfinite(value):
            continue

        score = float(scores[index])

        if abs(score) <= multiplier:
            continue

        expected = float(fitted[index])

        anomalies.append({
            "period_start": pd.Timestamp(starts.iloc[index]),
            "period_label": band["period_label"].iloc[index],
            "value": float(value),
            "expected": expected,
            "deviation": float(value - expected),
            "relative_deviation": (
                float((value - expected) / abs(expected))
                if expected
                else None
            ),
            "score": score,
            "direction": "above" if score > 0 else "below",
        })

    if not anomalies:
        notes.append(
            "No period fell outside the band. "
            + describe_false_alarm_rate()
        )

    return {
        "success": True,
        "band": band,
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "multiplier": multiplier,
        "scale": scale,
        "observed_periods": observed,
        "trend": fitted_trend,
        "false_alarm_rate": FALSE_ALARM_RATE,
        "notes": notes,
    }


def describe_anomalies(result: dict[str, Any]) -> str:
    """
    Summarise the detection in one sentence.

    Args:
        result:
            Output of detect_anomalies.

    Returns:
        A sentence, or the reason no detection was possible.
    """

    if not result.get("success"):
        return result.get("error", "No detection was possible.")

    count = result["anomaly_count"]

    if count == 0:
        return (
            "Every period sits inside the expected band. "
            + describe_false_alarm_rate()
        )

    highest = max(
        result["anomalies"],
        key=lambda entry: abs(entry["score"])
    )

    return (
        f"{formatting.pluralise(count, 'period')} fell outside the "
        f"expected band. The furthest was {highest['period_label']}, "
        f"{formatting.format_number(abs(highest['deviation']))} "
        f"{highest['direction']} the trendline."
    )
