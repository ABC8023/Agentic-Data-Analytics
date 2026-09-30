"""
Calibrate the anomaly band by simulation.

The band around the trendline is a multiple of the robust spread. Picking
that multiple from a normal table would be wrong twice over. The spread is
estimated from a median absolute deviation, which is biased at small
sample sizes, and the question being asked is not "is this one period
unusual" but "does this series contain an unusual period", which is a
different and much easier test to fail by chance the more periods there
are.

So the multiplier is measured instead. Many stable series are simulated,
each is put through the same trend fit and the same spread estimator the
application uses, and the largest standardised residual is recorded. The
multiplier for a given length is the percentile of that distribution
matching the false-alarm rate we are willing to accept.

The result is a family-wise rate: with the default of one in twenty, a
genuinely stable series raises a false flag in about five per cent of
analyses, not five per cent of periods.

Run from the repository root:

    python tools/calibrate_anomaly_threshold.py
    python tools/calibrate_anomaly_threshold.py --trials 20000 --check

The table is printed ready to paste into anomalies.py and written to
tools/anomaly_threshold_table.json for the record.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import timeseries  # noqa: E402

# Series lengths to calibrate. Dense where short series behave oddly,
# sparser once the curve flattens.
SERIES_LENGTHS = (
    4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 30, 36, 48, 60, 84,
    120, 180, 240,
)

DEFAULT_TRIALS = 8000
DEFAULT_FALSE_ALARM_RATE = 0.05

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "anomaly_threshold_table.json"
)


def largest_standardised_residual(
    length: int,
    generator: np.random.Generator
) -> float:
    """
    Simulate one stable series and return its most extreme period.

    The series has no trend and normally distributed noise. It is fitted
    and scaled exactly as a real series would be, so the multiplier
    inherits every quirk of those estimators.

    Args:
        length:
            Number of periods.

        generator:
            Random source.

    Returns:
        The largest absolute standardised residual, or nan when the
        series produced no usable spread.
    """

    positions = np.arange(length, dtype=float)
    values = generator.normal(0.0, 1.0, length)

    slope, intercept, _, _ = stats.theilslopes(values, positions)

    residuals = values - (intercept + slope * positions)
    scale = timeseries.robust_scale(residuals)

    if scale <= 0:
        return float("nan")

    return float(np.max(np.abs(residuals / scale)))


def calibrate(
    trials: int = DEFAULT_TRIALS,
    false_alarm_rate: float = DEFAULT_FALSE_ALARM_RATE,
    seed: int = 20260811
) -> dict[int, float]:
    """
    Measure the multiplier for every calibrated series length.

    Args:
        trials:
            Simulated series per length.

        false_alarm_rate:
            Share of stable series allowed to raise a flag.

        seed:
            Fixed so the table is reproducible.

    Returns:
        Series length mapped to its multiplier.
    """

    generator = np.random.default_rng(seed)
    percentile = (1.0 - false_alarm_rate) * 100.0

    table: dict[int, float] = {}

    for length in SERIES_LENGTHS:
        extremes = np.array([
            largest_standardised_residual(length, generator)
            for _ in range(trials)
        ])

        usable = extremes[np.isfinite(extremes)]

        if usable.size == 0:
            raise RuntimeError(
                f"No usable simulations at length {length}."
            )

        table[length] = round(
            float(np.percentile(usable, percentile)),
            3
        )

        print(
            f"  n={length:>4}  multiplier={table[length]:>6.3f}  "
            f"({usable.size:,} usable trials)",
            flush=True
        )

    return table


def measure_achieved_rate(
    table: dict[int, float],
    trials: int,
    seed: int = 771
) -> dict[int, float]:
    """
    Check the table against fresh simulations.

    A calibration that is only ever tested on the data it was fitted to
    proves nothing, so this uses a different seed.

    Args:
        table:
            Multipliers to test.

        trials:
            Simulated series per length.

        seed:
            Random seed, deliberately different from the calibration.

    Returns:
        Series length mapped to the share of stable series that raised a
        flag.
    """

    generator = np.random.default_rng(seed)
    achieved: dict[int, float] = {}

    for length, multiplier in table.items():
        flags = 0
        usable = 0

        for _ in range(trials):
            extreme = largest_standardised_residual(length, generator)

            if not np.isfinite(extreme):
                continue

            usable += 1

            if extreme > multiplier:
                flags += 1

        achieved[length] = round(flags / usable, 4) if usable else 0.0

    return achieved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the anomaly band multiplier."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Simulated series per length."
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_FALSE_ALARM_RATE,
        help="Family-wise false-alarm rate to target."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the table against fresh simulations."
    )

    arguments = parser.parse_args()

    print(
        f"Calibrating for a {arguments.rate:.0%} family-wise "
        f"false-alarm rate, {arguments.trials:,} trials per length."
    )

    table = calibrate(arguments.trials, arguments.rate)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "false_alarm_rate": arguments.rate,
                "trials": arguments.trials,
                "multipliers": {
                    str(length): value
                    for length, value in table.items()
                },
            },
            handle,
            indent=2
        )

    print(f"\nWritten to {OUTPUT_PATH}")
    print("\nPaste into anomalies.py:\n")
    print("THRESHOLD_TABLE = {")

    for length, value in table.items():
        print(f"    {length}: {value},")

    print("}")

    if arguments.check:
        print("\nVerifying against fresh simulations...")

        achieved = measure_achieved_rate(
            table,
            max(2000, arguments.trials // 4)
        )

        worst = 0.0

        for length, rate in achieved.items():
            drift = abs(rate - arguments.rate)
            worst = max(worst, drift)

            print(
                f"  n={length:>4}  achieved={rate:.3f}  "
                f"target={arguments.rate:.3f}"
            )

        print(f"\nLargest deviation from target: {worst:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
