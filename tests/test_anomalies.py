"""
The anomaly band, and whether its calibration is honest.

The last test class is the important one. It re-runs the simulation the
calibration was built from, with a different seed, and checks the band
actually behaves at the advertised rate. A threshold nobody verifies is
just a number in a constant.
"""

import unittest

import numpy as np
import pandas as pd
from scipy import stats

import anomalies
import timeseries


def period_frame(values, start="2022-01-01", freq="MS"):
    starts = pd.date_range(start, periods=len(values), freq=freq)

    return pd.DataFrame({
        "period_start": starts,
        "period_label": [str(stamp.date()) for stamp in starts],
        "value": pd.Series(values, dtype="float64"),
    })


class ThresholdTableTest(unittest.TestCase):

    def test_the_table_starts_where_detection_starts(self):
        self.assertEqual(
            anomalies.MINIMUM_PERIODS_FOR_ANOMALIES,
            min(anomalies.THRESHOLD_TABLE)
        )

    def test_a_calibrated_length_returns_its_own_value(self):
        for length, expected in anomalies.THRESHOLD_TABLE.items():
            with self.subTest(length=length):
                self.assertAlmostEqual(
                    anomalies.threshold_for(length),
                    expected,
                    places=3
                )

    def test_an_uncalibrated_length_is_interpolated_between_neighbours(self):
        value = anomalies.threshold_for(11)

        self.assertLess(value, anomalies.THRESHOLD_TABLE[10])
        self.assertGreater(value, anomalies.THRESHOLD_TABLE[12])

    def test_short_series_are_clamped_to_the_first_entry(self):
        self.assertEqual(
            anomalies.threshold_for(2),
            anomalies.THRESHOLD_TABLE[8]
        )

    def test_long_series_are_clamped_to_the_last_entry(self):
        longest = max(anomalies.THRESHOLD_TABLE)

        self.assertEqual(
            anomalies.threshold_for(5000),
            anomalies.THRESHOLD_TABLE[longest]
        )

    def test_the_multiplier_falls_as_short_series_grow(self):
        # The median absolute deviation understates spread in small
        # samples, so the multiplier has to be larger there.
        self.assertGreater(
            anomalies.threshold_for(10),
            anomalies.threshold_for(30)
        )

    def test_every_multiplier_is_a_plausible_number_of_deviations(self):
        for length, value in anomalies.THRESHOLD_TABLE.items():
            with self.subTest(length=length):
                self.assertGreater(value, 3.0)
                self.assertLess(value, 8.0)

    def test_the_embedded_table_matches_the_calibration_output(self):
        # Guards against the constant drifting away from what the tool
        # last produced. Skipped when the record file is absent.
        import json
        import os

        record = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tools",
            "anomaly_threshold_table.json"
        )

        if not os.path.exists(record):
            self.skipTest("no calibration record present")

        with open(record, encoding="utf-8") as handle:
            saved = json.load(handle)

        self.assertEqual(
            saved["false_alarm_rate"],
            anomalies.FALSE_ALARM_RATE
        )

        for length, value in anomalies.THRESHOLD_TABLE.items():
            with self.subTest(length=length):
                self.assertAlmostEqual(
                    saved["multipliers"][str(length)],
                    value,
                    places=3
                )


class DetectionTest(unittest.TestCase):

    def test_a_planted_spike_is_found(self):
        values = [100 + 2 * index for index in range(30)]
        values[17] = 400.0

        result = anomalies.detect_anomalies(period_frame(values))

        self.assertTrue(result["success"])
        self.assertEqual(result["anomaly_count"], 1)
        self.assertEqual(
            result["anomalies"][0]["period_start"].strftime("%Y-%m"),
            "2023-06"
        )
        self.assertEqual(result["anomalies"][0]["direction"], "above")

    def test_a_planted_collapse_is_found(self):
        values = [500 + 2 * index for index in range(30)]
        values[9] = 50.0

        result = anomalies.detect_anomalies(period_frame(values))

        self.assertEqual(result["anomaly_count"], 1)
        self.assertEqual(result["anomalies"][0]["direction"], "below")

    def test_the_deviation_is_reported_against_the_trendline(self):
        values = [100.0] * 30
        values[15] = 250.0

        result = anomalies.detect_anomalies(period_frame(values))

        anomaly = result["anomalies"][0]

        self.assertAlmostEqual(anomaly["expected"], 100.0, delta=5)
        self.assertAlmostEqual(anomaly["deviation"], 150.0, delta=6)
        self.assertGreater(anomaly["relative_deviation"], 1.0)

    def test_a_clean_trend_raises_nothing(self):
        values = [100 + 3 * index for index in range(36)]

        result = anomalies.detect_anomalies(period_frame(values))

        self.assertEqual(result["anomaly_count"], 0)

    def test_ordinary_noise_raises_nothing(self):
        rng = np.random.default_rng(9)
        values = 100 + rng.normal(0, 5, 36)

        result = anomalies.detect_anomalies(period_frame(values))

        self.assertEqual(result["anomaly_count"], 0)

    def test_the_band_brackets_the_expected_line(self):
        values = [100 + 2 * index + (index % 3) for index in range(24)]

        result = anomalies.detect_anomalies(period_frame(values))
        band = result["band"]

        self.assertTrue((band["lower"] <= band["expected"]).all())
        self.assertTrue((band["upper"] >= band["expected"]).all())

    def test_a_short_series_is_refused_with_the_reason(self):
        result = anomalies.detect_anomalies(period_frame([1.0] * 6))

        self.assertFalse(result["success"])
        self.assertIn("at least", result["error"])
        self.assertEqual(result["observed_periods"], 6)

    def test_a_flat_series_has_nothing_to_flag(self):
        result = anomalies.detect_anomalies(period_frame([42.0] * 20))

        self.assertTrue(result["success"])
        self.assertEqual(result["anomaly_count"], 0)
        self.assertEqual(result["scale"], 0.0)
        self.assertTrue(
            any("nothing for the band" in note for note in result["notes"])
        )

    def test_gaps_are_never_flagged(self):
        values = [100 + 2 * index for index in range(30)]
        values[5] = np.nan
        values[12] = np.nan

        result = anomalies.detect_anomalies(period_frame(values))

        flagged = {
            entry["period_start"]
            for entry in result["anomalies"]
        }
        gap_dates = set(
            period_frame(values)["period_start"].iloc[[5, 12]]
        )

        self.assertEqual(flagged & gap_dates, set())

    def test_a_fitted_trend_can_be_reused(self):
        values = [100 + 2 * index for index in range(24)]
        frame = period_frame(values)

        trend = timeseries.fit_trend(
            frame["period_start"],
            frame["value"]
        )

        result = anomalies.detect_anomalies(frame, trend=trend)

        self.assertTrue(result["success"])
        self.assertIs(result["trend"], trend)

    def test_a_missing_column_is_refused(self):
        result = anomalies.detect_anomalies(
            pd.DataFrame({"period_start": [1, 2, 3]})
        )

        self.assertFalse(result["success"])


class DescriptionTest(unittest.TestCase):

    def test_a_quiet_series_says_so_and_states_the_rate(self):
        values = [100 + 3 * index for index in range(30)]

        sentence = anomalies.describe_anomalies(
            anomalies.detect_anomalies(period_frame(values))
        )

        self.assertIn("inside the expected band", sentence)
        self.assertIn("one analysis in 20", sentence)

    def test_a_flagged_series_names_the_worst_period(self):
        values = [100.0] * 30
        values[20] = 900.0

        sentence = anomalies.describe_anomalies(
            anomalies.detect_anomalies(period_frame(values))
        )

        self.assertIn("fell outside", sentence)
        self.assertIn("above", sentence)

    def test_a_refusal_explains_itself(self):
        sentence = anomalies.describe_anomalies(
            anomalies.detect_anomalies(period_frame([1.0] * 5))
        )

        self.assertIn("at least", sentence)


class CalibrationHoldsTest(unittest.TestCase):
    """
    Verify the advertised false-alarm rate on fresh simulations.

    This repeats the calibration experiment with a different seed and far
    fewer trials, so it stays fast enough to run on every commit while
    still catching a table that has drifted away from what the code does.
    """

    TRIALS = 1200
    LENGTHS = (12, 24, 60)

    def observed_false_alarm_rate(self, length, seed):
        generator = np.random.default_rng(seed)
        multiplier = anomalies.threshold_for(length)

        positions = np.arange(length, dtype=float)
        flags = 0
        usable = 0

        for _ in range(self.TRIALS):
            values = generator.normal(0.0, 1.0, length)

            slope, intercept, _, _ = stats.theilslopes(values, positions)
            residuals = values - (intercept + slope * positions)
            scale = timeseries.robust_scale(residuals)

            if scale <= 0:
                continue

            usable += 1

            if np.max(np.abs(residuals / scale)) > multiplier:
                flags += 1

        return flags / usable if usable else 0.0

    def test_stable_series_are_flagged_at_about_the_advertised_rate(self):
        for index, length in enumerate(self.LENGTHS):
            with self.subTest(length=length):
                rate = self.observed_false_alarm_rate(
                    length,
                    seed=4200 + index
                )

                # Sampling error on 1,200 trials at a 5 per cent rate has
                # a standard error near 0.6 points, so a three point
                # window is generous without being meaningless.
                self.assertAlmostEqual(
                    rate,
                    anomalies.FALSE_ALARM_RATE,
                    delta=0.03,
                    msg=(
                        f"length {length} flagged {rate:.1%} of stable "
                        f"series, target "
                        f"{anomalies.FALSE_ALARM_RATE:.1%}"
                    )
                )

    def test_a_planted_spike_is_caught_far_more_often_than_by_chance(self):
        # The band has to be tight enough to be useful, not just safe.
        generator = np.random.default_rng(77)
        length = 24
        caught = 0
        trials = 300

        for _ in range(trials):
            values = 100 + generator.normal(0.0, 5.0, length)
            values[12] += 60.0

            result = anomalies.detect_anomalies(period_frame(values))

            if result["anomaly_count"] >= 1:
                caught += 1

        self.assertGreater(caught / trials, 0.8)


if __name__ == "__main__":
    unittest.main()
