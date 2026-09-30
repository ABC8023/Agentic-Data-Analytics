"""The robust trendline and the spread around it."""

import unittest

import numpy as np
import pandas as pd

import timeseries


def monthly_starts(count, start="2024-01-01"):
    return pd.date_range(start, periods=count, freq="MS")


class CalendarPositionTest(unittest.TestCase):

    def test_positions_are_days_from_the_first_period(self):
        positions = timeseries.calendar_positions(
            pd.to_datetime(["2024-01-01", "2024-01-11", "2024-02-01"])
        )

        self.assertEqual(list(positions), [0.0, 10.0, 31.0])

    def test_an_empty_series_gives_an_empty_result(self):
        self.assertEqual(
            timeseries.calendar_positions(pd.Series([], dtype="datetime64[ns]")).size,
            0
        )


class RobustScaleTest(unittest.TestCase):

    def test_spread_is_estimated_for_clean_noise(self):
        rng = np.random.default_rng(1)
        residuals = rng.normal(0, 5, 400)

        scale = timeseries.robust_scale(residuals)

        self.assertAlmostEqual(scale, 5.0, delta=1.0)

    def test_a_single_extreme_value_does_not_inflate_the_spread(self):
        rng = np.random.default_rng(2)
        residuals = rng.normal(0, 2, 200)
        residuals[50] = 500.0

        robust = timeseries.robust_scale(residuals)
        classical = float(np.std(residuals, ddof=1))

        self.assertLess(robust, 4.0)
        self.assertGreater(classical, 20.0)

    def test_identical_residuals_give_zero_spread(self):
        self.assertEqual(
            timeseries.robust_scale(np.zeros(10)),
            0.0
        )

    def test_missing_values_are_ignored(self):
        residuals = np.array([1.0, np.nan, -1.0, np.nan, 1.0, -1.0])

        self.assertGreater(timeseries.robust_scale(residuals), 0.0)

    def test_an_empty_input_gives_zero(self):
        self.assertEqual(timeseries.robust_scale([]), 0.0)


class TrendDirectionTest(unittest.TestCase):

    def test_a_rising_series_is_called_rising(self):
        values = [100 + 5 * index for index in range(24)]

        trend = timeseries.fit_trend(monthly_starts(24), values)

        self.assertTrue(trend["success"])
        self.assertEqual(trend["direction"], "rising")
        self.assertTrue(trend["trend_is_significant"])
        self.assertGreater(trend["slope_per_day"], 0)

    def test_a_falling_series_is_called_falling(self):
        values = [500 - 8 * index for index in range(24)]

        trend = timeseries.fit_trend(monthly_starts(24), values)

        self.assertEqual(trend["direction"], "falling")
        self.assertLess(trend["slope_per_period"], 0)

    def test_pure_noise_is_called_flat(self):
        # Claiming a direction here would assert more than the data
        # supports, so the confidence interval decides.
        rng = np.random.default_rng(3)
        values = rng.normal(100, 10, 36)

        trend = timeseries.fit_trend(monthly_starts(36), values)

        self.assertEqual(trend["direction"], "flat")
        self.assertFalse(trend["trend_is_significant"])

    def test_a_flat_line_is_not_called_a_trend(self):
        trend = timeseries.fit_trend(
            monthly_starts(12),
            [200.0] * 12
        )

        self.assertTrue(trend["success"])
        self.assertEqual(trend["direction"], "flat")
        self.assertEqual(trend["slope_per_day"], 0.0)


class RobustnessTest(unittest.TestCase):

    def test_one_extreme_period_barely_moves_the_line(self):
        # This is why Theil-Sen is used. Least squares would chase the
        # outlier and then the anomaly detector would miss it.
        values = [10 + 0.5 * index for index in range(20)]
        values[10] = 500.0

        starts = monthly_starts(20)
        trend = timeseries.fit_trend(starts, values)

        positions = timeseries.calendar_positions(starts)
        least_squares_slope = float(
            np.polyfit(positions, values, 1)[0]
        )

        clean_slope_per_day = 0.5 / trend["days_per_period"]

        # Theil-Sen stays on the underlying slope.
        self.assertAlmostEqual(
            trend["slope_per_day"],
            clean_slope_per_day,
            delta=clean_slope_per_day * 0.25
        )

        # Least squares is dragged well off it by the same single point.
        self.assertGreater(
            least_squares_slope,
            clean_slope_per_day * 1.5
        )

    def test_the_outlier_stands_out_in_standardised_terms(self):
        values = [10 + 0.5 * index for index in range(20)]
        values[10] = 500.0

        trend = timeseries.fit_trend(monthly_starts(20), values)
        scores = timeseries.standardised_residuals(trend)

        self.assertEqual(int(np.argmax(np.abs(scores))), 10)
        self.assertGreater(abs(scores[10]), 5)


class GapAndShapeTest(unittest.TestCase):

    def test_missing_periods_are_skipped_but_still_fitted(self):
        values = [100 + 5 * index for index in range(12)]
        values[4] = np.nan
        values[7] = np.nan

        trend = timeseries.fit_trend(monthly_starts(12), values)

        self.assertTrue(trend["success"])
        self.assertEqual(trend["points_used"], 10)
        self.assertEqual(len(trend["fitted"]), 12)
        self.assertTrue(np.all(np.isfinite(trend["fitted"])))

    def test_uneven_spacing_is_measured_in_days(self):
        starts = pd.to_datetime([
            "2024-01-01", "2024-02-01", "2024-05-01", "2024-06-01",
            "2024-07-01",
        ])

        trend = timeseries.fit_trend(starts, [10.0, 20.0, 50.0, 60.0, 70.0])

        self.assertTrue(trend["success"])
        self.assertGreater(trend["span_days"], 180)

    def test_the_period_length_comes_from_the_data(self):
        trend = timeseries.fit_trend(
            monthly_starts(24),
            [100 + index for index in range(24)]
        )

        self.assertAlmostEqual(trend["days_per_period"], 30.5, delta=1.5)

    def test_mismatched_lengths_are_refused(self):
        trend = timeseries.fit_trend(monthly_starts(5), [1.0, 2.0])

        self.assertFalse(trend["success"])
        self.assertIn("different lengths", trend["error"])

    def test_too_few_points_is_refused_with_the_count(self):
        trend = timeseries.fit_trend(monthly_starts(3), [1.0, 2.0, 3.0])

        self.assertFalse(trend["success"])
        self.assertIn("at least", trend["error"])
        self.assertEqual(trend["points_used"], 3)

    def test_a_series_that_is_all_gaps_is_refused(self):
        trend = timeseries.fit_trend(
            monthly_starts(8),
            [np.nan] * 8
        )

        self.assertFalse(trend["success"])


class ReportedFiguresTest(unittest.TestCase):

    def setUp(self):
        self.trend = timeseries.fit_trend(
            monthly_starts(24),
            [100 + 5 * index for index in range(24)]
        )

    def test_total_change_matches_the_span(self):
        self.assertAlmostEqual(
            self.trend["total_change"],
            5 * 23,
            delta=6
        )

    def test_change_per_period_is_reported(self):
        self.assertAlmostEqual(
            self.trend["slope_per_period"],
            5.0,
            delta=0.5
        )

    def test_relative_change_uses_the_typical_level(self):
        self.assertIsNotNone(self.trend["relative_change_per_period"])
        self.assertGreater(self.trend["relative_change_per_period"], 0)

    def test_a_zero_level_leaves_relative_change_undefined(self):
        # Dividing by a level of zero would produce infinity, which is
        # not a useful thing to show a reader.
        trend = timeseries.fit_trend(
            monthly_starts(12),
            [0.0] * 6 + [0.0] * 6
        )

        self.assertIsNone(trend["relative_change_per_period"])

    def test_standardised_residuals_are_zero_without_spread(self):
        trend = timeseries.fit_trend(monthly_starts(12), [7.0] * 12)

        scores = timeseries.standardised_residuals(trend)

        self.assertTrue(np.all(scores == 0))


class DescriptionTest(unittest.TestCase):

    def test_a_rising_series_is_described_with_its_rate(self):
        trend = timeseries.fit_trend(
            monthly_starts(24),
            [100 + 5 * index for index in range(24)]
        )

        sentence = timeseries.describe_trend(trend, "month")

        self.assertIn("Rising", sentence)
        self.assertIn("per month", sentence)

    def test_a_flat_series_says_there_is_no_direction(self):
        rng = np.random.default_rng(4)
        trend = timeseries.fit_trend(
            monthly_starts(36),
            rng.normal(100, 10, 36)
        )

        self.assertIn("No clear direction", timeseries.describe_trend(trend))

    def test_a_failed_fit_explains_itself(self):
        trend = timeseries.fit_trend(monthly_starts(2), [1.0, 2.0])

        self.assertIn("at least", timeseries.describe_trend(trend))


if __name__ == "__main__":
    unittest.main()
