"""The baseline projection, its seasonal adjustment, and its backtest."""

import unittest

import numpy as np
import pandas as pd

import forecasting


def period_frame(values, start="2022-01-01", freq="MS"):
    starts = pd.date_range(start, periods=len(values), freq=freq)

    return pd.DataFrame({
        "period_start": starts,
        "value": pd.Series(values, dtype="float64"),
    })


def seasonal_values(months, level=100.0, growth=1.0, swing=40.0):
    """A rising series with a strong December peak and June trough."""

    starts = pd.date_range("2022-01-01", periods=months, freq="MS")
    values = []

    for index, start in enumerate(starts):
        season = swing if start.month == 12 else (
            -swing if start.month == 6 else 0.0
        )
        values.append(level + growth * index + season)

    return values


class SeasonalAdjustmentTest(unittest.TestCase):

    def test_a_non_monthly_series_is_not_adjusted(self):
        frame = period_frame(list(range(40)), freq="QS")

        result = forecasting.seasonal_adjustments(
            frame["period_start"],
            frame["value"],
            "quarter"
        )

        self.assertFalse(result["applied"])
        self.assertIn("monthly", result["reason"])

    def test_a_short_monthly_series_is_not_adjusted(self):
        # One year cannot separate a seasonal shape from noise.
        frame = period_frame([1.0] * 12)

        result = forecasting.seasonal_adjustments(
            frame["period_start"],
            frame["value"],
            "month"
        )

        self.assertFalse(result["applied"])
        self.assertIn("at least", result["reason"])

    def test_two_years_of_history_is_enough(self):
        frame = period_frame([1.0] * 30)

        result = forecasting.seasonal_adjustments(
            frame["period_start"],
            frame["value"],
            "month"
        )

        self.assertTrue(result["applied"])

    def test_the_seasonal_shape_is_recovered(self):
        values = seasonal_values(36)
        frame = period_frame(values)

        import timeseries

        trend = timeseries.fit_trend(
            frame["period_start"],
            frame["value"]
        )

        result = forecasting.seasonal_adjustments(
            frame["period_start"],
            trend["residuals"],
            "month"
        )

        self.assertTrue(result["applied"])
        self.assertGreater(result["adjustments"][12], 20)
        self.assertLess(result["adjustments"][6], -20)
        self.assertAlmostEqual(result["adjustments"][3], 0, delta=10)


class ForecastShapeTest(unittest.TestCase):

    def test_a_short_series_is_refused_with_the_requirement(self):
        result = forecasting.baseline_forecast(period_frame([1.0] * 4))

        self.assertFalse(result["success"])
        self.assertIn("at least", result["error"])

    def test_the_horizon_is_bounded(self):
        frame = period_frame([float(index) for index in range(24)])

        for horizon in [0, -1, forecasting.MAX_HORIZON + 1]:
            with self.subTest(horizon=horizon):
                result = forecasting.baseline_forecast(
                    frame,
                    horizon=horizon
                )

                self.assertFalse(result["success"])

    def test_one_row_is_produced_per_period_ahead(self):
        frame = period_frame([float(index) for index in range(24)])

        result = forecasting.baseline_forecast(frame, horizon=4)

        self.assertTrue(result["success"])
        self.assertEqual(len(result["forecast"]), 4)
        self.assertEqual(
            list(result["forecast"]["steps_ahead"]),
            [1, 2, 3, 4]
        )

    def test_the_projected_periods_continue_the_calendar(self):
        frame = period_frame(
            [float(index) for index in range(24)],
            start="2022-01-01"
        )

        result = forecasting.baseline_forecast(frame, horizon=2)

        self.assertEqual(
            list(
                result["forecast"]["period_start"]
                .dt.strftime("%Y-%m-%d")
            ),
            ["2024-01-01", "2024-02-01"]
        )

    def test_a_clean_trend_is_extrapolated_accurately(self):
        values = [100 + 5 * index for index in range(24)]

        result = forecasting.baseline_forecast(
            period_frame(values),
            horizon=3
        )

        expected_next = 100 + 5 * 24

        self.assertAlmostEqual(
            result["forecast"]["forecast"].iloc[0],
            expected_next,
            delta=6
        )

    def test_the_band_widens_with_distance(self):
        rng = np.random.default_rng(7)
        values = [
            100 + 2 * index + noise
            for index, noise in enumerate(rng.normal(0, 8, 30))
        ]

        result = forecasting.baseline_forecast(
            period_frame(values),
            horizon=6
        )

        widths = (
            result["forecast"]["upper"] - result["forecast"]["lower"]
        )

        self.assertGreater(widths.iloc[-1], widths.iloc[0])

    def test_a_perfectly_straight_series_has_no_band(self):
        # Daily periods, because months are not evenly spaced in days: a
        # series that rises by a fixed amount per month is very slightly
        # curved against a calendar axis, so it keeps a small spread.
        values = [float(10 * index) for index in range(14)]

        result = forecasting.baseline_forecast(
            period_frame(values, freq="D"),
            grain="day",
            horizon=2
        )

        self.assertEqual(result["scale"], 0.0)
        self.assertTrue(
            any("no uncertainty band" in note for note in result["notes"])
        )

    def test_a_monthly_straight_line_keeps_a_small_spread(self):
        # Documents the consequence of fitting on calendar days rather
        # than on row numbers.
        values = [float(10 * index) for index in range(24)]

        result = forecasting.baseline_forecast(
            period_frame(values),
            horizon=2
        )

        self.assertGreater(result["scale"], 0.0)
        self.assertLess(result["scale"], 1.0)

    def test_seasonality_shows_up_in_the_projection(self):
        # The fixture has a June trough. Three years from January 2022
        # ends in December 2024, so a six month horizon covers January to
        # June 2025 and June should be projected below March.
        values = seasonal_values(36)

        result = forecasting.baseline_forecast(
            period_frame(values),
            horizon=6
        )

        self.assertTrue(result["seasonality_applied"])

        projected = dict(zip(
            result["forecast"]["period_start"].dt.month,
            result["forecast"]["forecast"],
            strict=True
        ))

        self.assertLess(projected[6], projected[3])

    def test_a_full_year_horizon_reaches_the_seasonal_peak(self):
        values = seasonal_values(36)

        result = forecasting.baseline_forecast(
            period_frame(values),
            horizon=12
        )

        projected = dict(zip(
            result["forecast"]["period_start"].dt.month,
            result["forecast"]["forecast"],
            strict=True
        ))

        self.assertGreater(projected[12], projected[3])
        self.assertGreater(projected[12], projected[6])

    def test_seasonality_can_be_switched_off(self):
        result = forecasting.baseline_forecast(
            period_frame(seasonal_values(36)),
            horizon=3,
            use_seasonality=False
        )

        self.assertFalse(result["seasonality_applied"])

    def test_gaps_do_not_stop_a_projection(self):
        values = [float(index) for index in range(24)]
        values[5] = np.nan
        values[11] = np.nan

        result = forecasting.baseline_forecast(
            period_frame(values),
            horizon=3
        )

        self.assertTrue(result["success"])


class BacktestTest(unittest.TestCase):

    def test_a_clean_trend_beats_assuming_no_change(self):
        values = [100 + 5 * index for index in range(36)]

        result = forecasting.backtest(
            period_frame(values),
            horizon=3
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["beat_naive"])
        self.assertGreater(result["skill"], 0)
        self.assertIn("more accurate", result["verdict"])

    def test_a_step_change_is_better_served_by_no_change(self):
        # After a level shift the trendline keeps climbing while the last
        # observed value is exactly right. The report has to admit that.
        values = [100.0] * 14 + [200.0] * 14

        result = forecasting.backtest(
            period_frame(values),
            horizon=3
        )

        self.assertTrue(result["success"])
        self.assertFalse(result["beat_naive"])
        self.assertIn("no better than assuming no change", result["verdict"])

    def test_the_losing_verdict_tells_the_reader_what_to_do(self):
        values = [100.0] * 14 + [200.0] * 14

        result = forecasting.backtest(period_frame(values), horizon=3)

        self.assertIn("last observed value", result["verdict"])

    def test_folds_are_reported_individually(self):
        values = [100 + 5 * index for index in range(36)]

        result = forecasting.backtest(
            period_frame(values),
            horizon=3,
            folds=3
        )

        self.assertEqual(result["fold_count"], 3)

        for fold in result["folds"]:
            self.assertIn("model_error", fold)
            self.assertIn("naive_error", fold)
            self.assertIn("first_tested_period", fold)

    def test_each_fold_trains_on_less_history(self):
        values = [100 + 5 * index for index in range(36)]

        result = forecasting.backtest(
            period_frame(values),
            horizon=3,
            folds=3
        )

        trained = [fold["trained_on"] for fold in result["folds"]]

        self.assertEqual(trained, sorted(trained, reverse=True))

    def test_a_short_series_is_refused_with_the_requirement(self):
        result = forecasting.backtest(
            period_frame([1.0] * 7),
            horizon=3
        )

        self.assertFalse(result["success"])
        self.assertIn("not enough history", result["error"])

    def test_the_test_never_sees_the_periods_it_predicts(self):
        # A backtest that trained on the answer would always win.
        values = [100 + 5 * index for index in range(24)]

        result = forecasting.backtest(
            period_frame(values),
            horizon=3,
            folds=1
        )

        self.assertEqual(
            result["folds"][0]["trained_on"],
            len(values) - 3
        )


if __name__ == "__main__":
    unittest.main()
