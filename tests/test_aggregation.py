"""
Period aggregation, and the two adjustments that keep a timeline honest.
"""

import unittest

import numpy as np
import pandas as pd

import aggregation


def daily_frame(days: int, start: str = "2024-01-01", value: float = 10.0):
    """One row per day, every day, with a constant measure."""

    dates = pd.date_range(start, periods=days, freq="D")

    return pd.DataFrame({
        "when": dates,
        "amount": [value] * days,
    })


class GrainDetectionTest(unittest.TestCase):

    def grain_for_span(self, days):
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        dates = pd.Series([
            dates[0],
            dates[0] + pd.Timedelta(days=days),
        ])

        return aggregation.detect_grain(dates)

    def test_a_short_span_reads_daily(self):
        self.assertEqual(self.grain_for_span(20), "day")

    def test_a_few_months_reads_weekly(self):
        self.assertEqual(self.grain_for_span(90), "week")

    def test_a_year_or_two_reads_monthly(self):
        self.assertEqual(self.grain_for_span(500), "month")

    def test_several_years_reads_quarterly(self):
        self.assertEqual(self.grain_for_span(1500), "quarter")

    def test_a_long_history_reads_yearly(self):
        self.assertEqual(self.grain_for_span(4000), "year")

    def test_an_empty_column_falls_back_to_monthly(self):
        self.assertEqual(
            aggregation.detect_grain(pd.Series([], dtype="datetime64[ns]")),
            "month"
        )


class PeriodBoundaryTest(unittest.TestCase):

    def test_month_starts_are_aligned(self):
        dates = pd.Series(pd.to_datetime([
            "2024-03-01", "2024-03-17", "2024-03-31", "2024-04-01",
        ]))

        starts = aggregation.period_starts(dates, "month")

        self.assertEqual(
            list(starts.dt.strftime("%Y-%m-%d")),
            ["2024-03-01", "2024-03-01", "2024-03-01", "2024-04-01"]
        )

    def test_a_month_ends_on_its_last_day(self):
        self.assertEqual(
            aggregation.period_end("2024-02-01", "month").strftime("%Y-%m-%d"),
            "2024-02-29"
        )

    def test_a_quarter_ends_on_its_last_day(self):
        self.assertEqual(
            aggregation.period_end("2024-07-01", "quarter").strftime("%Y-%m-%d"),
            "2024-09-30"
        )

    def test_a_month_with_data_to_its_end_is_complete(self):
        self.assertFalse(
            aggregation.final_period_is_incomplete(
                "2024-03-01",
                "2024-03-31",
                "month"
            )
        )

    def test_a_month_with_data_stopping_early_is_incomplete(self):
        self.assertTrue(
            aggregation.final_period_is_incomplete(
                "2024-03-01",
                "2024-03-11",
                "month"
            )
        )

    def test_a_day_at_midnight_is_not_called_incomplete(self):
        # Timestamps usually sit at midnight. Comparing at finer
        # resolution would mark every daily series unfinished.
        self.assertFalse(
            aggregation.final_period_is_incomplete(
                "2024-03-11",
                "2024-03-11",
                "day"
            )
        )


class AggregationTest(unittest.TestCase):

    def test_a_total_per_month_is_correct(self):
        frame = daily_frame(90, start="2024-01-01", value=2.0)

        result = aggregation.aggregate_periods(
            frame,
            "when",
            "amount",
            aggregation="sum",
            grain="month"
        )

        self.assertTrue(result["success"])

        totals = dict(zip(
            result["periods"]["period_label"],
            result["periods"]["value"],
            strict=True
        ))

        self.assertEqual(totals["Jan 2024"], 62.0)
        self.assertEqual(totals["Feb 2024"], 58.0)

    def test_row_counts_are_reported_alongside_the_value(self):
        frame = daily_frame(31, start="2024-01-01")

        result = aggregation.aggregate_periods(
            frame, "when", "amount", grain="month"
        )

        self.assertEqual(result["periods"]["row_count"].iloc[0], 31)

    def test_counting_rows_needs_no_measure(self):
        frame = daily_frame(60)

        result = aggregation.aggregate_periods(
            frame, "when", grain="month"
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["counting_rows"])
        self.assertEqual(result["periods"]["value"].iloc[0], 31)

    def test_an_average_is_not_a_total(self):
        frame = daily_frame(31, value=4.0)

        result = aggregation.aggregate_periods(
            frame, "when", "amount", aggregation="mean", grain="month"
        )

        self.assertEqual(result["periods"]["value"].iloc[0], 4.0)

    def test_the_grain_is_detected_when_not_given(self):
        frame = daily_frame(500)

        result = aggregation.aggregate_periods(frame, "when", "amount")

        self.assertEqual(result["grain"], "month")

    def test_unreadable_dates_are_excluded_and_counted(self):
        frame = daily_frame(40)
        # Written as text, the way a real export arrives.
        frame["when"] = frame["when"].astype(str)
        frame.loc[0:4, "when"] = "not a date"

        result = aggregation.aggregate_periods(
            frame, "when", "amount", grain="month"
        )

        self.assertEqual(result["unreadable_dates"], 5)
        self.assertTrue(
            any("could not be read" in note for note in result["notes"])
        )

    def test_non_numeric_measures_are_excluded_and_counted(self):
        frame = daily_frame(40)
        frame["amount"] = frame["amount"].astype(object)
        frame.loc[0:2, "amount"] = "n/a"

        result = aggregation.aggregate_periods(
            frame, "when", "amount", grain="month"
        )

        self.assertEqual(result["unusable_measure"], 3)

    def test_a_missing_column_is_refused_by_name(self):
        frame = daily_frame(10)

        result = aggregation.aggregate_periods(frame, "nope", "amount")

        self.assertFalse(result["success"])
        self.assertIn("nope", result["error"])

    def test_an_unsupported_aggregation_lists_the_alternatives(self):
        frame = daily_frame(10)

        result = aggregation.aggregate_periods(
            frame, "when", "amount", aggregation="geometric_mean"
        )

        self.assertFalse(result["success"])
        self.assertIn("sum", result["error"])

    def test_a_column_with_no_readable_dates_is_refused(self):
        frame = pd.DataFrame({
            "when": ["x", "y", "z"],
            "amount": [1.0, 2.0, 3.0],
        })

        result = aggregation.aggregate_periods(frame, "when", "amount")

        self.assertFalse(result["success"])
        self.assertIn("No usable dates", result["error"])


class GapFillingTest(unittest.TestCase):

    def frame_with_a_hole(self):
        # January and March have rows. February has none.
        dates = list(pd.date_range("2024-01-01", periods=5, freq="D"))
        dates += list(pd.date_range("2024-03-01", periods=5, freq="D"))

        return pd.DataFrame({
            "when": dates,
            "amount": [1.0] * 10,
        })

    def test_a_missing_month_becomes_zero_for_a_total(self):
        result = aggregation.aggregate_periods(
            self.frame_with_a_hole(),
            "when",
            "amount",
            aggregation="sum",
            grain="month",
            drop_incomplete_final=False
        )

        labels = list(result["periods"]["period_label"])
        values = list(result["periods"]["value"])

        self.assertEqual(labels, ["Jan 2024", "Feb 2024", "Mar 2024"])
        self.assertEqual(values, [5.0, 0.0, 5.0])
        self.assertEqual(result["zero_filled_periods"], 1)

    def test_a_missing_month_becomes_zero_for_a_count(self):
        result = aggregation.aggregate_periods(
            self.frame_with_a_hole(),
            "when",
            aggregation="count",
            grain="month",
            drop_incomplete_final=False
        )

        self.assertEqual(list(result["periods"]["value"]), [5, 0, 5])

    def test_a_missing_month_is_left_empty_for_an_average(self):
        # A month with no rows has no average. Filling zero would invent
        # a data point that never existed.
        result = aggregation.aggregate_periods(
            self.frame_with_a_hole(),
            "when",
            "amount",
            aggregation="mean",
            grain="month",
            drop_incomplete_final=False
        )

        values = list(result["periods"]["value"])

        self.assertEqual(values[0], 1.0)
        self.assertTrue(pd.isna(values[1]))
        self.assertEqual(result["gaps_left_empty"], 1)
        self.assertEqual(result["zero_filled_periods"], 0)

    def test_the_filling_is_explained_in_the_notes(self):
        result = aggregation.aggregate_periods(
            self.frame_with_a_hole(),
            "when",
            "amount",
            aggregation="sum",
            grain="month",
            drop_incomplete_final=False
        )

        self.assertTrue(
            any("shown as zero" in note for note in result["notes"])
        )

    def test_filling_can_be_switched_off(self):
        result = aggregation.aggregate_periods(
            self.frame_with_a_hole(),
            "when",
            "amount",
            grain="month",
            fill_gaps=False,
            drop_incomplete_final=False
        )

        self.assertEqual(result["period_count"], 2)


class IncompleteFinalPeriodTest(unittest.TestCase):

    def frame_stopping_mid_month(self):
        # Three full months, then eleven days of the fourth.
        dates = list(pd.date_range("2024-01-01", "2024-03-31", freq="D"))
        dates += list(pd.date_range("2024-04-01", periods=11, freq="D"))

        return pd.DataFrame({
            "when": dates,
            "amount": [1.0] * len(dates),
        })

    def test_the_unfinished_month_is_removed(self):
        result = aggregation.aggregate_periods(
            self.frame_stopping_mid_month(),
            "when",
            "amount",
            aggregation="sum",
            grain="month"
        )

        labels = list(result["periods"]["period_label"])

        self.assertNotIn("Apr 2024", labels)
        self.assertTrue(result["excluded_final_period"])
        self.assertEqual(result["excluded_period_label"], "Apr 2024")
        self.assertEqual(result["excluded_period_rows"], 11)

    def test_the_removal_is_explained(self):
        result = aggregation.aggregate_periods(
            self.frame_stopping_mid_month(),
            "when",
            "amount",
            grain="month"
        )

        self.assertTrue(
            any("still in progress" in note for note in result["notes"])
        )

    def test_the_last_full_month_is_kept(self):
        result = aggregation.aggregate_periods(
            self.frame_stopping_mid_month(),
            "when",
            "amount",
            aggregation="sum",
            grain="month"
        )

        self.assertEqual(
            result["periods"]["period_label"].iloc[-1],
            "Mar 2024"
        )
        self.assertEqual(result["periods"]["value"].iloc[-1], 31.0)

    def test_a_complete_final_month_is_kept(self):
        frame = pd.DataFrame({
            "when": pd.date_range("2024-01-01", "2024-03-31", freq="D"),
            "amount": 1.0,
        })

        result = aggregation.aggregate_periods(
            frame, "when", "amount", grain="month"
        )

        self.assertFalse(result["excluded_final_period"])
        self.assertEqual(
            result["periods"]["period_label"].iloc[-1],
            "Mar 2024"
        )

    def test_a_very_short_series_keeps_its_partial_period(self):
        # Removing it would leave almost nothing to show, so it stays and
        # the reader is told instead.
        dates = list(pd.date_range("2024-01-01", "2024-01-31", freq="D"))
        dates += list(pd.date_range("2024-02-01", periods=3, freq="D"))

        frame = pd.DataFrame({"when": dates, "amount": 1.0})

        result = aggregation.aggregate_periods(
            frame, "when", "amount", grain="month"
        )

        self.assertFalse(result["excluded_final_period"])
        self.assertTrue(
            any("too little history" in note for note in result["notes"])
        )

    def test_the_exclusion_can_be_switched_off(self):
        result = aggregation.aggregate_periods(
            self.frame_stopping_mid_month(),
            "when",
            "amount",
            grain="month",
            drop_incomplete_final=False
        )

        self.assertIn(
            "Apr 2024",
            list(result["periods"]["period_label"])
        )


class ColumnDiscoveryTest(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(5)
        rows = 60

        self.frame = pd.DataFrame({
            "order_date": pd.date_range(
                "2024-01-01",
                periods=rows,
                freq="D"
            ),
            "shipped_on": pd.date_range(
                "2024-01-02",
                periods=rows,
                freq="D"
            ).astype(str),
            "revenue": rng.normal(100, 10, rows),
            "units": rng.integers(1, 9, rows),
            "region": rng.choice(["north", "south"], rows),
            "flag": rng.choice([True, False], rows),
            "constant": 1,
            "year_number": [2024] * rows,
        })

    def test_real_date_columns_are_found(self):
        found = aggregation.usable_date_columns(self.frame)

        self.assertIn("order_date", found)
        self.assertIn("shipped_on", found)

    def test_text_and_numbers_are_not_mistaken_for_dates(self):
        found = aggregation.usable_date_columns(self.frame)

        self.assertNotIn("region", found)
        self.assertNotIn("revenue", found)

    def test_a_bare_year_number_is_not_treated_as_a_date(self):
        # A column of 2024s is far more likely to be a measure or a label
        # than a timeline.
        found = aggregation.usable_date_columns(self.frame)

        self.assertNotIn("year_number", found)

    def test_numeric_measures_are_found(self):
        found = aggregation.usable_measure_columns(self.frame)

        self.assertIn("revenue", found)
        self.assertIn("units", found)

    def test_booleans_and_constants_are_not_measures(self):
        found = aggregation.usable_measure_columns(self.frame)

        self.assertNotIn("flag", found)
        self.assertNotIn("constant", found)

    def test_the_chosen_date_column_can_be_excluded(self):
        found = aggregation.usable_measure_columns(
            self.frame,
            exclude=("revenue",)
        )

        self.assertNotIn("revenue", found)


class TrendReadinessTest(unittest.TestCase):

    def test_a_long_series_is_ready_for_a_trend(self):
        result = aggregation.aggregate_periods(
            daily_frame(400), "when", "amount", grain="month"
        )

        self.assertTrue(result["enough_for_trend"])

    def test_a_two_period_series_is_not(self):
        frame = pd.DataFrame({
            "when": pd.date_range("2024-01-01", "2024-02-29", freq="D"),
            "amount": 1.0,
        })

        result = aggregation.aggregate_periods(
            frame, "when", "amount", grain="month"
        )

        self.assertFalse(result["enough_for_trend"])


if __name__ == "__main__":
    unittest.main()
