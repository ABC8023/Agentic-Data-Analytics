"""
Segment breakdowns, and the arithmetic that has to tie up.

The central property under test is reconciliation: the parts of a total
must add back to the total, in every period, including the periods with no
rows and the rows with no label. Everything else here is a consequence of
that, or a case where reconciliation is impossible and the module has to
say so instead of guessing.
"""

import unittest

import numpy as np
import pandas as pd

import aggregation
import segments


def segmented_frame(
    days: int = 730,
    weights: dict[str, float] | None = None,
    start: str = "2023-01-01"
) -> pd.DataFrame:
    """
    Build a dated table split into parts with known relative sizes.

    Args:
        days:
            One row per part per day.

        weights:
            Measure value per part. Controls concentration exactly, so a
            test can assert a share rather than approximate it.

        start:
            First date.

    Returns:
        The frame.
    """

    weights = weights or {"north": 60.0, "south": 30.0, "east": 10.0}
    dates = pd.date_range(start, periods=days, freq="D")

    rows = []

    for name, value in weights.items():
        for offset, moment in enumerate(dates):
            rows.append({
                "order_date": moment,
                "region": name,
                "revenue": value,
                # Varies inside every region, so a drill-down has
                # something to find.
                "channel": "web" if offset % 2 else "retail",
                "salesperson": f"REP-{name}-{offset:05d}",
            })

    return pd.DataFrame(rows)


def periods_for(
    frame: pd.DataFrame,
    aggregation_name: str = "sum",
    measure: str | None = "revenue",
    grain: str = "month"
):
    """Run the headline aggregation the breakdown has to agree with."""

    return aggregation.aggregate_periods(
        frame,
        date_column="order_date",
        measure_column=measure,
        aggregation=aggregation_name,
        grain=grain
    )


class SegmentLabelTest(unittest.TestCase):

    def test_missing_values_get_a_named_label(self):
        labels = segments.segment_labels(
            pd.Series(["north", None, "south", np.nan])
        )

        self.assertEqual(
            list(labels),
            [
                "north",
                segments.UNLABELLED_TEXT,
                "south",
                segments.UNLABELLED_TEXT,
            ]
        )

    def test_blank_and_whitespace_strings_count_as_missing(self):
        labels = segments.segment_labels(pd.Series(["", "   ", "north"]))

        self.assertEqual(
            list(labels),
            [
                segments.UNLABELLED_TEXT,
                segments.UNLABELLED_TEXT,
                "north",
            ]
        )

    def test_numeric_labels_survive_as_text(self):
        labels = segments.segment_labels(pd.Series([1, 2, 2]))

        self.assertEqual(list(labels), ["1", "2", "2"])

    def test_no_rows_are_dropped(self):
        values = pd.Series(["a", None, "", "b"])

        self.assertEqual(len(segments.segment_labels(values)), len(values))


class SuitabilityTest(unittest.TestCase):

    def test_a_missing_column_is_refused(self):
        result = segments.describe_segment_suitability(
            segmented_frame(days=5),
            "nope"
        )

        self.assertFalse(result["usable"])
        self.assertIn("no column called", result["reason"])

    def test_a_constant_column_is_refused(self):
        frame = pd.DataFrame({"only": ["x"] * 10})

        result = segments.describe_segment_suitability(frame, "only")

        self.assertFalse(result["usable"])
        self.assertIn("same value in every row", result["reason"])

    def test_an_identifier_column_is_refused(self):
        frame = pd.DataFrame({
            "reference": [f"REF-{index}" for index in range(50)]
        })

        result = segments.describe_segment_suitability(frame, "reference")

        self.assertFalse(result["usable"])
        self.assertIn("identifies rows", result["reason"])

    def test_an_ordinary_category_is_accepted(self):
        result = segments.describe_segment_suitability(
            segmented_frame(days=30),
            "region"
        )

        self.assertTrue(result["usable"])
        self.assertEqual(result["distinct"], 3)


class MatrixTest(unittest.TestCase):

    def setUp(self):
        self.frame = segmented_frame()
        self.aggregated = periods_for(self.frame)
        self.result = segments.segment_matrix(
            self.frame,
            date="order_date",
            segment="region",
            periods=self.aggregated["periods"],
            measure="revenue",
            aggregation_name="sum",
            grain="month"
        )

    def test_the_matrix_is_built(self):
        self.assertTrue(self.result["success"])
        self.assertEqual(self.result["segment_count"], 3)

    def test_rows_match_the_headline_series_exactly(self):
        self.assertEqual(
            list(self.result["matrix"].index),
            list(self.aggregated["periods"]["period_start"])
        )

    def test_the_parts_add_back_to_every_period_total(self):
        summed = self.result["matrix"].sum(axis=1).to_numpy()
        headline = self.aggregated["periods"]["value"].to_numpy()

        np.testing.assert_allclose(summed, headline)
        self.assertTrue(self.result["reconciles"])

    def test_the_period_labels_line_up_with_the_rows(self):
        self.assertEqual(
            self.result["period_labels"],
            list(self.aggregated["periods"]["period_label"])
        )

    def test_a_bad_aggregation_is_refused(self):
        result = segments.segment_matrix(
            self.frame,
            date="order_date",
            segment="region",
            periods=self.aggregated["periods"],
            measure="revenue",
            aggregation_name="geometric_mean"
        )

        self.assertFalse(result["success"])

    def test_a_missing_measure_column_is_refused(self):
        result = segments.segment_matrix(
            self.frame,
            date="order_date",
            segment="region",
            periods=self.aggregated["periods"],
            measure="nope"
        )

        self.assertFalse(result["success"])

    def test_a_missing_date_column_is_refused(self):
        result = segments.segment_matrix(
            self.frame,
            date="nope",
            segment="region",
            periods=self.aggregated["periods"],
            measure="revenue"
        )

        self.assertFalse(result["success"])


class UnlabelledRowsTest(unittest.TestCase):
    """
    Rows with no segment are the classic way a breakdown stops adding up.
    """

    def setUp(self):
        self.frame = segmented_frame(days=365)
        self.frame.loc[0:199, "region"] = None

        self.aggregated = periods_for(self.frame)
        self.result = segments.segment_matrix(
            self.frame,
            date="order_date",
            segment="region",
            periods=self.aggregated["periods"],
            measure="revenue",
            aggregation_name="sum",
            grain="month"
        )

    def test_unlabelled_rows_are_counted(self):
        self.assertEqual(self.result["unlabelled_rows"], 200)

    def test_unlabelled_rows_become_a_part_rather_than_a_gap(self):
        self.assertIn(
            segments.UNLABELLED_TEXT,
            self.result["matrix"].columns
        )

    def test_the_total_still_reconciles(self):
        np.testing.assert_allclose(
            self.result["matrix"].sum(axis=1).to_numpy(),
            self.aggregated["periods"]["value"].to_numpy()
        )
        self.assertTrue(self.result["reconciles"])


class EmptyPeriodTest(unittest.TestCase):
    """
    A zero-filled period in the headline series needs a zero-filled row in
    the breakdown, or the two disagree.
    """

    def setUp(self):
        frame = segmented_frame(days=400)
        gap = (
            (frame["order_date"] >= "2023-05-01")
            & (frame["order_date"] < "2023-06-01")
        )
        self.frame = frame[~gap].reset_index(drop=True)

        self.aggregated = periods_for(self.frame)
        self.result = segments.segment_matrix(
            self.frame,
            date="order_date",
            segment="region",
            periods=self.aggregated["periods"],
            measure="revenue",
            aggregation_name="sum",
            grain="month"
        )

    def test_the_headline_series_zero_filled_the_gap(self):
        self.assertGreater(self.aggregated["zero_filled_periods"], 0)

    def test_the_breakdown_zero_fills_the_same_period(self):
        row = self.result["matrix"].loc[pd.Timestamp("2023-05-01")]

        self.assertEqual(float(row.sum()), 0.0)

    def test_reconciliation_survives_the_gap(self):
        self.assertTrue(self.result["reconciles"])


class PerDayRateTest(unittest.TestCase):
    """
    A per-day rate divides by the calendar. Every part has to divide by
    the same number of days or the parts stop adding to the total.
    """

    def setUp(self):
        self.frame = segmented_frame()
        self.aggregated = periods_for(self.frame, "sum_per_day")
        self.result = segments.segment_matrix(
            self.frame,
            date="order_date",
            segment="region",
            periods=self.aggregated["periods"],
            measure="revenue",
            aggregation_name="sum_per_day",
            grain="month"
        )

    def test_a_rate_still_reconciles(self):
        np.testing.assert_allclose(
            self.result["matrix"].sum(axis=1).to_numpy(),
            self.aggregated["periods"]["value"].to_numpy()
        )
        self.assertTrue(self.result["reconciles"])

    def test_the_rate_is_smaller_than_the_total(self):
        total = periods_for(self.frame, "sum")

        self.assertLess(
            float(self.aggregated["periods"]["value"].iloc[0]),
            float(total["periods"]["value"].iloc[0])
        )


class CountingRowsTest(unittest.TestCase):

    def setUp(self):
        self.frame = segmented_frame(days=365)
        self.frame.loc[0:99, "revenue"] = np.nan

        self.aggregated = periods_for(self.frame, "count", measure=None)
        self.result = segments.segment_matrix(
            self.frame,
            date="order_date",
            segment="region",
            periods=self.aggregated["periods"],
            measure=None,
            aggregation_name="count",
            grain="month"
        )

    def test_counting_keeps_rows_with_an_unusable_measure(self):
        self.assertEqual(
            self.result["rows_used"],
            int(len(self.frame))
        )

    def test_counts_reconcile(self):
        np.testing.assert_allclose(
            self.result["matrix"].sum(axis=1).to_numpy(),
            self.aggregated["periods"]["value"].to_numpy()
        )


class NonAdditiveTest(unittest.TestCase):
    """
    An average of the parts is not the average of the whole, and the
    module has to admit that rather than present a false decomposition.
    """

    def setUp(self):
        self.frame = segmented_frame()
        self.aggregated = periods_for(self.frame, "mean")
        self.result = segments.segment_matrix(
            self.frame,
            date="order_date",
            segment="region",
            periods=self.aggregated["periods"],
            measure="revenue",
            aggregation_name="mean",
            grain="month"
        )

    def test_the_matrix_is_still_built(self):
        self.assertTrue(self.result["success"])

    def test_it_is_marked_as_not_additive(self):
        self.assertFalse(self.result["additive"])

    def test_it_does_not_claim_to_reconcile(self):
        self.assertFalse(self.result["reconciles"])
        self.assertIn("not expected", self.result["reconciliation_note"])

    def test_an_average_of_the_parts_is_not_the_whole(self):
        # Three parts averaging 60, 30 and 10 give a mean of 100 when
        # summed, against a true overall mean of about 33.
        self.assertNotAlmostEqual(
            float(self.result["matrix"].iloc[0].sum()),
            float(self.aggregated["periods"]["value"].iloc[0]),
            places=3
        )

    def test_missing_cells_are_left_empty_not_zeroed(self):
        frame = segmented_frame(days=400)
        gone = (
            (frame["region"] == "east")
            & (frame["order_date"] >= "2023-05-01")
            & (frame["order_date"] < "2023-06-01")
        )
        frame = frame[~gone].reset_index(drop=True)

        aggregated = periods_for(frame, "mean")
        result = segments.segment_matrix(
            frame,
            date="order_date",
            segment="region",
            periods=aggregated["periods"],
            measure="revenue",
            aggregation_name="mean",
            grain="month"
        )

        cell = result["matrix"].loc[pd.Timestamp("2023-05-01"), "east"]

        self.assertTrue(pd.isna(cell))


class BreakdownTest(unittest.TestCase):

    def setUp(self):
        frame = segmented_frame()
        aggregated = periods_for(frame)
        self.matrix = segments.segment_matrix(
            frame,
            date="order_date",
            segment="region",
            periods=aggregated["periods"],
            measure="revenue",
            aggregation_name="sum",
            grain="month"
        )["matrix"]

        self.breakdown = segments.segment_breakdown(
            self.matrix,
            additive=True
        )

    def test_parts_are_ranked_largest_first(self):
        self.assertEqual(
            list(self.breakdown["table"]["segment"]),
            ["north", "south", "east"]
        )

    def test_the_largest_share_is_the_designed_weight(self):
        self.assertAlmostEqual(
            self.breakdown["largest_share"],
            0.6,
            places=6
        )

    def test_shares_sum_to_one(self):
        self.assertAlmostEqual(
            float(self.breakdown["table"]["share"].sum()),
            1.0,
            places=9
        )

    def test_two_parts_reach_eighty_percent(self):
        self.assertEqual(self.breakdown["parts_for_80_percent"], 2)

    def test_the_top_list_is_capped(self):
        breakdown = segments.segment_breakdown(
            self.matrix,
            additive=True,
            top_count=2
        )

        self.assertEqual(len(breakdown["top"]), 2)


class WithheldSharesTest(unittest.TestCase):

    def matrix_with(self, values):
        index = pd.date_range("2024-01-01", periods=2, freq="MS")

        return pd.DataFrame(values, index=index)

    def test_a_negative_part_withholds_shares(self):
        matrix = self.matrix_with({
            "profit_centre": [100.0, 120.0],
            "loss_centre": [-40.0, -50.0],
        })

        breakdown = segments.segment_breakdown(matrix, additive=True)

        self.assertFalse(breakdown["shares_meaningful"])
        self.assertIsNone(breakdown["largest_share"])
        self.assertIn("negative", breakdown["shares_withheld_because"])

    def test_a_zero_total_withholds_shares(self):
        matrix = self.matrix_with({
            "a": [0.0, 0.0],
            "b": [0.0, 0.0],
        })

        breakdown = segments.segment_breakdown(matrix, additive=True)

        self.assertFalse(breakdown["shares_meaningful"])
        self.assertIn("zero", breakdown["shares_withheld_because"])

    def test_a_non_additive_aggregation_withholds_shares(self):
        matrix = self.matrix_with({
            "a": [10.0, 12.0],
            "b": [20.0, 22.0],
        })

        breakdown = segments.segment_breakdown(matrix, additive=False)

        self.assertFalse(breakdown["shares_meaningful"])
        self.assertIn("add up", breakdown["shares_withheld_because"])

    def test_the_totals_are_still_reported(self):
        matrix = self.matrix_with({
            "a": [10.0, 12.0],
            "b": [20.0, 22.0],
        })

        breakdown = segments.segment_breakdown(matrix, additive=False)

        self.assertEqual(breakdown["largest_segment"], "b")
        self.assertAlmostEqual(
            float(breakdown["table"]["total"].iloc[0]),
            42.0
        )


class WaterfallTest(unittest.TestCase):

    def build(self, values, totals=None):
        index = pd.date_range("2024-01-01", periods=2, freq="MS")
        matrix = pd.DataFrame(values, index=index)

        if totals is None:
            totals = matrix.sum(axis=1).to_numpy()

        periods = pd.DataFrame({
            "period_start": index,
            "value": totals,
            "period_days": [31, 29],
            "period_label": ["Jan 2024", "Feb 2024"],
        })

        return segments.segment_waterfall(
            matrix,
            periods,
            additive=True,
            grain="month"
        )

    def test_the_parts_account_for_the_whole_change(self):
        result = self.build({
            "north": [100.0, 80.0],
            "south": [50.0, 55.0],
        })

        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["total_change"], -15.0)
        self.assertAlmostEqual(result["accounted_change"], -15.0)
        self.assertAlmostEqual(result["residual"], 0.0)
        self.assertTrue(result["reconciles"])

    def test_the_biggest_mover_comes_first(self):
        result = self.build({
            "north": [100.0, 80.0],
            "south": [50.0, 55.0],
        })

        self.assertEqual(result["table"]["segment"].iloc[0], "north")

    def test_contributions_are_relative_to_the_net_change(self):
        result = self.build({
            "north": [100.0, 80.0],
            "south": [50.0, 55.0],
        })

        table = result["table"].set_index("segment")

        self.assertAlmostEqual(
            float(table.loc["north", "contribution"]),
            20.0 / 15.0,
            places=6
        )

    def test_offsetting_movements_are_flagged(self):
        result = self.build({
            "north": [100.0, 80.0],
            "south": [50.0, 55.0],
        })

        self.assertTrue(result["offsetting_movements"])
        self.assertEqual(result["parts_moved_up"], 1)
        self.assertEqual(result["parts_moved_down"], 1)

    def test_a_part_that_stopped_recording_is_flagged(self):
        result = self.build({
            "north": [100.0, 100.0],
            "south": [40.0, 0.0],
        })

        table = result["table"].set_index("segment")

        self.assertTrue(bool(table.loc["south", "exited"]))
        self.assertEqual(result["exited_count"], 1)

    def test_a_new_part_is_flagged(self):
        result = self.build({
            "north": [100.0, 100.0],
            "south": [0.0, 40.0],
        })

        table = result["table"].set_index("segment")

        self.assertTrue(bool(table.loc["south", "entered"]))
        self.assertEqual(result["entered_count"], 1)

    def test_a_residual_is_reported_rather_than_hidden(self):
        result = self.build(
            {"north": [100.0, 80.0], "south": [50.0, 55.0]},
            totals=[150.0, 200.0]
        )

        self.assertAlmostEqual(result["total_change"], 50.0)
        self.assertAlmostEqual(result["accounted_change"], -15.0)
        self.assertAlmostEqual(result["residual"], 65.0)
        self.assertFalse(result["reconciles"])
        self.assertFalse(result["explains_total"])

    def test_one_period_cannot_be_compared(self):
        index = pd.date_range("2024-01-01", periods=1, freq="MS")
        matrix = pd.DataFrame({"north": [10.0]}, index=index)
        periods = pd.DataFrame({
            "period_start": index,
            "value": [10.0],
            "period_days": [31],
            "period_label": ["Jan 2024"],
        })

        result = segments.segment_waterfall(matrix, periods, additive=True)

        self.assertFalse(result["success"])
        self.assertIn("only one period", result["error"])

    def test_a_missing_period_value_stops_the_split(self):
        index = pd.date_range("2024-01-01", periods=2, freq="MS")
        matrix = pd.DataFrame(
            {"north": [10.0, 12.0]},
            index=index
        )
        periods = pd.DataFrame({
            "period_start": index,
            "value": [10.0, np.nan],
            "period_days": [31, 29],
            "period_label": ["Jan 2024", "Feb 2024"],
        })

        result = segments.segment_waterfall(matrix, periods, additive=True)

        self.assertFalse(result["success"])

    def test_a_non_additive_waterfall_does_not_claim_to_explain(self):
        index = pd.date_range("2024-01-01", periods=2, freq="MS")
        matrix = pd.DataFrame(
            {"north": [10.0, 12.0], "south": [20.0, 18.0]},
            index=index
        )
        periods = pd.DataFrame({
            "period_start": index,
            "value": [15.0, 15.0],
            "period_days": [31, 29],
            "period_label": ["Jan 2024", "Feb 2024"],
        })

        result = segments.segment_waterfall(
            matrix,
            periods,
            additive=False
        )

        self.assertTrue(result["success"])
        self.assertFalse(result["explains_total"])
        self.assertTrue(result["table"]["contribution"].isna().all())


class WaterfallStepTest(unittest.TestCase):

    def table_with(self, count):
        return pd.DataFrame({
            "segment": [f"part_{number:02d}" for number in range(count)],
            "change": [float(count - number) for number in range(count)],
        })

    def test_a_short_table_is_shown_in_full(self):
        result = segments.waterfall_steps(self.table_with(4), limit=8)

        self.assertEqual(result["grouped_count"], 0)
        self.assertEqual(len(result["steps"]), 4)

    def test_a_long_table_groups_its_tail(self):
        result = segments.waterfall_steps(self.table_with(30), limit=8)

        self.assertEqual(result["grouped_count"], 22)
        self.assertEqual(len(result["steps"]), 9)
        self.assertIn("22", result["other_label"])

    def test_grouping_preserves_the_total_change(self):
        table = self.table_with(30)

        result = segments.waterfall_steps(table, limit=8)

        self.assertAlmostEqual(
            float(result["steps"]["change"].sum()),
            float(table["change"].sum()),
            places=6
        )

    def test_the_largest_movers_are_kept(self):
        result = segments.waterfall_steps(self.table_with(30), limit=3)

        self.assertEqual(
            list(result["steps"]["segment"])[:3],
            ["part_00", "part_01", "part_02"]
        )

    def test_offsetting_movers_are_ranked_by_size_not_sign(self):
        table = pd.DataFrame({
            "segment": ["small", "big_down", "medium"],
            "change": [1.0, -50.0, 10.0],
        })

        result = segments.waterfall_steps(table, limit=3)

        self.assertEqual(
            list(result["steps"]["segment"]),
            ["big_down", "medium", "small"]
        )


class CollapseTest(unittest.TestCase):

    def wide_matrix(self, parts):
        index = pd.date_range("2024-01-01", periods=3, freq="MS")

        return pd.DataFrame(
            {
                f"part_{number:02d}": [float(parts - number)] * 3
                for number in range(parts)
            },
            index=index
        )

    def test_a_narrow_matrix_is_left_alone(self):
        matrix = self.wide_matrix(4)

        result = segments.collapse_matrix(matrix, limit=12)

        self.assertEqual(result["grouped_count"], 0)
        self.assertEqual(list(result["matrix"].columns), [
            "part_00", "part_01", "part_02", "part_03",
        ])

    def test_the_tail_is_grouped(self):
        matrix = self.wide_matrix(20)

        result = segments.collapse_matrix(matrix, limit=5)

        self.assertEqual(result["grouped_count"], 15)
        self.assertEqual(result["matrix"].shape[1], 6)
        self.assertIn("15", result["other_label"])

    def test_grouping_preserves_the_row_totals(self):
        matrix = self.wide_matrix(20)

        result = segments.collapse_matrix(matrix, limit=5)

        np.testing.assert_allclose(
            result["matrix"].sum(axis=1).to_numpy(),
            matrix.sum(axis=1).to_numpy()
        )

    def test_the_kept_parts_are_the_largest(self):
        matrix = self.wide_matrix(20)

        result = segments.collapse_matrix(matrix, limit=3)

        self.assertEqual(
            list(result["matrix"].columns)[:3],
            ["part_00", "part_01", "part_02"]
        )


class PeriodShareTest(unittest.TestCase):

    def test_each_row_shares_out_of_its_own_total(self):
        index = pd.date_range("2024-01-01", periods=2, freq="MS")
        matrix = pd.DataFrame(
            {"a": [30.0, 50.0], "b": [70.0, 50.0]},
            index=index
        )

        shares = segments.segment_period_shares(matrix)

        np.testing.assert_allclose(
            shares.to_numpy(),
            [[0.3, 0.7], [0.5, 0.5]]
        )

    def test_a_zero_total_row_is_left_empty(self):
        index = pd.date_range("2024-01-01", periods=2, freq="MS")
        matrix = pd.DataFrame(
            {"a": [0.0, 50.0], "b": [0.0, 50.0]},
            index=index
        )

        shares = segments.segment_period_shares(matrix)

        self.assertTrue(shares.iloc[0].isna().all())
        self.assertFalse(shares.iloc[1].isna().any())

    def test_a_row_with_a_negative_part_is_left_empty(self):
        index = pd.date_range("2024-01-01", periods=1, freq="MS")
        matrix = pd.DataFrame(
            {"a": [-20.0], "b": [120.0]},
            index=index
        )

        shares = segments.segment_period_shares(matrix)

        self.assertTrue(shares.iloc[0].isna().all())


class DrillDownTest(unittest.TestCase):

    def setUp(self):
        self.frame = segmented_frame(days=365)
        self.aggregated = periods_for(self.frame)

    def test_it_regroups_inside_one_part(self):
        result = segments.drill_down(
            self.frame,
            segment="region",
            value="north",
            next_segment="channel",
            date="order_date",
            periods=self.aggregated["periods"],
            measure="revenue",
            aggregation_name="sum",
            grain="month"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["parent_value"], "north")
        self.assertEqual(
            list(result["matrix"].columns),
            ["retail", "web"]
        )

    def test_the_inner_parts_add_up_to_the_part_itself(self):
        result = segments.drill_down(
            self.frame,
            segment="region",
            value="north",
            next_segment="channel",
            date="order_date",
            periods=self.aggregated["periods"],
            measure="revenue",
            aggregation_name="sum",
            grain="month"
        )

        outer = segments.segment_matrix(
            self.frame,
            date="order_date",
            segment="region",
            periods=self.aggregated["periods"],
            measure="revenue",
            aggregation_name="sum",
            grain="month"
        )["matrix"]

        np.testing.assert_allclose(
            result["matrix"].sum(axis=1).to_numpy(),
            outer["north"].to_numpy()
        )

    def test_a_dimension_with_no_variation_inside_the_part_is_refused(self):
        frame = self.frame.copy()
        frame.loc[frame["region"] == "north", "channel"] = "web"

        result = segments.drill_down(
            frame,
            segment="region",
            value="north",
            next_segment="channel",
            date="order_date",
            periods=self.aggregated["periods"],
            measure="revenue",
            grain="month"
        )

        self.assertFalse(result["success"])
        self.assertIn("north", result["error"])

    def test_an_identifier_dimension_is_refused(self):
        result = segments.drill_down(
            self.frame,
            segment="region",
            value="north",
            next_segment="salesperson",
            date="order_date",
            periods=self.aggregated["periods"],
            measure="revenue",
            grain="month"
        )

        self.assertFalse(result["success"])

    def test_it_reports_how_much_of_the_data_it_covers(self):
        result = segments.drill_down(
            self.frame,
            segment="region",
            value="north",
            next_segment="channel",
            date="order_date",
            periods=self.aggregated["periods"],
            measure="revenue",
            grain="month"
        )

        self.assertAlmostEqual(result["parent_row_share"], 1 / 3, places=6)

    def test_a_drill_down_never_claims_to_tie_to_the_total(self):
        result = segments.drill_down(
            self.frame,
            segment="region",
            value="north",
            next_segment="channel",
            date="order_date",
            periods=self.aggregated["periods"],
            measure="revenue",
            grain="month"
        )

        self.assertFalse(result["reconciles"])
        self.assertIn("only the rows", result["reconciliation_note"])

    def test_an_unknown_part_is_refused(self):
        result = segments.drill_down(
            self.frame,
            segment="region",
            value="atlantis",
            next_segment="channel",
            date="order_date",
            periods=self.aggregated["periods"],
            measure="revenue"
        )

        self.assertFalse(result["success"])
        self.assertIn("atlantis", result["error"])

    def test_a_missing_parent_column_is_refused(self):
        result = segments.drill_down(
            self.frame,
            segment="nope",
            value="north",
            next_segment="channel",
            date="order_date",
            periods=self.aggregated["periods"],
            measure="revenue"
        )

        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
