"""
Running a plan, and the figures that come out.

The dataset here is built so that every answer has a value that can be
worked out by hand. That is the point: a test asserting only that a number
came back would pass just as happily on a wrong number.

Two behaviours get particular attention, because both change answers and
both are easy to get quietly wrong. A relative window is resolved against
the data rather than the calendar, and an unfinished final period is left
out of one.
"""

import unittest

import numpy as np
import pandas as pd

import nlq
import nlq_answer
import nlq_rules


def flat_frame(
    start: str = "2023-01-01",
    days: int = 365,
    value: float = 10.0
) -> pd.DataFrame:
    """
    One row per day with a constant measure, so totals are arithmetic.

    A year of 10.0 per day totals 3,650, and any month totals ten times
    its length in days.
    """

    dates = pd.date_range(start, periods=days, freq="D")
    regions = ["north", "south", "east", "west"]

    return pd.DataFrame({
        "order_date": dates,
        "revenue": [value] * days,
        "units": list(range(1, days + 1)),
        "region": [regions[index % 4] for index in range(days)],
    })


def weighted_frame(weights: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per part per day, with known relative sizes."""

    weights = weights or {"whale": 70.0, "minnow": 20.0, "shrimp": 10.0}
    dates = pd.date_range("2023-01-01", periods=120, freq="D")

    rows = []

    for name, value in weights.items():
        for moment in dates:
            rows.append({
                "order_date": moment,
                "region": name,
                "revenue": value,
            })

    return pd.DataFrame(rows)


def run(question: str, frame: pd.DataFrame) -> nlq_answer.Answer:
    """Parse a question with the rules, then execute it."""

    parsed = nlq_rules.parse_question(question, frame)

    if not parsed.understood:
        raise AssertionError(f"not understood: {parsed.reason}")

    return nlq_answer.execute_plan(frame, parsed.plan)


class WindowResolutionTest(unittest.TestCase):

    def dates(self, start="2023-01-01", days=365):
        return pd.Series(pd.date_range(start, periods=days, freq="D"))

    def test_everything_has_no_bounds(self):
        resolved = nlq_answer.resolve_window(
            self.dates(),
            nlq.Window()
        )

        self.assertTrue(resolved["success"])
        self.assertIsNone(resolved["start"])
        self.assertIsNone(resolved["end"])

    def test_a_relative_window_is_anchored_to_the_data(self):
        # The data ends in 2023. Anchoring to today would return nothing
        # and read as a collapse rather than as a stale extract.
        resolved = nlq_answer.resolve_window(
            self.dates(),
            nlq.Window("last_periods", count=1, grain="month")
        )

        self.assertEqual(resolved["start"].year, 2023)
        self.assertEqual(resolved["start"].month, 12)

    def test_three_months_spans_three_months(self):
        resolved = nlq_answer.resolve_window(
            self.dates(),
            nlq.Window("last_periods", count=3, grain="month")
        )

        self.assertEqual(resolved["start"].month, 10)
        self.assertEqual(resolved["end"].month, 12)

    def test_an_unfinished_final_period_is_left_out_and_reported(self):
        # Four days into January is not a month, and letting it stand as
        # "last month" turns a partial extract into a 90% decline.
        dates = self.dates("2023-01-01", 369)

        resolved = nlq_answer.resolve_window(
            dates,
            nlq.Window("last_periods", count=1, grain="month")
        )

        self.assertEqual(resolved["start"].month, 12)
        self.assertEqual(resolved["start"].year, 2023)
        self.assertTrue(
            any("still in progress" in note for note in resolved["notes"]),
            resolved["notes"]
        )

    def test_an_offset_steps_back_a_further_period(self):
        resolved = nlq_answer.resolve_window(
            self.dates(),
            nlq.Window("last_periods", count=1, grain="month", offset=1)
        )

        self.assertEqual(resolved["start"].month, 11)

    def test_a_named_period_is_used_as_given(self):
        resolved = nlq_answer.resolve_window(
            self.dates(),
            nlq.Window("period", grain="month", start="2023-03-01")
        )

        self.assertEqual(resolved["start"].month, 3)
        self.assertEqual(resolved["end"].month, 3)
        self.assertEqual(resolved["end"].day, 31)

    def test_a_range_includes_the_whole_closing_day(self):
        resolved = nlq_answer.resolve_window(
            self.dates(),
            nlq.Window(
                "between_dates",
                start="2023-01-01",
                end="2023-01-31"
            )
        )

        self.assertEqual(resolved["end"].day, 31)
        self.assertGreater(resolved["end"].hour, 0)

    def test_an_open_ended_range_leaves_the_end_unbounded(self):
        resolved = nlq_answer.resolve_window(
            self.dates(),
            nlq.Window("between_dates", start="2023-06-01")
        )

        self.assertIsNone(resolved["end"])

    def test_no_usable_dates_is_reported(self):
        resolved = nlq_answer.resolve_window(
            pd.Series([pd.NaT, pd.NaT]),
            nlq.Window("last_periods", count=1, grain="month")
        )

        self.assertFalse(resolved["success"])


class FilterExecutionTest(unittest.TestCase):

    def setUp(self):
        self.frame = flat_frame()

    def filtered(self, *filters):
        trail = nlq_answer.Trail()

        return nlq_answer.apply_filters(self.frame, filters, trail), trail

    def test_a_label_filter_is_case_insensitive(self):
        # The parser lowercases what it reads, and the column may not be.
        frame = self.frame.copy()
        frame["region"] = frame["region"].str.capitalize()

        trail = nlq_answer.Trail()
        result = nlq_answer.apply_filters(
            frame,
            (nlq.Filter("region", "equals", ("north",)),),
            trail
        )

        self.assertEqual(len(result), 92)

    def test_a_numeric_filter_keeps_the_right_rows(self):
        result, _ = self.filtered(
            nlq.Filter("units", "greater_than", (360,))
        )

        self.assertEqual(len(result), 5)

    def test_greater_or_equal_includes_the_boundary(self):
        result, _ = self.filtered(
            nlq.Filter("units", "greater_or_equal", (365,))
        )

        self.assertEqual(len(result), 1)

    def test_a_range_is_inclusive_at_both_ends(self):
        result, _ = self.filtered(
            nlq.Filter("units", "between", (10, 20))
        )

        self.assertEqual(len(result), 11)

    def test_not_equals_keeps_everything_else(self):
        result, _ = self.filtered(
            nlq.Filter("region", "not_equals", ("north",))
        )

        self.assertEqual(len(result), 365 - 92)

    def test_in_accepts_several_values(self):
        result, _ = self.filtered(
            nlq.Filter("region", "in", ("north", "south"))
        )

        self.assertEqual(len(result), 183)

    def test_is_missing_finds_the_blanks(self):
        frame = self.frame.copy()
        frame.loc[0:9, "region"] = None

        trail = nlq_answer.Trail()
        result = nlq_answer.apply_filters(
            frame,
            (nlq.Filter("region", "is_missing", ()),),
            trail
        )

        self.assertEqual(len(result), 10)

    def test_a_value_that_is_not_a_number_does_not_match_as_zero(self):
        # Units run 1 to 365, so "below 5" would be four rows. Blanking
        # the first leaves three: treating the blank as zero would put it
        # back and quietly inflate the count.
        frame = self.frame.copy()
        frame["units"] = frame["units"].astype(object)
        frame.loc[0, "units"] = "n/a"

        trail = nlq_answer.Trail()
        result = nlq_answer.apply_filters(
            frame,
            (nlq.Filter("units", "less_than", (5,)),),
            trail
        )

        self.assertEqual(len(result), 3)

    def test_each_filter_is_recorded_with_what_it_removed(self):
        _, trail = self.filtered(
            nlq.Filter("region", "equals", ("north",))
        )

        self.assertEqual(len(trail.steps), 1)
        self.assertIn("rows remain", trail.steps[0]["result"])

    def test_contains_matches_part_of_a_label(self):
        result, _ = self.filtered(
            nlq.Filter("region", "contains", ("or",))
        )

        self.assertEqual(len(result), 92)


class AggregateAnswerTest(unittest.TestCase):

    def setUp(self):
        self.frame = flat_frame()

    def test_a_total_is_arithmetic(self):
        answer = run("total revenue", self.frame)

        self.assertTrue(answer.success, answer.error)
        self.assertAlmostEqual(answer.figures["value"], 3650.0)
        self.assertIn("3,650", answer.headline)

    def test_an_average_is_arithmetic(self):
        answer = run("average revenue", self.frame)

        self.assertAlmostEqual(answer.figures["value"], 10.0)

    def test_a_count_counts_rows(self):
        answer = run("how many orders", self.frame)

        self.assertEqual(answer.figures["value"], 365.0)

    def test_a_window_narrows_the_total(self):
        # December has 31 days at 10.0 each.
        answer = run("total revenue last month", self.frame)

        self.assertAlmostEqual(answer.figures["value"], 310.0)

    def test_a_named_month_narrows_the_total(self):
        # February 2023 has 28 days.
        answer = run("total revenue in february 2023", self.frame)

        self.assertAlmostEqual(answer.figures["value"], 280.0)

    def test_a_filter_narrows_the_total(self):
        answer = run("total revenue in north", self.frame)

        self.assertAlmostEqual(answer.figures["value"], 920.0)

    def test_a_filter_and_a_window_combine(self):
        # December has 31 days; north takes every fourth, so 8 of them.
        answer = run("total revenue in north last month", self.frame)

        self.assertAlmostEqual(answer.figures["value"], 80.0)

    def test_a_per_day_rate_divides_by_the_days_covered(self):
        answer = run("revenue daily rate", self.frame)

        self.assertAlmostEqual(answer.figures["value"], 10.0)

    def test_conditions_matching_nothing_say_so(self):
        plan, _ = nlq.plan_from_dict({
            "intent": "aggregate",
            "measure": "revenue",
            "filters": [{
                "column": "region",
                "operator": "equals",
                "values": ["atlantis"],
            }],
        })

        answer = nlq_answer.execute_plan(self.frame, plan)

        self.assertTrue(answer.success)
        self.assertIn("No rows match", answer.headline)
        self.assertEqual(answer.figures["rows"], 0)

    def test_a_window_with_no_rows_says_so(self):
        plan, _ = nlq.plan_from_dict({
            "intent": "aggregate",
            "measure": "revenue",
            "date": "order_date",
            "window": {
                "kind": "between_dates",
                "start": "2019-01-01",
                "end": "2019-12-31",
            },
        })

        answer = nlq_answer.execute_plan(self.frame, plan)

        self.assertTrue(answer.success)
        self.assertIn("No rows fall in", answer.headline)


class TimeseriesAnswerTest(unittest.TestCase):

    def setUp(self):
        self.frame = flat_frame()

    def test_a_monthly_series_has_twelve_periods(self):
        answer = run("revenue by month", self.frame)

        self.assertTrue(answer.success, answer.error)
        self.assertEqual(answer.figures["period_count"], 12)

    def test_the_table_carries_a_row_per_period(self):
        answer = run("revenue by month", self.frame)

        self.assertEqual(len(answer.table), 12)
        self.assertIn("Period", answer.table.columns)

    def test_a_quarterly_series_has_four_periods(self):
        answer = run("revenue each quarter", self.frame)

        self.assertEqual(answer.figures["period_count"], 4)

    def test_january_totals_ten_times_its_days(self):
        answer = run("revenue by month", self.frame)
        first = answer.table.iloc[0]

        self.assertAlmostEqual(float(first.iloc[1]), 310.0)


class BreakdownAnswerTest(unittest.TestCase):

    def setUp(self):
        self.frame = weighted_frame()

    def test_the_largest_part_is_named(self):
        answer = run("revenue by region", self.frame)

        self.assertTrue(answer.success, answer.error)
        self.assertEqual(answer.figures["leader"], "whale")

    def test_the_share_is_the_designed_weight(self):
        answer = run("revenue by region", self.frame)

        self.assertIn("70.0%", answer.headline)

    def test_the_lowest_direction_reverses_the_ranking(self):
        answer = run("which region is worst for revenue", self.frame)

        self.assertEqual(answer.figures["leader"], "shrimp")

    def test_a_top_count_limits_the_table(self):
        answer = run("top 2 regions by revenue", self.frame)

        self.assertEqual(len(answer.table), 2)

    def test_a_breakdown_works_without_a_timeline(self):
        frame = pd.DataFrame({
            "category": ["brake", "filter", "belt"] * 20,
            "price": [10.0, 20.0, 30.0] * 20,
        })

        answer = run("total price by category", frame)

        self.assertTrue(answer.success, answer.error)
        self.assertEqual(answer.figures["leader"], "belt")

    def test_an_average_breakdown_withholds_shares_and_says_why(self):
        answer = run("average revenue by region", self.frame)

        self.assertTrue(answer.success, answer.error)
        self.assertNotIn("Share", answer.table.columns)
        self.assertTrue(
            any("add up" in note for note in answer.notes),
            answer.notes
        )

    def test_the_parts_add_back_to_the_total(self):
        breakdown = run("revenue by region", self.frame)
        total = run("total revenue", self.frame)

        column = [
            name for name in breakdown.table.columns
            if name.startswith("Total")
        ][0]

        self.assertAlmostEqual(
            float(breakdown.table[column].sum()),
            total.figures["value"]
        )


class ExtremeAnswerTest(unittest.TestCase):

    def test_the_largest_month_is_found(self):
        frame = flat_frame()
        spike = frame["order_date"].dt.month == 7
        frame.loc[spike, "revenue"] = 100.0

        answer = run("which month had the highest revenue", frame)

        self.assertTrue(answer.success, answer.error)
        self.assertIn("Jul", answer.figures["period"])

    def test_the_smallest_month_is_found(self):
        frame = flat_frame()

        answer = run("which month had the lowest revenue", frame)

        self.assertTrue(answer.success, answer.error)
        # February is shortest, so its total is lowest.
        self.assertIn("Feb", answer.figures["period"])

    def test_a_bare_question_measures_whatever_it_resolved_to(self):
        # The fixture's revenue never changes, so schema detection treats
        # it as a constant and picks units instead. The answer is right
        # for the question it actually asked, and the plan says which
        # column that was.
        answer = run("which month was worst", flat_frame())

        self.assertTrue(answer.success, answer.error)
        self.assertEqual(answer.plan.measure, "units")
        self.assertIn("units", answer.plan.describe())

    def test_when_was_lowest_finds_a_period(self):
        answer = run("when was revenue lowest", flat_frame())

        self.assertTrue(answer.success, answer.error)
        self.assertTrue(answer.figures["period"])


class TrendAnswerTest(unittest.TestCase):

    def test_a_rising_series_reads_as_rising(self):
        frame = flat_frame()
        frame["revenue"] = np.linspace(10.0, 200.0, len(frame))

        answer = run("is revenue growing", frame)

        self.assertTrue(answer.success, answer.error)
        self.assertEqual(answer.figures["direction"], "rising")

    def test_a_falling_series_reads_as_falling(self):
        frame = flat_frame()
        frame["revenue"] = np.linspace(200.0, 10.0, len(frame))

        answer = run("is revenue declining", frame)

        self.assertEqual(answer.figures["direction"], "falling")

    def test_a_flat_series_is_not_called_a_trend(self):
        answer = run("is revenue growing", flat_frame())

        self.assertTrue(answer.success, answer.error)
        self.assertFalse(answer.figures["significant"])

    def test_too_little_history_is_refused_with_a_reason(self):
        # Three days is three daily periods, below the minimum a trendline
        # is worth fitting to.
        answer = run("is revenue growing", flat_frame(days=3))

        self.assertFalse(answer.success)
        self.assertTrue(answer.error)


class CompareAnswerTest(unittest.TestCase):

    def setUp(self):
        # Ends on the last day of a month, so nothing is in progress.
        self.frame = flat_frame("2023-01-01", 365)

    def test_two_periods_are_measured_separately(self):
        answer = run(
            "revenue this month compared to last month",
            self.frame
        )

        self.assertTrue(answer.success, answer.error)
        self.assertAlmostEqual(answer.figures["latest"], 310.0)
        self.assertAlmostEqual(answer.figures["previous"], 300.0)

    def test_the_change_is_reported_both_ways(self):
        answer = run(
            "revenue this month compared to last month",
            self.frame
        )

        self.assertAlmostEqual(answer.figures["change"], 10.0)
        self.assertAlmostEqual(
            answer.figures["relative_change"],
            10.0 / 300.0
        )

    def test_the_sentence_names_both_periods(self):
        answer = run(
            "revenue this month compared to last month",
            self.frame
        )

        self.assertIn("Dec 2023", answer.headline)
        self.assertIn("Nov 2023", answer.headline)

    def test_a_zero_baseline_gives_no_percentage(self):
        frame = self.frame.copy()
        november = frame["order_date"].dt.month == 11
        frame.loc[november, "revenue"] = 0.0

        answer = run(
            "revenue this month compared to last month",
            frame
        )

        self.assertIsNone(answer.figures["relative_change"])
        self.assertIn("no percentage", answer.headline)


class ShareAnswerTest(unittest.TestCase):

    def setUp(self):
        self.frame = weighted_frame()

    def test_a_share_is_the_designed_weight(self):
        answer = run("what share of revenue is whale", self.frame)

        self.assertTrue(answer.success, answer.error)
        self.assertAlmostEqual(answer.figures["share"], 0.7, places=6)

    def test_the_sentence_gives_both_figures(self):
        answer = run("what share of revenue is minnow", self.frame)

        self.assertIn("20.0%", answer.headline)
        self.assertIn("of", answer.headline)

    def test_a_part_with_no_rows_in_the_window_is_refused(self):
        plan, errors = nlq.plan_from_dict({
            "intent": "share",
            "measure": "revenue",
            "segment": "region",
            "segment_value": "whale",
            "date": "order_date",
            "window": {
                "kind": "between_dates",
                "start": "2023-01-01",
                "end": "2023-01-02",
            },
        })

        self.assertEqual(errors, [])

        frame = self.frame.copy()
        early = frame["order_date"] <= "2023-01-02"
        frame = frame[~(early & (frame["region"] == "whale"))]

        answer = nlq_answer.execute_plan(frame, plan)

        self.assertFalse(answer.success)
        self.assertIn("no share", answer.error)


class TrailTest(unittest.TestCase):
    """
    The trail is the reason to plan a question rather than ask a model to
    answer it, so every answer has to carry one.
    """

    def setUp(self):
        self.frame = flat_frame()

    def test_the_first_step_states_the_question_as_read(self):
        answer = run("total revenue", self.frame)

        self.assertEqual(answer.trail[0]["action"], "Read the question")
        self.assertIn("revenue", answer.trail[0]["detail"])

    def test_every_step_carries_its_arithmetic(self):
        answer = run("total revenue in north last 3 months", self.frame)

        for step in answer.trail:
            self.assertTrue(step["detail"], step)

    def test_steps_are_numbered_from_one_without_gaps(self):
        answer = run("revenue by region", self.frame)

        self.assertEqual(
            [step["step"] for step in answer.trail],
            list(range(1, len(answer.trail) + 1))
        )

    def test_a_filter_appears_in_the_trail(self):
        answer = run("total revenue in north", self.frame)

        actions = [step["action"] for step in answer.trail]

        self.assertIn("Filtered rows", actions)

    def test_a_window_appears_in_the_trail(self):
        answer = run("total revenue last month", self.frame)

        actions = [step["action"] for step in answer.trail]

        self.assertIn("Selected the period", actions)

    def test_every_intent_produces_a_trail(self):
        questions = [
            "total revenue",
            "revenue by month",
            "revenue by region",
            "which month had the highest revenue",
            "is revenue growing",
            "revenue this month vs last month",
        ]

        for question in questions:
            with self.subTest(question=question):
                answer = run(question, self.frame)

                self.assertTrue(answer.success, answer.error)
                self.assertGreaterEqual(len(answer.trail), 2)

    def test_the_trail_is_summarised_for_a_label(self):
        answer = run("total revenue", self.frame)

        self.assertIn("step", nlq_answer.describe_trail(answer.trail))

    def test_an_empty_trail_says_nothing_ran(self):
        self.assertIn("No calculation", nlq_answer.describe_trail([]))


class ValidationAtExecutionTest(unittest.TestCase):
    """
    Execution revalidates. A plan reaching this point unchecked is the one
    path that could put an unverified model suggestion on screen as a
    figure.
    """

    def test_an_invalid_plan_is_refused_at_execution(self):
        frame = flat_frame()
        plan = nlq.QueryPlan(intent="aggregate", measure="nonexistent")

        answer = nlq_answer.execute_plan(frame, plan)

        self.assertFalse(answer.success)
        self.assertIn("nonexistent", answer.error)

    def test_a_share_of_an_average_is_refused_at_execution(self):
        frame = weighted_frame()
        plan = nlq.QueryPlan(
            intent="share",
            measure="revenue",
            aggregation="mean",
            segment="region",
            segment_value="whale"
        )

        answer = nlq_answer.execute_plan(frame, plan)

        self.assertFalse(answer.success)
        self.assertIn("do not add up", answer.error)

    def test_a_plan_with_no_rows_at_all_is_refused(self):
        frame = flat_frame().iloc[0:0]
        plan = nlq.QueryPlan(intent="aggregate", measure="revenue")

        answer = nlq_answer.execute_plan(frame, plan)

        self.assertFalse(answer.success)
        self.assertIn("no rows", answer.error)


if __name__ == "__main__":
    unittest.main()


class UnlabelledRankingTest(unittest.TestCase):
    """
    Rows with no segment label are part of the total but not part of the
    business. Answering "which region is worst" with "(not set)" is
    arithmetically true and tells the reader nothing about regions.
    """

    def setUp(self):
        frame = weighted_frame()
        # Blank the label on a slice small enough to be the lowest group.
        frame.loc[0:9, "region"] = None

        self.frame = frame

    def test_the_unlabelled_group_is_the_smallest(self):
        # Establishes the premise: without the substitution it would win.
        answer = run("revenue by region", self.frame)
        column = [
            name for name in answer.table.columns
            if name.startswith("Total")
        ][0]
        totals = dict(zip(
            answer.table[str(answer.plan.segment)],
            answer.table[column],
            strict=True
        ))

        self.assertLess(
            min(totals.values()),
            totals["shrimp"]
        )

    def test_a_named_part_is_given_as_the_lowest(self):
        answer = run("which region is worst for revenue", self.frame)

        self.assertTrue(answer.success, answer.error)
        self.assertEqual(answer.figures["leader"], "shrimp")

    def test_the_substitution_is_stated(self):
        answer = run("which region is worst for revenue", self.frame)

        self.assertTrue(
            any("left out of the ranking" in note for note in answer.notes),
            answer.notes
        )

    def test_the_unlabelled_rows_are_still_in_the_total(self):
        breakdown = run("revenue by region", self.frame)
        total = run("total revenue", self.frame)

        column = [
            name for name in breakdown.table.columns
            if name.startswith("Total")
        ][0]

        self.assertAlmostEqual(
            float(breakdown.table[column].sum()),
            total.figures["value"],
            places=6
        )

    def test_the_highest_part_is_unaffected(self):
        answer = run("which region is best for revenue", self.frame)

        self.assertEqual(answer.figures["leader"], "whale")
        self.assertEqual(answer.notes, [])
