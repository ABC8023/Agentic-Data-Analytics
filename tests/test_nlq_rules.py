"""
Reading a question without a model.

The tests are written as questions a person would actually type, because
that is the only honest specification for a parser. Three properties are
under test throughout.

The right plan comes out. Not just any plan: the intent, the measure, the
window and the filters each have to match what the sentence says.

Word order that changes meaning is respected. "Last 3 months" is a window
and "top 3 regions" is a ranking, and a parser that reads either as the
other produces a confident wrong answer.

What was not understood is reported. A question half read still yields a
plan, and the unused words are the only signal a reader gets.
"""

import unittest

import numpy as np
import pandas as pd

import nlq
import nlq_rules


def shop_frame(days: int = 800) -> pd.DataFrame:
    """A dated table with a measure, two segments and an identifier."""

    dates = pd.date_range("2022-06-01", periods=days, freq="D")
    regions = ["north", "south", "east"]
    channels = ["retail", "wholesale"]

    return pd.DataFrame({
        "order_date": dates,
        "revenue": np.linspace(100.0, 300.0, days).round(2),
        "units": np.arange(1, days + 1),
        "region": [regions[index % 3] for index in range(days)],
        "channel": [channels[index % 2] for index in range(days)],
        "order_reference": [f"ORD-{index:06d}" for index in range(days)],
    })


class ScannerTest(unittest.TestCase):

    def test_a_consumed_phrase_cannot_be_matched_again(self):
        scanner = nlq_rules.Scanner(" total revenue total ")

        first = scanner.take_phrase(" total ")
        second = scanner.take_phrase(" total ")

        self.assertGreaterEqual(first, 0)
        self.assertGreaterEqual(second, 0)
        self.assertNotEqual(first, second)

    def test_a_phrase_runs_out(self):
        scanner = nlq_rules.Scanner(" total ")

        scanner.take_phrase(" total ")

        self.assertEqual(scanner.take_phrase(" total "), -1)

    def test_unused_words_are_reported(self):
        scanner = nlq_rules.Scanner(" total revenue for wednesday ")

        scanner.take_phrase(" total ")

        self.assertIn("wednesday", scanner.remaining_words())

    def test_a_match_overlapping_used_text_is_skipped(self):
        scanner = nlq_rules.Scanner(" last month ")

        scanner.take(r"\blast month\b")

        self.assertIsNone(scanner.search(r"\bmonth\b"))


class NormalisationTest(unittest.TestCase):

    def test_case_and_punctuation_are_removed(self):
        self.assertEqual(
            nlq_rules.normalise_question("Total Revenue?!"),
            " total revenue "
        )

    def test_a_percent_sign_becomes_a_word(self):
        self.assertIn(
            "percent",
            nlq_rules.normalise_question("what % of revenue")
        )

    def test_the_text_is_padded_so_edge_phrases_match(self):
        # A phrase pattern written with spaces, such as " vs ", has to
        # match at the start and end of the question too.
        text = nlq_rules.normalise_question("north vs south")

        self.assertTrue(text.startswith(" "))
        self.assertTrue(text.endswith(" "))

    def test_iso_dates_survive(self):
        self.assertIn(
            "2024-01-01",
            nlq_rules.normalise_question("since 2024-01-01")
        )


class AggregateQuestionTest(unittest.TestCase):

    def setUp(self):
        self.frame = shop_frame()

    def parse(self, question):
        return nlq_rules.parse_question(question, self.frame)

    def test_a_total_is_understood(self):
        result = self.parse("what was total revenue")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "aggregate")
        self.assertEqual(result.plan.aggregation, "sum")
        self.assertEqual(result.plan.measure, "revenue")

    def test_an_average_is_understood(self):
        result = self.parse("what is the average revenue")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.aggregation, "mean")

    def test_a_median_is_understood(self):
        result = self.parse("median revenue")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.aggregation, "median")

    def test_a_row_count_is_understood(self):
        result = self.parse("how many orders are there")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.aggregation, "count")
        self.assertIsNone(result.plan.measure)

    def test_a_maximum_is_an_aggregation_not_a_ranking(self):
        result = self.parse("what is the maximum revenue")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.aggregation, "max")

    def test_a_named_measure_beats_the_detected_one(self):
        result = self.parse("total units")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.measure, "units")

    def test_a_total_number_of_is_a_count_not_a_sum(self):
        # "Total" and "number of" are both present, and the second one
        # decides.
        result = self.parse("total number of orders")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.aggregation, "count")

    def test_two_measures_are_refused_rather_than_guessed(self):
        result = self.parse("total revenue and units")

        self.assertFalse(result.understood)
        self.assertIn("more than one thing to measure", result.reason)

    def test_an_empty_question_is_refused(self):
        result = self.parse("   ")

        self.assertFalse(result.understood)
        self.assertIn("empty", result.reason)

    def test_an_empty_dataset_is_refused(self):
        result = nlq_rules.parse_question(
            "total revenue",
            self.frame.iloc[0:0]
        )

        self.assertFalse(result.understood)
        self.assertIn("no rows", result.reason)


class WindowQuestionTest(unittest.TestCase):

    def setUp(self):
        self.frame = shop_frame()

    def parse(self, question):
        return nlq_rules.parse_question(question, self.frame)

    def test_a_counted_window_is_read(self):
        result = self.parse("total revenue in the last 3 months")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.kind, "last_periods")
        self.assertEqual(result.plan.window.count, 3)
        self.assertEqual(result.plan.window.grain, "month")

    def test_a_single_period_window_is_read(self):
        result = self.parse("revenue last month")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.count, 1)
        self.assertEqual(result.plan.window.grain, "month")

    def test_this_year_means_the_most_recent_year_in_the_data(self):
        result = self.parse("revenue this year")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.kind, "last_periods")
        self.assertEqual(result.plan.window.grain, "year")

    def test_year_to_date_is_read(self):
        result = self.parse("revenue year to date")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.grain, "year")

    def test_a_named_year_is_read_as_a_period(self):
        result = self.parse("total revenue in 2023")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.kind, "period")
        self.assertEqual(result.plan.window.start, "2023-01-01")

    def test_a_named_month_and_year_is_read(self):
        result = self.parse("revenue in march 2023")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.kind, "period")
        self.assertEqual(result.plan.window.start, "2023-03-01")
        self.assertEqual(result.plan.window.grain, "month")

    def test_a_quarter_is_read(self):
        result = self.parse("revenue in q2 2023")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.start, "2023-04-01")
        self.assertEqual(result.plan.window.grain, "quarter")

    def test_a_bare_month_takes_the_latest_year_and_says_so(self):
        result = self.parse("revenue in march")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.start, "2024-03-01")
        self.assertTrue(
            any("latest year" in note for note in result.matched),
            result.matched
        )

    def test_an_explicit_range_is_read(self):
        result = self.parse(
            "revenue between 2023-01-01 and 2023-06-30"
        )

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.kind, "between_dates")
        self.assertEqual(result.plan.window.start, "2023-01-01")
        self.assertEqual(result.plan.window.end, "2023-06-30")

    def test_an_open_ended_range_is_read(self):
        result = self.parse("revenue since 2023-06-01")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.start, "2023-06-01")
        self.assertIsNone(result.plan.window.end)

    def test_an_upper_bound_is_read(self):
        result = self.parse("revenue before 2023-06-01")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.end, "2023-06-01")

    def test_a_year_end_bound_covers_the_whole_year(self):
        result = self.parse("revenue until 2023")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.end, "2023-12-31")

    def test_a_window_pulls_in_the_date_column(self):
        result = self.parse("revenue last month")

        self.assertEqual(result.plan.date, "order_date")

    def test_last_three_months_is_not_read_as_a_ranking(self):
        result = self.parse("total revenue in the last 3 months")

        self.assertIsNone(result.plan.top_n)


class GrainQuestionTest(unittest.TestCase):

    def setUp(self):
        self.frame = shop_frame()

    def parse(self, question):
        return nlq_rules.parse_question(question, self.frame)

    def test_by_month_is_a_period_not_a_breakdown(self):
        result = self.parse("revenue by month")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "timeseries")
        self.assertEqual(result.plan.grain, "month")
        self.assertIsNone(result.plan.segment)

    def test_monthly_is_a_period(self):
        result = self.parse("show me monthly revenue")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.grain, "month")

    def test_over_time_asks_for_a_series(self):
        result = self.parse("revenue over time")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "timeseries")

    def test_each_quarter_is_a_period(self):
        result = self.parse("revenue each quarter")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.grain, "quarter")

    def test_a_daily_rate_is_an_aggregation(self):
        result = self.parse("revenue daily rate")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.aggregation, "sum_per_day")

    def test_a_daily_rate_by_month_is_not_answered_as_one_figure(self):
        # The rules read "daily rate" but not "by month" here. One figure
        # would drop the per-month split the reader asked for.
        result = self.parse("revenue daily rate by month")

        self.assertFalse(result.understood)
        self.assertIn("'month'", result.reason)


class BreakdownQuestionTest(unittest.TestCase):

    def setUp(self):
        self.frame = shop_frame()

    def parse(self, question):
        return nlq_rules.parse_question(question, self.frame)

    def test_by_a_column_is_a_breakdown(self):
        result = self.parse("revenue by region")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "breakdown")
        self.assertEqual(result.plan.segment, "region")

    def test_per_a_column_is_a_breakdown(self):
        result = self.parse("total revenue per channel")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.segment, "channel")

    def test_which_column_ranks_and_keeps_one(self):
        result = self.parse("which region has the highest revenue")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "breakdown")
        self.assertEqual(result.plan.segment, "region")
        self.assertEqual(result.plan.direction, "highest")
        self.assertEqual(result.plan.top_n, 1)

    def test_a_top_count_is_read(self):
        result = self.parse("top 2 regions by revenue")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.top_n, 2)
        self.assertEqual(result.plan.direction, "highest")

    def test_a_bottom_count_reverses_the_direction(self):
        result = self.parse("bottom 2 regions by revenue")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.top_n, 2)
        self.assertEqual(result.plan.direction, "lowest")

    def test_worst_sets_the_direction_without_a_count(self):
        result = self.parse("which channel is worst for revenue")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.direction, "lowest")

    def test_an_identifier_column_is_refused_as_a_segment(self):
        result = self.parse("revenue by order reference")

        self.assertFalse(result.understood)
        self.assertIn("identifies rows", result.reason)

    def test_a_breakdown_can_carry_a_window(self):
        result = self.parse("revenue by region last 6 months")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.segment, "region")
        self.assertEqual(result.plan.window.count, 6)


class TrendQuestionTest(unittest.TestCase):

    def setUp(self):
        self.frame = shop_frame()

    def parse(self, question):
        return nlq_rules.parse_question(question, self.frame)

    def test_is_it_growing_is_a_trend(self):
        result = self.parse("is revenue growing")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "trend")
        self.assertEqual(result.plan.date, "order_date")

    def test_declining_is_a_trend(self):
        result = self.parse("is revenue declining")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "trend")

    def test_a_bare_trend_question_resolves_its_columns(self):
        result = self.parse("what is the trend")

        self.assertTrue(result.understood, result.reason)
        self.assertIsNotNone(result.plan.date)
        self.assertIsNotNone(result.plan.measure)


class ExtremeQuestionTest(unittest.TestCase):

    def setUp(self):
        self.frame = shop_frame()

    def parse(self, question):
        return nlq_rules.parse_question(question, self.frame)

    def test_which_month_asks_about_periods(self):
        result = self.parse("which month had the highest revenue")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "extreme")
        self.assertEqual(result.plan.grain, "month")
        self.assertEqual(result.plan.direction, "highest")

    def test_which_month_was_worst_reverses_the_direction(self):
        result = self.parse("which month was worst")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.direction, "lowest")

    def test_when_was_asks_about_periods(self):
        result = self.parse("when was revenue lowest")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "extreme")

    def test_which_region_is_a_breakdown_not_a_period(self):
        result = self.parse("which region was worst")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "breakdown")


class CompareQuestionTest(unittest.TestCase):

    def setUp(self):
        self.frame = shop_frame()

    def parse(self, question):
        return nlq_rules.parse_question(question, self.frame)

    def test_compared_to_last_month_builds_two_windows(self):
        result = self.parse(
            "how did revenue do compared to last month"
        )

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "compare")
        self.assertEqual(result.plan.window.offset, 0)
        self.assertEqual(result.plan.comparison.offset, 1)

    def test_the_grain_follows_the_phrasing(self):
        result = self.parse("revenue this quarter vs last quarter")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.grain, "quarter")
        self.assertEqual(result.plan.comparison.grain, "quarter")

    def test_year_on_year_is_a_comparison(self):
        result = self.parse("revenue year on year")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "compare")

    def test_a_comparison_defaults_to_months(self):
        result = self.parse("is revenue better than the previous period")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.grain, "month")


class ShareQuestionTest(unittest.TestCase):

    def setUp(self):
        self.frame = shop_frame()

    def parse(self, question):
        return nlq_rules.parse_question(question, self.frame)

    def test_a_share_of_a_named_part_is_read(self):
        result = self.parse("what share of revenue is north")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "share")
        self.assertEqual(result.plan.segment, "region")
        self.assertEqual(result.plan.segment_value, "north")

    def test_a_percentage_question_is_a_share(self):
        result = self.parse("what percentage of revenue is wholesale")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.segment, "channel")
        self.assertEqual(result.plan.segment_value, "wholesale")

    def test_a_share_with_no_named_part_is_refused(self):
        result = self.parse("what share of revenue")

        self.assertFalse(result.understood)
        self.assertIn("does not name which part", result.reason)


class FilterQuestionTest(unittest.TestCase):

    def setUp(self):
        self.frame = shop_frame()

    def parse(self, question):
        return nlq_rules.parse_question(question, self.frame)

    def test_a_category_value_becomes_a_filter(self):
        result = self.parse("total revenue in north")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(len(result.plan.filters), 1)
        self.assertEqual(result.plan.filters[0].column, "region")
        self.assertEqual(result.plan.filters[0].values, ("north",))

    def test_two_category_values_become_two_filters(self):
        result = self.parse("revenue for north retail")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(len(result.plan.filters), 2)

    def test_a_numeric_condition_attaches_to_the_named_column(self):
        result = self.parse("how many orders have units over 500")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(len(result.plan.filters), 1)
        self.assertEqual(result.plan.filters[0].column, "units")
        self.assertEqual(result.plan.filters[0].operator, "greater_than")
        self.assertEqual(result.plan.filters[0].values, (500.0,))

    def test_at_least_is_inclusive(self):
        result = self.parse("count orders with units at least 100")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(
            result.plan.filters[0].operator,
            "greater_or_equal"
        )

    def test_below_is_exclusive(self):
        result = self.parse("count orders with units below 100")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.filters[0].operator, "less_than")

    def test_a_numeric_range_is_read(self):
        result = self.parse("count orders with units between 100 and 200")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.filters[0].operator, "between")
        self.assertEqual(result.plan.filters[0].values, (100.0, 200.0))

    def test_a_four_digit_range_is_not_read_as_years(self):
        # "Between 1000 and 2000" is a quantity, and reading it as a pair
        # of years would answer a question about money with a date filter.
        result = self.parse("count orders with units between 1000 and 2000")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.kind, "all")
        self.assertEqual(result.plan.filters[0].operator, "between")

    def test_a_condition_with_no_named_column_is_reported_not_guessed(self):
        result = self.parse("how many orders over 500")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.filters, ())
        self.assertTrue(
            any("not used" in note for note in result.warnings),
            result.warnings
        )

    def test_a_filter_and_a_window_can_coexist(self):
        result = self.parse("revenue in north last 3 months")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.filters[0].values, ("north",))
        self.assertEqual(result.plan.window.count, 3)


class ReportingTest(unittest.TestCase):
    """
    A question half understood is not answered from the half. The words
    that were not read are reported, and the question goes to the model
    planner or back to the reader.
    """

    def setUp(self):
        self.frame = shop_frame()

    def parse(self, question):
        return nlq_rules.parse_question(question, self.frame)

    def test_unused_words_are_reported(self):
        result = self.parse("total revenue for our best customers")

        self.assertIn("customers", result.ignored)

    def test_a_question_read_only_in_part_is_not_answered(self):
        # Answering "total revenue" here would drop "best customers", and
        # the figure would look like an answer to the question asked.
        result = self.parse("total revenue for our best customers")

        self.assertFalse(result.understood)
        self.assertIsNone(result.plan)
        self.assertIn("'customers'", result.reason)

    def test_an_unknown_column_is_not_answered_as_the_default_measure(self):
        result = self.parse("average age of customers")

        self.assertFalse(result.understood)
        self.assertIn("'age'", result.reason)

    def test_wrapping_words_do_not_stop_an_answer(self):
        for question in (
            "how much money did we make in total",
            "how many orders",
            "what did the north bring in",
        ):
            with self.subTest(question=question):
                result = self.parse(question)

                self.assertTrue(result.understood, result.reason)

    def test_the_month_before_is_read_as_the_second_period(self):
        result = self.parse(
            "revenue last month compared with the month before"
        )

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "compare")
        self.assertEqual(result.ignored, [])

    def test_filler_words_are_not_reported_as_unused(self):
        result = self.parse("what was the total revenue")

        self.assertEqual(result.ignored, [])

    def test_what_was_matched_is_reported(self):
        result = self.parse("total revenue last 3 months")

        self.assertTrue(result.matched)
        self.assertTrue(
            any("window" in note for note in result.matched),
            result.matched
        )

    def test_a_refusal_says_why(self):
        result = self.parse("total revenue and units")

        self.assertFalse(result.understood)
        self.assertTrue(result.reason)

    def test_every_understood_plan_validates(self):
        questions = [
            "total revenue",
            "average revenue by region",
            "revenue by month",
            "is revenue growing",
            "which month was best",
            "revenue this month vs last month",
            "what share of revenue is north",
            "how many orders",
            "top 2 regions by revenue",
            "revenue in north last 3 months",
        ]

        for question in questions:
            with self.subTest(question=question):
                result = self.parse(question)

                self.assertTrue(result.understood, result.reason)

                validation = nlq.validate_plan(result.plan, self.frame)

                self.assertTrue(validation.valid, validation.errors)

    def test_the_plan_keeps_the_question_it_came_from(self):
        result = self.parse("total revenue")

        self.assertEqual(result.plan.question, "total revenue")

    def test_a_rule_parsed_plan_is_labelled_as_such(self):
        result = self.parse("total revenue")

        self.assertEqual(result.plan.source, "rules")


class DatelessDatasetTest(unittest.TestCase):
    """
    A table with no timeline can still answer some questions, and has to
    refuse the rest with a reason rather than an empty chart.
    """

    def setUp(self):
        self.frame = pd.DataFrame({
            "part_number": [f"P-{index:05d}" for index in range(60)],
            "price": np.linspace(5.0, 90.0, 60).round(2),
            "category": ["brake", "filter", "belt"] * 20,
        })

    def parse(self, question):
        return nlq_rules.parse_question(question, self.frame)

    def test_a_total_still_works(self):
        result = self.parse("total price")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.measure, "price")

    def test_a_breakdown_still_works(self):
        result = self.parse("price by category")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.segment, "category")

    def test_a_trend_is_refused_with_a_reason(self):
        result = self.parse("is price growing")

        self.assertFalse(result.understood)
        self.assertIn("date column", result.reason)

    def test_a_window_is_refused_with_a_reason(self):
        result = self.parse("total price last 3 months")

        self.assertFalse(result.understood)
        self.assertIn("date", result.reason)


if __name__ == "__main__":
    unittest.main()


class TextStoredMeasureTest(unittest.TestCase):
    """
    A measure column holding a few unreadable cells arrives as text.

    Gating on the storage type excluded it from being a measure, so every
    question naming it was answered by totalling whatever column detection
    liked instead. The figure looked plausible and was for the wrong thing.
    """

    def setUp(self):
        frame = shop_frame(days=400)
        frame["revenue"] = frame["revenue"].astype(object)
        frame.loc[0:19, "revenue"] = "not recorded"

        self.frame = frame

    def parse(self, question):
        return nlq_rules.parse_question(question, self.frame)

    def test_a_named_text_measure_is_used(self):
        result = self.parse("total revenue")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.measure, "revenue")

    def test_it_is_not_swapped_for_another_numeric_column(self):
        result = self.parse("total revenue")

        self.assertNotEqual(result.plan.measure, "units")

    def test_a_bare_question_still_resolves_to_it(self):
        result = self.parse("is revenue growing")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.measure, "revenue")

    def test_the_unreadable_rows_are_disclosed(self):
        result = self.parse("total revenue")

        self.assertTrue(
            any("20 row(s)" in note for note in result.warnings),
            result.warnings
        )

    def test_a_share_question_still_finds_its_part(self):
        result = self.parse("what share of revenue is north")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.segment, "region")
        self.assertEqual(result.plan.segment_value, "north")


class ComparisonPhraseTest(unittest.TestCase):
    """
    A comparison names two periods, and both should be consumed. A phrase
    left over is reported as ignored, which reads as though half the
    question was dropped.
    """

    def setUp(self):
        self.frame = shop_frame()

    def parse(self, question):
        return nlq_rules.parse_question(question, self.frame)

    def test_both_period_phrases_are_used(self):
        result = self.parse("revenue this month compared to last month")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.ignored, [])

    def test_the_second_phrase_can_set_the_grain(self):
        result = self.parse("revenue compared to last quarter")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.window.grain, "quarter")
        self.assertEqual(result.plan.comparison.grain, "quarter")

    def test_a_vague_previous_period_still_works(self):
        result = self.parse("revenue vs the previous period")

        self.assertTrue(result.understood, result.reason)
        self.assertEqual(result.plan.intent, "compare")

    def test_common_verbs_are_not_reported_as_unused(self):
        result = self.parse("which month had the highest revenue")

        self.assertTrue(result.understood, result.reason)
        self.assertNotIn("had", result.ignored)
