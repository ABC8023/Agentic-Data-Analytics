"""How numbers and periods are written down."""

import unittest

import numpy as np
import pandas as pd

import formatting


class NumberTest(unittest.TestCase):

    def test_thousands_are_separated(self):
        self.assertEqual(formatting.format_number(1234567), "1,234,567")

    def test_a_whole_float_loses_its_decimals(self):
        self.assertEqual(formatting.format_number(1200.0), "1,200")

    def test_precision_follows_the_scale_of_the_value(self):
        self.assertEqual(formatting.format_number(1234.567), "1,235")
        self.assertEqual(formatting.format_number(12.345), "12.3")
        self.assertEqual(formatting.format_number(1.2345), "1.23")
        self.assertEqual(formatting.format_number(0.012345), "0.0123")

    def test_decimals_can_be_pinned(self):
        self.assertEqual(
            formatting.format_number(1234.5678, decimals=2),
            "1,234.57"
        )

    def test_negatives_keep_their_sign(self):
        self.assertEqual(formatting.format_number(-4200), "-4,200")

    def test_missing_values_are_named_not_zero(self):
        for value in [None, np.nan, pd.NA, pd.NaT]:
            with self.subTest(value=value):
                self.assertEqual(
                    formatting.format_number(value),
                    formatting.MISSING_TEXT
                )

    def test_unformattable_input_is_returned_as_text(self):
        self.assertEqual(formatting.format_number("north"), "north")


class CompactTest(unittest.TestCase):

    def test_small_numbers_stay_plain(self):
        self.assertEqual(formatting.format_compact(847), "847")

    def test_thousands_are_abbreviated(self):
        self.assertEqual(formatting.format_compact(1200), "1.2k")

    def test_millions_are_abbreviated(self):
        self.assertEqual(formatting.format_compact(3_400_000), "3.4M")

    def test_billions_are_abbreviated(self):
        self.assertEqual(formatting.format_compact(2_000_000_000), "2B")

    def test_trailing_zeros_are_dropped(self):
        self.assertEqual(formatting.format_compact(2000), "2k")

    def test_negatives_keep_their_sign(self):
        self.assertEqual(formatting.format_compact(-1500), "-1.5k")


class PercentTest(unittest.TestCase):

    def test_a_proportion_becomes_a_percentage(self):
        self.assertEqual(formatting.format_percent(0.184), "18.4%")

    def test_an_already_scaled_value_is_not_multiplied_again(self):
        self.assertEqual(
            formatting.format_percent(18.4, already_scaled=True),
            "18.4%"
        )

    def test_decimals_can_be_pinned(self):
        self.assertEqual(
            formatting.format_percent(0.18437, decimals=2),
            "18.44%"
        )


class SignedTest(unittest.TestCase):

    def test_a_rise_gets_a_plus(self):
        self.assertEqual(formatting.format_signed(1204), "+1,204")

    def test_a_fall_gets_a_minus(self):
        self.assertEqual(formatting.format_signed(-38.5), "-38.5")

    def test_no_change_gets_no_sign(self):
        self.assertEqual(formatting.format_signed(0), "0")

    def test_a_signed_percentage_reads_clearly(self):
        self.assertEqual(
            formatting.format_signed_percent(-0.072),
            "-7.2%"
        )
        self.assertEqual(
            formatting.format_signed_percent(0.072),
            "+7.2%"
        )


class PeriodTest(unittest.TestCase):

    def test_a_month_reads_as_month_and_year(self):
        self.assertEqual(
            formatting.format_period("2026-08-01", "month"),
            "Aug 2026"
        )

    def test_a_quarter_reads_as_a_quarter_number(self):
        self.assertEqual(
            formatting.format_period("2026-08-01", "quarter"),
            "Q3 2026"
        )

    def test_a_year_reads_as_a_year(self):
        self.assertEqual(
            formatting.format_period("2026-08-01", "year"),
            "2026"
        )

    def test_a_day_reads_as_a_date(self):
        self.assertEqual(
            formatting.format_period("2026-08-01", "day"),
            "01 Aug 2026"
        )

    def test_a_week_is_named_by_its_start(self):
        self.assertIn(
            "week of",
            formatting.format_period("2026-08-03", "week")
        )

    def test_a_missing_period_is_named(self):
        self.assertEqual(
            formatting.format_period(None),
            formatting.MISSING_TEXT
        )


class RangeTest(unittest.TestCase):

    def test_a_span_reads_from_first_to_last(self):
        self.assertEqual(
            formatting.format_period_range(
                "2024-01-01",
                "2026-07-01",
                "month"
            ),
            "Jan 2024 to Jul 2026"
        )

    def test_a_single_period_is_not_written_twice(self):
        self.assertEqual(
            formatting.format_period_range(
                "2024-01-01",
                "2024-01-31",
                "month"
            ),
            "Jan 2024"
        )


class WordingTest(unittest.TestCase):

    def test_one_is_singular(self):
        self.assertEqual(
            formatting.pluralise(1, "period"),
            "1 period"
        )

    def test_other_counts_are_plural(self):
        self.assertEqual(
            formatting.pluralise(7, "period"),
            "7 periods"
        )

    def test_zero_is_plural(self):
        self.assertEqual(
            formatting.pluralise(0, "period"),
            "0 periods"
        )

    def test_an_irregular_plural_can_be_given(self):
        self.assertEqual(
            formatting.pluralise(3, "analysis", "analyses"),
            "3 analyses"
        )

    def test_a_grain_reads_as_an_adjective(self):
        self.assertEqual(formatting.describe_grain("month"), "monthly")
        self.assertEqual(formatting.describe_grain("day"), "daily")


if __name__ == "__main__":
    unittest.main()
