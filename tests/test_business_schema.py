"""
Working out which column is the measure, the date and the segment.

The refusal cases matter as much as the successes. A parts catalogue has
no measure and no timeline, and detection has to say so rather than pick
something arbitrary and produce a dashboard of nonsense.
"""

import unittest

import numpy as np
import pandas as pd

import schema


def sales_frame(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(41)

    return pd.DataFrame({
        "order_id": [f"ORD-{index:06d}" for index in range(rows)],
        "order_date": pd.date_range("2024-01-01", periods=rows, freq="D"),
        "region": rng.choice(["north", "south", "east", "west"], rows),
        "channel": rng.choice(["web", "store"], rows),
        "revenue": rng.normal(500, 90, rows).round(2),
        "units": rng.integers(1, 12, rows),
        "customer_email": [f"user{index}@example.com" for index in range(rows)],
    })


def catalogue_frame(rows: int = 300) -> pd.DataFrame:
    """A reference table: every column is a label or an identifier."""

    rng = np.random.default_rng(42)

    return pd.DataFrame({
        "part_code": [f"PC-{index:06d}" for index in range(rows)],
        "ktype": [f"KT{index:05d}" for index in range(rows)],
        "body_style": rng.choice(["sedan", "coupe", "touring"], rows),
        "drivetrain": rng.choice(["rwd", "awd"], rows),
    })


class MeasureDetectionTest(unittest.TestCase):

    def test_a_named_money_column_wins(self):
        detected = schema.detect_business_schema(sales_frame())

        self.assertEqual(detected["measure"], "revenue")

    def test_the_reason_mentions_the_name(self):
        detected = schema.detect_business_schema(sales_frame())

        self.assertIn("name reads like a measure", detected["measure_reason"])

    def test_revenue_is_preferred_over_units(self):
        candidates = schema.measure_candidates(sales_frame())

        self.assertLess(
            candidates.index("revenue"),
            candidates.index("units")
        )

    def test_a_year_column_is_not_a_measure(self):
        # Totalling a column of years produces a meaningless number.
        frame = pd.DataFrame({
            "model_year": [2019, 2020, 2021, 2022] * 25,
            "listings": np.arange(100, dtype=float),
        })

        self.assertEqual(
            schema.detect_business_schema(frame)["measure"],
            "listings"
        )

    def test_a_constant_column_is_not_a_measure(self):
        frame = pd.DataFrame({
            "always_one": [1.0] * 50,
            "spend": np.arange(50, dtype=float),
        })

        candidates = schema.measure_candidates(frame)

        self.assertNotIn("always_one", candidates)

    def test_a_boolean_column_is_not_a_measure(self):
        frame = pd.DataFrame({
            "is_paid": [True, False] * 25,
            "amount": np.arange(50, dtype=float),
        })

        self.assertNotIn("is_paid", schema.measure_candidates(frame))

    def test_an_identity_named_number_is_not_a_measure(self):
        frame = pd.DataFrame({
            "customer_id": np.arange(1000, 1100),
            "amount": np.arange(100, dtype=float),
        })

        self.assertEqual(
            schema.detect_business_schema(frame)["measure"],
            "amount"
        )

    def test_shape_decides_when_no_name_helps(self):
        rng = np.random.default_rng(3)
        frame = pd.DataFrame({
            "alpha": rng.normal(50, 10, 200),
            "beta": rng.choice([1, 2, 3], 200),
        })

        detected = schema.detect_business_schema(frame)

        self.assertEqual(detected["measure"], "alpha")
        self.assertIn("from its shape", detected["measure_reason"])

    def test_a_catalogue_has_no_measure(self):
        detected = schema.detect_business_schema(catalogue_frame())

        self.assertIsNone(detected["measure"])
        self.assertFalse(detected["can_measure"])

    def test_the_absence_of_a_measure_is_explained(self):
        detected = schema.detect_business_schema(catalogue_frame())

        self.assertTrue(
            any("no total to track" in note for note in detected["notes"])
        )
        self.assertTrue(
            any("Counting rows" in note for note in detected["notes"])
        )


class DateDetectionTest(unittest.TestCase):

    def test_a_named_date_column_is_found(self):
        detected = schema.detect_business_schema(sales_frame())

        self.assertEqual(detected["date"], "order_date")

    def test_a_catalogue_has_no_date(self):
        detected = schema.detect_business_schema(catalogue_frame())

        self.assertIsNone(detected["date"])
        self.assertFalse(detected["can_track_time"])

    def test_the_absence_of_a_date_names_what_is_lost(self):
        detected = schema.detect_business_schema(catalogue_frame())

        note = " ".join(detected["notes"])

        self.assertIn("movement over time", note)
        self.assertIn("projections", note)

    def test_a_named_date_beats_an_unnamed_one(self):
        rows = 60
        frame = pd.DataFrame({
            "mystery": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "invoice_date": pd.date_range(
                "2024-03-01",
                periods=rows,
                freq="D"
            ),
            "amount": np.arange(rows, dtype=float),
        })

        candidates = schema.date_candidates(frame)

        self.assertEqual(candidates[0], "invoice_date")


class SegmentDetectionTest(unittest.TestCase):

    def test_a_named_segment_is_found(self):
        detected = schema.detect_business_schema(sales_frame())

        self.assertIn(detected["segment"], {"region", "channel"})

    def test_an_identifier_is_never_a_segment(self):
        candidates = schema.segment_candidates(sales_frame())

        self.assertNotIn("order_id", candidates)
        self.assertNotIn("customer_email", candidates)

    def test_a_measure_is_not_offered_as_a_segment(self):
        detected = schema.detect_business_schema(sales_frame())

        self.assertNotIn(
            detected["measure"],
            detected["segment_candidates"]
        )

    def test_a_low_cardinality_number_can_be_a_segment(self):
        frame = pd.DataFrame({
            "store_number": [1, 2, 3, 4] * 25,
            "revenue": np.arange(100, dtype=float),
        })

        self.assertIn(
            "store_number",
            schema.segment_candidates(frame, exclude=("revenue",))
        )

    def test_a_wide_number_is_not_a_segment(self):
        frame = pd.DataFrame({
            "reading": np.arange(200, dtype=float),
            "revenue": np.arange(200, dtype=float),
        })

        self.assertNotIn(
            "reading",
            schema.segment_candidates(frame, exclude=("revenue",))
        )

    def test_a_column_with_too_many_parts_is_not_a_segment(self):
        frame = pd.DataFrame({
            "label": [f"L{index % 80}" for index in range(400)],
            "revenue": np.arange(400, dtype=float),
        })

        self.assertNotIn(
            "label",
            schema.segment_candidates(frame, exclude=("revenue",))
        )

    def test_a_comfortable_split_is_preferred(self):
        rng = np.random.default_rng(8)
        rows = 300
        frame = pd.DataFrame({
            "wide_split": [f"W{index % 40}" for index in range(rows)],
            "narrow_split": rng.choice(["a", "b", "c", "d"], rows),
            "revenue": rng.normal(100, 10, rows),
        })

        candidates = schema.segment_candidates(
            frame,
            exclude=("revenue",)
        )

        self.assertLess(
            candidates.index("narrow_split"),
            candidates.index("wide_split")
        )

    def test_a_dataset_without_a_segment_says_so(self):
        frame = pd.DataFrame({
            "when": pd.date_range("2024-01-01", periods=50, freq="D"),
            "revenue": np.arange(50, dtype=float),
        })

        detected = schema.detect_business_schema(frame)

        self.assertIsNone(detected["segment"])
        self.assertFalse(detected["can_split"])
        self.assertTrue(
            any("cannot be broken down" in note for note in detected["notes"])
        )


class OverrideTest(unittest.TestCase):

    def test_the_measure_can_be_overridden(self):
        detected = schema.detect_business_schema(
            sales_frame(),
            measure="units"
        )

        self.assertEqual(detected["measure"], "units")

    def test_the_date_can_be_overridden(self):
        frame = sales_frame()
        frame["shipped_date"] = frame["order_date"] + pd.Timedelta(days=2)

        detected = schema.detect_business_schema(
            frame,
            date="shipped_date"
        )

        self.assertEqual(detected["date"], "shipped_date")

    def test_the_segment_can_be_overridden(self):
        detected = schema.detect_business_schema(
            sales_frame(),
            segment="channel"
        )

        self.assertEqual(detected["segment"], "channel")

    def test_an_unknown_override_falls_back_to_detection(self):
        detected = schema.detect_business_schema(
            sales_frame(),
            measure="not_a_column"
        )

        self.assertEqual(detected["measure"], "revenue")


class CapabilityTest(unittest.TestCase):

    def test_a_sales_extract_can_build_a_dashboard(self):
        detected = schema.detect_business_schema(sales_frame())

        self.assertTrue(detected["can_build_dashboard"])

    def test_a_catalogue_cannot(self):
        detected = schema.detect_business_schema(catalogue_frame())

        self.assertFalse(detected["can_build_dashboard"])

    def test_identifiers_are_listed(self):
        detected = schema.detect_business_schema(sales_frame())

        self.assertIn("order_id", detected["identifier_columns"])

    def test_the_summary_names_the_roles(self):
        detected = schema.detect_business_schema(sales_frame())

        sentence = schema.describe_business_schema(detected)

        self.assertIn("revenue", sentence)
        self.assertIn("order_date", sentence)

    def test_the_summary_admits_when_nothing_was_found(self):
        frame = pd.DataFrame({
            "ref": [f"R{index}" for index in range(30)],
        })

        sentence = schema.describe_business_schema(
            schema.detect_business_schema(frame)
        )

        self.assertIn("No measure, date or segment", sentence)


class NameHintTest(unittest.TestCase):

    def test_a_hint_is_found_by_position(self):
        self.assertEqual(
            schema.name_hint_rank("total_revenue", ("revenue", "units")),
            0
        )

    def test_the_first_matching_hint_wins(self):
        self.assertEqual(
            schema.name_hint_rank("unit_price", ("price", "units")),
            0
        )

    def test_no_match_returns_nothing(self):
        self.assertIsNone(
            schema.name_hint_rank("colour", ("revenue",))
        )

    def test_separators_and_case_do_not_matter(self):
        self.assertIsNotNone(
            schema.name_hint_rank("Order Date", schema.DATE_NAME_HINTS)
        )


class YearDetectionTest(unittest.TestCase):

    def test_four_digit_years_are_recognised(self):
        self.assertTrue(
            schema.looks_like_year(pd.Series([2019, 2020, 2021] * 10))
        )

    def test_money_is_not_a_year(self):
        self.assertFalse(
            schema.looks_like_year(pd.Series([1999.99, 2020.50]))
        )

    def test_values_outside_the_range_are_not_years(self):
        self.assertFalse(
            schema.looks_like_year(pd.Series([1500, 2020, 3000]))
        )

    def test_text_is_not_a_year(self):
        self.assertFalse(
            schema.looks_like_year(pd.Series(["2020", "2021"]))
        )


if __name__ == "__main__":
    unittest.main()


class MeasurabilityTest(unittest.TestCase):
    """
    Whether a column can be totalled, judged on its values rather than on
    how pandas happened to store them.

    This is the rule the question parser, the plan validator and role
    detection all share. When they disagreed, a revenue column holding a
    few cells of "not recorded" arrived as text, was excluded from being a
    measure, and every question about revenue was answered by quietly
    totalling a different column.
    """

    def test_a_numeric_column_is_measurable(self):
        self.assertTrue(
            schema.is_measurable(pd.Series([1.0, 2.0, 3.0]))
        )

    def test_a_money_column_stored_as_text_is_measurable(self):
        series = pd.Series(["10.5", "20.25", "not recorded", "30.0"])

        self.assertTrue(schema.is_measurable(series))

    def test_a_label_column_is_not_measurable(self):
        series = pd.Series(["north", "south", "east", "west"])

        self.assertFalse(schema.is_measurable(series))

    def test_a_boolean_column_is_not_measurable(self):
        self.assertFalse(
            schema.is_measurable(pd.Series([True, False, True]))
        )

    def test_a_date_column_is_not_measurable(self):
        series = pd.Series(pd.date_range("2024-01-01", periods=5))

        self.assertFalse(schema.is_measurable(series))

    def test_an_empty_column_is_not_measurable(self):
        self.assertFalse(
            schema.is_measurable(pd.Series([None, None], dtype="object"))
        )

    def test_the_share_is_measured_against_filled_in_values(self):
        # Mostly blank but unambiguously numeric where present. Judging
        # against every row would call a sparse money column unmeasurable.
        series = pd.Series([None] * 90 + ["12.5"] * 10)

        self.assertAlmostEqual(schema.numeric_share(series), 1.0)
        self.assertTrue(schema.is_measurable(series))

    def test_a_mostly_text_column_is_not_measurable(self):
        series = pd.Series(["1"] + ["unknown"] * 9)

        self.assertLess(schema.numeric_share(series), 0.5)
        self.assertFalse(schema.is_measurable(series))

    def test_reading_as_numbers_keeps_the_column_name(self):
        series = pd.Series(["1", "2", "bad"], name="revenue")

        converted = schema.as_numbers(series)

        self.assertEqual(converted.name, "revenue")
        self.assertTrue(pd.isna(converted.iloc[2]))

    def test_a_text_measure_is_detected_as_the_measure(self):
        frame = pd.DataFrame({
            "order_date": pd.date_range("2024-01-01", periods=60),
            "revenue": (
                ["100.0"] * 30 + ["not recorded"] + ["200.0"] * 29
            ),
            "region": ["north", "south", "east"] * 20,
        })

        detected = schema.detect_business_schema(frame)

        self.assertEqual(detected["measure"], "revenue")
