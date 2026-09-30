"""Column classification and the structural description of a dataset."""

import unittest

import numpy as np
import pandas as pd

import schema
from tests.fixtures import messy_frame


class ColumnKindTest(unittest.TestCase):

    def test_numeric(self):
        self.assertEqual(
            schema.column_kind(pd.Series([1.0, 2.5, 3.0])),
            "numeric"
        )

    def test_boolean_is_not_numeric(self):
        self.assertEqual(
            schema.column_kind(pd.Series([True, False])),
            "boolean"
        )

    def test_datetime(self):
        self.assertEqual(
            schema.column_kind(
                pd.Series(pd.date_range("2024-01-01", periods=3))
            ),
            "datetime"
        )

    def test_text_is_categorical(self):
        self.assertEqual(
            schema.column_kind(pd.Series(["a", "b"])),
            "categorical"
        )


class DescribeColumnTest(unittest.TestCase):

    def test_constant_column_is_flagged(self):
        description = schema.describe_column(
            pd.Series(["same"] * 30, name="source"),
            30
        )

        self.assertTrue(description["is_constant"])
        self.assertEqual(description["distinct"], 1)

    def test_missing_values_are_counted_and_shared(self):
        series = pd.Series([1.0, None, 3.0, None], name="amount")

        description = schema.describe_column(series, 4)

        self.assertEqual(description["missing"], 2)
        self.assertEqual(description["missing_share"], 0.5)

    def test_mostly_unique_text_is_an_identifier(self):
        series = pd.Series(
            [f"REF-{index}" for index in range(40)],
            name="reference"
        )

        description = schema.describe_column(series, 40)

        self.assertTrue(description["looks_like_identifier"])

    def test_fully_unique_text_is_an_identifier_even_in_a_short_table(self):
        # The absolute threshold alone would miss this, because a 12 row
        # table cannot have more than 12 distinct values.
        series = pd.Series(
            [f"REF-{index}" for index in range(12)],
            name="reference"
        )

        description = schema.describe_column(series, 12)

        self.assertTrue(description["looks_like_identifier"])

    def test_repeated_labels_are_not_an_identifier(self):
        series = pd.Series(["north", "south"] * 30, name="region")

        description = schema.describe_column(series, 60)

        self.assertFalse(description["looks_like_identifier"])

    def test_a_continuous_measure_is_not_called_an_identifier(self):
        # A measure is nearly all distinct by nature. Treating that as
        # identity would make every revenue column an identifier.
        rng = np.random.default_rng(2)
        series = pd.Series(rng.normal(1000, 50, 300), name="revenue")

        description = schema.describe_column(series, 300)

        self.assertFalse(description["looks_like_identifier"])
        self.assertTrue(description["is_probably_continuous"])

    def test_high_cardinality_is_reported_separately(self):
        series = pd.Series(
            [f"code-{index}" for index in range(80)],
            name="code"
        )

        description = schema.describe_column(series, 80)

        self.assertTrue(description["is_high_cardinality"])


class ColumnSchemaTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.frame = messy_frame()
        cls.report = schema.column_schema(cls.frame)

    def test_shape_is_reported(self):
        self.assertEqual(self.report["rows"], len(self.frame))
        self.assertEqual(
            self.report["columns"],
            self.frame.shape[1]
        )

    def test_columns_are_grouped_by_kind(self):
        self.assertIn("monthly_spend", self.report["numeric_columns"])
        self.assertIn("plan", self.report["categorical_columns"])

    def test_identifier_column_is_surfaced(self):
        self.assertIn(
            "customer_id",
            self.report["identifier_like_columns"]
        )

    def test_constant_column_is_surfaced(self):
        self.assertIn("source", self.report["constant_columns"])

    def test_report_carries_no_cell_values(self):
        # The schema exists so the model never needs a row preview.
        flat = str(self.report)

        self.assertNotIn("CUST-00000", flat)
        self.assertNotIn("crm_export", flat)

    def test_every_column_is_described(self):
        described = {entry["name"] for entry in self.report["schema"]}

        self.assertEqual(described, set(self.frame.columns))


class NameNormalisationTest(unittest.TestCase):

    def test_separators_and_case_are_removed(self):
        self.assertEqual(
            schema.normalise_column_name("Customer_Name"),
            "customername"
        )
        self.assertEqual(
            schema.normalise_column_name("date of birth"),
            "dateofbirth"
        )


if __name__ == "__main__":
    unittest.main()
