"""Cleaning behaviour and its audit trail."""

import os
import unittest

os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

import cleaning  # noqa: E402
import dataset_store  # noqa: E402
from tests.fixtures import messy_frame  # noqa: E402


class CleaningTest(unittest.TestCase):

    def setUp(self):
        dataset_store.DATAFRAME_CACHE.clear()
        self.frame = messy_frame()
        self.key = dataset_store.make_dataset_key("clean", "messy.csv")
        dataset_store.register_dataset(self.key, self.frame)

    def test_defaults_remove_duplicates_and_fill_gaps(self):
        result = cleaning.clean_dataset(name=self.key)

        self.assertTrue(result["success"])
        self.assertEqual(result["duplicates_removed"], 8)
        self.assertEqual(result["remaining_missing_values"], 0)

    def test_the_original_is_left_untouched(self):
        before = len(self.frame)

        cleaning.clean_dataset(name=self.key)

        self.assertEqual(len(dataset_store.DATAFRAME_CACHE[self.key]), before)

    def test_dropping_columns_is_reported(self):
        result = cleaning.clean_dataset(
            name=self.key,
            drop_columns=["source"]
        )

        self.assertEqual(result["columns_removed"], ["source"])
        self.assertNotIn(
            "source",
            result["cleaned_dataframe"].columns
        )

    def test_an_unknown_column_is_refused_by_name(self):
        result = cleaning.clean_dataset(
            name=self.key,
            drop_columns=["nope"]
        )

        self.assertFalse(result["success"])
        self.assertIn("nope", result["error"])

    def test_leave_strategy_keeps_missing_values(self):
        result = cleaning.clean_dataset(
            name=self.key,
            numeric_missing_strategy="leave",
            categorical_missing_strategy="leave"
        )

        self.assertGreater(result["remaining_missing_values"], 0)

    def test_dropping_rows_for_missing_values_is_counted(self):
        result = cleaning.clean_dataset(
            name=self.key,
            numeric_missing_strategy="drop_rows",
            categorical_missing_strategy="leave"
        )

        self.assertGreater(
            result["rows_removed_for_missing_values"],
            0
        )

    def test_invalid_strategies_are_refused(self):
        numeric = cleaning.clean_dataset(
            name=self.key,
            numeric_missing_strategy="teleport"
        )
        categorical = cleaning.clean_dataset(
            name=self.key,
            categorical_missing_strategy="teleport"
        )

        self.assertFalse(numeric["success"])
        self.assertFalse(categorical["success"])

    def test_missing_label_strategy_uses_a_visible_placeholder(self):
        result = cleaning.clean_dataset(
            name=self.key,
            categorical_missing_strategy="missing_label"
        )

        self.assertIn(
            "Missing",
            set(result["cleaned_dataframe"]["plan"])
        )

    def test_removing_every_column_is_refused(self):
        result = cleaning.clean_dataset(
            name=self.key,
            drop_columns=list(self.frame.columns)
        )

        self.assertFalse(result["success"])

    def test_audit_numbers_reconcile(self):
        result = cleaning.clean_dataset(name=self.key)

        self.assertEqual(
            result["cleaned_rows"],
            result["original_rows"]
            - result["duplicates_removed"]
            - result["rows_removed_for_missing_values"]
        )


if __name__ == "__main__":
    unittest.main()
