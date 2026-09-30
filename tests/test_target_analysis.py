"""
Target column analysis and the trainability guard.

The guard exists because of a real failure: a product code with 4,339
distinct values was recommended for classification, then training died
with an unexplained stratification error.
"""

import os
import unittest

import numpy as np
import pandas as pd

os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

import dataset_store  # noqa: E402
import ml  # noqa: E402
from tests.fixtures import (  # noqa: E402
    classification_frame,
    regression_frame,
)


class TaskRecommendationTest(unittest.TestCase):

    def setUp(self):
        dataset_store.DATAFRAME_CACHE.clear()

    def register(self, frame, name="t.csv"):
        key = dataset_store.make_dataset_key("target", name)
        dataset_store.register_dataset(key, frame)
        return key

    def test_text_target_reads_as_classification(self):
        key = self.register(classification_frame())

        report = ml.analyse_tool(
            name=key,
            target_column="churned"
        )

        self.assertEqual(report["recommended_task"], "Classification")
        self.assertTrue(report["is_trainable"])

    def test_continuous_target_reads_as_regression(self):
        key = self.register(regression_frame())

        report = ml.analyse_tool(name=key, target_column="target")

        self.assertEqual(report["recommended_task"], "Regression")
        self.assertTrue(report["is_trainable"])

    def test_recommendation_reason_is_a_sentence(self):
        key = self.register(classification_frame())

        report = ml.analyse_tool(
            name=key,
            target_column="churned"
        )

        reason = report["recommendation_reason"]

        self.assertTrue(reason.endswith("."))
        self.assertNotIn("so therefore", reason)

    def test_few_repeated_numbers_read_as_classes(self):
        frame = pd.DataFrame({
            "feature": np.arange(200, dtype=float),
            "grade": [1, 2, 3] * 66 + [1, 2],
        })
        key = self.register(frame)

        report = ml.analyse_tool(name=key, target_column="grade")

        self.assertEqual(report["recommended_task"], "Classification")


class TrainabilityGuardTest(unittest.TestCase):

    def setUp(self):
        dataset_store.DATAFRAME_CACHE.clear()

    def register(self, frame, name="t.csv"):
        key = dataset_store.make_dataset_key("guard", name)
        dataset_store.register_dataset(key, frame)
        return key

    def test_identifier_target_is_refused_with_its_numbers(self):
        rows = 500
        frame = pd.DataFrame({
            "code": [f"PC{index:05d}" for index in range(rows)],
            "value": np.random.default_rng(1).normal(0, 1, rows),
        })
        key = self.register(frame)

        report = ml.analyse_tool(name=key, target_column="code")

        self.assertEqual(report["recommended_task"], "Classification")
        self.assertFalse(report["is_trainable"])
        self.assertTrue(report["looks_like_identifier"])
        self.assertIn("code", report["blocking_reason"])
        self.assertIn(str(rows), report["blocking_reason"])

    def test_constant_target_is_refused(self):
        frame = pd.DataFrame({
            "feature": range(50),
            "flat": ["same"] * 50,
        })
        key = self.register(frame)

        report = ml.analyse_tool(name=key, target_column="flat")

        self.assertFalse(report["is_trainable"])
        self.assertIn("same value in every row", report["blocking_reason"])

    def test_single_row_categories_are_counted_in_the_refusal(self):
        frame = pd.DataFrame({
            "feature": np.random.default_rng(2).normal(0, 1, 60),
            "label": ["a"] * 28 + ["b"] * 28 + [
                "rare1", "rare2", "rare3", "rare4"
            ],
        })
        key = self.register(frame)

        report = ml.analyse_tool(name=key, target_column="label")

        self.assertFalse(report["is_trainable"])
        self.assertEqual(report["single_row_categories"], 4)
        self.assertIn("4 of the 6", report["blocking_reason"])

    def test_a_continuous_target_is_never_blocked_for_being_unique(self):
        # A regression target is unique per row by nature. An earlier
        # version of the guard refused every regression dataset.
        key = self.register(regression_frame())

        report = ml.analyse_tool(name=key, target_column="target")

        self.assertTrue(report["is_trainable"])
        self.assertFalse(report["looks_like_identifier"])
        self.assertEqual(report["blocking_reason"], "")

    def test_trainer_explains_the_stratification_failure(self):
        frame = pd.DataFrame({
            "feature": np.random.default_rng(3).normal(0, 1, 60),
            "label": ["a"] * 28 + ["b"] * 28 + [
                "rare1", "rare2", "rare3", "rare4"
            ],
        })
        key = self.register(frame)

        result = ml.train_classification_model(
            name=key,
            target_column="label"
        )

        self.assertFalse(result["success"])
        self.assertIn("only one", result["error"])
        self.assertIn("rows in total", result["error"])


if __name__ == "__main__":
    unittest.main()
