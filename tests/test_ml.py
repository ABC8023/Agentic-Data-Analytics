"""Model training, the feature guards, and the split disclosure."""

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


class FeaturePreparationTest(unittest.TestCase):

    def setUp(self):
        dataset_store.DATAFRAME_CACHE.clear()
        rng = np.random.default_rng(21)
        rows = 300

        self.frame = pd.DataFrame({
            "customer_id": [f"CUST-{i:05d}" for i in range(rows)],
            "signup_date": pd.date_range(
                "2024-01-01",
                periods=rows
            ).astype(str),
            "amount": rng.normal(100, 20, rows),
            "region": rng.choice(["north", "south", "east"], rows),
            "source": "one_value",
            "churned": rng.choice(["yes", "no"], rows),
        })

        self.key = dataset_store.make_dataset_key("ml", "features.csv")
        dataset_store.register_dataset(self.key, self.frame)

    def test_identifier_features_are_dropped(self):
        prepared = ml.prepare_ml_data(
            name=self.key,
            target_column="churned"
        )

        self.assertEqual(
            prepared["high_cardinality_columns_removed"],
            ["customer_id"]
        )

    def test_date_text_becomes_one_numeric_feature(self):
        prepared = ml.prepare_ml_data(
            name=self.key,
            target_column="churned"
        )

        self.assertEqual(
            prepared["datetime_columns_converted"],
            ["signup_date"]
        )
        self.assertIn("signup_date", prepared["numeric_columns"])

    def test_constant_features_are_dropped(self):
        prepared = ml.prepare_ml_data(
            name=self.key,
            target_column="churned"
        )

        self.assertEqual(
            prepared["constant_columns_removed"],
            ["source"]
        )

    def test_encoded_width_stays_small(self):
        # Without the identifier guard, one column of 300 unique codes
        # became 300 one-hot columns.
        result = ml.train_classification_model(
            name=self.key,
            target_column="churned"
        )

        pipeline = result["trained_models"]["Random Forest"]
        encoded = pipeline.named_steps["preprocessor"].transform(
            result["X_test"]
        )

        self.assertLess(encoded.shape[1], 20)

    def test_a_dataset_of_only_identifiers_is_refused_clearly(self):
        rows = 60
        frame = pd.DataFrame({
            "ref_a": [f"A{i}" for i in range(rows)],
            "ref_b": [f"B{i}" for i in range(rows)],
            "churned": ["yes", "no"] * (rows // 2),
        })
        key = dataset_store.make_dataset_key("ml", "ids.csv")
        dataset_store.register_dataset(key, frame)

        prepared = ml.prepare_ml_data(
            name=key,
            target_column="churned"
        )

        self.assertFalse(prepared["success"])
        self.assertIn("identifier", prepared["error"])


class ClassificationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dataset_store.DATAFRAME_CACHE.clear()
        cls.key = dataset_store.make_dataset_key("ml", "clf.csv")
        dataset_store.register_dataset(cls.key, classification_frame())
        cls.result = ml.train_classification_model(
            name=cls.key,
            target_column="churned"
        )

    def test_training_succeeds(self):
        self.assertTrue(self.result["success"])

    def test_three_models_are_compared(self):
        self.assertEqual(len(self.result["leaderboard"]), 3)

    def test_leaderboard_is_ranked_by_cross_validated_macro_f1(self):
        # The test rows play no part in the ranking, so the test macro F1
        # need not be in order.
        scores = [
            row["cv_f1_macro"]
            for row in self.result["leaderboard"]
        ]

        self.assertEqual(self.result["ranked_by"], "cv_f1_macro")
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_best_model_matches_the_top_of_the_leaderboard(self):
        self.assertEqual(
            self.result["best_model_name"],
            self.result["leaderboard"][0]["model"]
        )

    def test_confusion_matrix_covers_every_class(self):
        matrix = self.result["confusion_matrices"][
            self.result["best_model_name"]
        ]

        self.assertEqual(
            len(matrix),
            self.result["number_of_classes"]
        )

    def test_split_sizes_reconcile_with_the_row_count(self):
        self.assertEqual(
            self.result["training_rows"] + self.result["testing_rows"],
            200
        )

    def test_no_adjustment_is_claimed_when_none_happened(self):
        self.assertFalse(self.result["test_size_adjusted"])
        self.assertEqual(self.result["test_size_adjustment_reason"], "")
        self.assertAlmostEqual(
            self.result["actual_test_size"],
            0.2,
            places=2
        )


class SplitDisclosureTest(unittest.TestCase):

    def setUp(self):
        dataset_store.DATAFRAME_CACHE.clear()

    def test_many_classes_force_a_larger_test_set_and_say_so(self):
        rows = 120
        frame = pd.DataFrame({
            "feature": np.random.default_rng(4).normal(0, 1, rows),
            "cls": [f"c{index % 30}" for index in range(rows)],
        })
        key = dataset_store.make_dataset_key("split", "many.csv")
        dataset_store.register_dataset(key, frame)

        result = ml.train_classification_model(
            name=key,
            target_column="cls",
            test_size=0.1
        )

        self.assertTrue(result["test_size_adjusted"])
        self.assertEqual(result["requested_test_size"], 0.1)
        self.assertAlmostEqual(result["actual_test_size"], 0.25, places=2)
        self.assertIn("stratified", result["test_size_adjustment_reason"])

    def test_a_tiny_regression_set_gets_the_two_row_minimum(self):
        frame = pd.DataFrame({
            "x": np.arange(8, dtype=float),
            "target": np.arange(8, dtype=float) * 2,
        })
        key = dataset_store.make_dataset_key("split", "tiny.csv")
        dataset_store.register_dataset(key, frame)

        result = ml.train_regression_model(
            name=key,
            target_column="target",
            test_size=0.1
        )

        self.assertTrue(result["test_size_adjusted"])
        self.assertEqual(result["testing_rows"], 2)

    def test_an_out_of_range_test_size_is_refused(self):
        key = dataset_store.make_dataset_key("split", "clf.csv")
        dataset_store.register_dataset(key, classification_frame())

        for bad in [0, 0.6, 1.0]:
            with self.subTest(test_size=bad):
                result = ml.train_classification_model(
                    name=key,
                    target_column="churned",
                    test_size=bad
                )

                self.assertFalse(result["success"])


class RegressionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dataset_store.DATAFRAME_CACHE.clear()
        cls.key = dataset_store.make_dataset_key("ml", "reg.csv")
        dataset_store.register_dataset(cls.key, regression_frame())
        cls.result = ml.train_regression_model(
            name=cls.key,
            target_column="target"
        )

    def test_training_succeeds(self):
        self.assertTrue(self.result["success"])

    def test_leaderboard_is_ranked_by_cross_validated_rmse_ascending(self):
        scores = [row["cv_rmse"] for row in self.result["leaderboard"]]

        self.assertEqual(self.result["ranked_by"], "cv_rmse")
        self.assertEqual(scores, sorted(scores))

    def test_linear_signal_is_recovered(self):
        # The fixture is linear with small noise, so a linear model should
        # win and explain nearly all the variance.
        self.assertEqual(
            self.result["best_model_name"],
            "Linear Regression"
        )
        self.assertGreater(
            self.result["leaderboard"][0]["r2_score"],
            0.9
        )

    def test_a_text_target_is_refused_for_regression(self):
        key = dataset_store.make_dataset_key("ml", "clf2.csv")
        dataset_store.register_dataset(key, classification_frame())

        result = ml.train_regression_model(
            name=key,
            target_column="churned"
        )

        self.assertFalse(result["success"])
        self.assertIn("numerical target", result["error"])


class FeatureImportanceTest(unittest.TestCase):

    def test_importance_ranks_the_columns_that_carry_the_signal(self):
        dataset_store.DATAFRAME_CACHE.clear()
        key = dataset_store.make_dataset_key("ml", "reg.csv")
        dataset_store.register_dataset(key, regression_frame())

        trained = ml.train_regression_model(
            name=key,
            target_column="target"
        )

        importance = ml.calculate_feature_importance(
            trained_model=trained["trained_models"][
                trained["best_model_name"]
            ],
            X_test=trained["X_test"],
            y_test=trained["y_test"],
            task="regression",
            n_repeats=3
        )

        self.assertTrue(importance["success"])

        ranked = [
            entry["feature"]
            for entry in importance["feature_importance"]
        ]

        self.assertEqual(ranked[0], "x2")

    def test_an_unknown_task_is_refused(self):
        dataset_store.DATAFRAME_CACHE.clear()
        key = dataset_store.make_dataset_key("ml", "reg2.csv")
        dataset_store.register_dataset(key, regression_frame())

        trained = ml.train_regression_model(
            name=key,
            target_column="target"
        )

        importance = ml.calculate_feature_importance(
            trained_model=trained["trained_models"][
                trained["best_model_name"]
            ],
            X_test=trained["X_test"],
            y_test=trained["y_test"],
            task="clustering",
            n_repeats=2
        )

        self.assertFalse(importance["success"])


if __name__ == "__main__":
    unittest.main()
