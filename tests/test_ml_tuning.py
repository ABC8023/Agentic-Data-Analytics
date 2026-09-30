"""Cross-validation and hyperparameter search in the training functions."""

import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

import dataset_store  # noqa: E402
import ml  # noqa: E402
import ml_tuning  # noqa: E402
from tests.fixtures import (  # noqa: E402
    classification_frame,
    regression_frame,
)


def register(frame, name):
    key = dataset_store.make_dataset_key("tuning", name)
    dataset_store.register_dataset(key, frame)

    return key


class FoldPlanTest(unittest.TestCase):

    def test_fewer_than_two_folds_switches_cross_validation_off(self):
        plan = ml_tuning.plan_folds(pd.Series(["a", "b"] * 10), "classification", 0)

        self.assertEqual(plan.folds, 0)
        self.assertIn("switched off", plan.note)

    def test_a_rare_class_reduces_the_folds_and_says_why(self):
        y = pd.Series(["common"] * 40 + ["rare"] * 3)

        plan = ml_tuning.plan_folds(y, "classification", 5)

        self.assertEqual(plan.folds, 3)
        self.assertIn("rarest class", plan.note)

    def test_a_single_row_class_skips_cross_validation(self):
        y = pd.Series(["common"] * 40 + ["rare"])

        plan = ml_tuning.plan_folds(y, "classification", 5)

        self.assertEqual(plan.folds, 0)
        self.assertIn("skipped", plan.note)

    def test_regression_keeps_two_rows_per_fold(self):
        plan = ml_tuning.plan_folds(pd.Series(np.arange(6.0)), "regression", 5)

        self.assertEqual(plan.folds, 3)

    def test_the_request_is_capped(self):
        plan = ml_tuning.plan_folds(pd.Series(np.arange(500.0)), "regression", 50)

        self.assertEqual(plan.folds, ml_tuning.MAX_FOLDS)

    def test_an_ample_request_is_used_as_is(self):
        plan = ml_tuning.plan_folds(pd.Series(["a", "b"] * 50), "classification", 5)

        self.assertEqual(plan.folds, 5)
        self.assertEqual(plan.note, "")


class CrossValidationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dataset_store.DATAFRAME_CACHE.clear()
        cls.key = register(classification_frame(), "clf.csv")
        cls.result = ml.train_classification_model(
            name=cls.key,
            target_column="churned"
        )

    def test_every_model_has_a_cross_validated_score_and_spread(self):
        for row in self.result["leaderboard"]:
            with self.subTest(model=row["model"]):
                self.assertGreaterEqual(row["cv_f1_macro"], 0.0)
                self.assertLessEqual(row["cv_f1_macro"], 1.0)
                self.assertGreaterEqual(row["cv_f1_macro_std"], 0.0)

    def test_the_summary_records_what_ran(self):
        summary = self.result["cross_validation"]

        self.assertEqual(summary["folds"], 5)
        self.assertEqual(summary["scoring"], "macro F1")
        self.assertFalse(summary["tuned"])
        self.assertEqual(self.result["tuned_parameters"], {})

    def test_the_test_rows_never_reach_cross_validation(self):
        # Any overlap would let the test score help choose the model.
        seen = []
        real = ml_tuning.fit_and_score

        def spy(pipeline, X_train, y_train, **kwargs):
            seen.append(set(X_train.index))

            return real(pipeline, X_train, y_train, **kwargs)

        with mock.patch.object(ml_tuning, "fit_and_score", side_effect=spy):
            result = ml.train_classification_model(
                name=self.key,
                target_column="churned"
            )

        test_index = set(result["X_test"].index)

        self.assertEqual(len(seen), 3)

        for training_index in seen:
            self.assertFalse(training_index & test_index)

    def test_switching_folds_off_ranks_on_the_test_score(self):
        result = ml.train_classification_model(
            name=self.key,
            target_column="churned",
            cv_folds=0
        )

        scores = [row["f1_macro"] for row in result["leaderboard"]]

        self.assertEqual(result["ranked_by"], "f1_macro")
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIsNone(result["leaderboard"][0]["cv_f1_macro"])

    def test_the_agent_tool_reports_the_comparison(self):
        output = ml.train_classification_model_tool.invoke({
            "name": self.key,
            "target_column": "churned",
        })

        self.assertEqual(output["cross_validation"]["folds"], 5)
        self.assertEqual(output["ranked_by"], "cv_f1_macro")


class TuningTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dataset_store.DATAFRAME_CACHE.clear()
        cls.key = register(regression_frame(), "reg.csv")
        cls.result = ml.train_regression_model(
            name=cls.key,
            target_column="target",
            cv_folds=3,
            tune=True
        )

    def test_the_chosen_settings_come_from_the_search_space(self):
        tuned = self.result["tuned_parameters"]

        self.assertEqual(set(tuned), {"Decision Tree", "Random Forest"})

        for model_name, settings in tuned.items():
            space = ml_tuning.search_space("regression", model_name)

            for name, value in settings.items():
                with self.subTest(model=model_name, setting=name):
                    self.assertIn(value, space[f"model__{name}"])

    def test_the_fitted_model_uses_the_chosen_settings(self):
        settings = self.result["tuned_parameters"]["Random Forest"]
        model = self.result["trained_models"]["Random Forest"].named_steps[
            "model"
        ]

        for name, value in settings.items():
            self.assertEqual(model.get_params()[name], value)

    def test_the_summary_records_the_search(self):
        summary = self.result["cross_validation"]

        self.assertTrue(summary["tuned"])
        self.assertEqual(summary["folds"], 3)
        self.assertEqual(
            summary["search_iterations"],
            ml_tuning.SEARCH_ITERATIONS
        )

    def test_the_linear_signal_still_wins(self):
        self.assertEqual(self.result["best_model_name"], "Linear Regression")

    def test_tuning_without_folds_is_declined_and_explained(self):
        result = ml.train_regression_model(
            name=self.key,
            target_column="target",
            cv_folds=0,
            tune=True
        )

        self.assertFalse(result["cross_validation"]["tuned"])
        self.assertIn("not tuned", result["cross_validation"]["note"])
        self.assertEqual(result["tuned_parameters"], {})

    def test_logistic_regression_searches_its_regularisation(self):
        space = ml_tuning.search_space("classification", "Logistic Regression")

        self.assertEqual(list(space), ["model__C"])


if __name__ == "__main__":
    unittest.main()
