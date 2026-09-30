"""Model download: the skops file, the model card and safe loading."""

import io
import json
import os
import unittest
import zipfile

import pandas as pd

os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

import dataset_store  # noqa: E402
import ml  # noqa: E402
import ml_export  # noqa: E402
from tests.fixtures import (  # noqa: E402
    classification_frame,
    messy_frame,
)


class Unfamiliar:
    """A type the app never writes into a model file."""

    def __init__(self):
        self.value = 1


def unzip(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


class ExportTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dataset_store.DATAFRAME_CACHE.clear()
        key = dataset_store.make_dataset_key("export", "clf.csv")
        dataset_store.register_dataset(key, classification_frame())

        cls.result = ml.train_classification_model(
            name=key,
            target_column="churned",
            cv_folds=3
        )
        cls.best = cls.result["best_model_name"]
        cls.files = unzip(ml_export.export_model(cls.result, cls.best))
        cls.card = json.loads(cls.files[ml_export.CARD_FILE])

    def test_the_zip_holds_the_model_card_and_guide(self):
        self.assertEqual(
            set(self.files),
            {
                ml_export.MODEL_FILE,
                ml_export.CARD_FILE,
                ml_export.GUIDE_FILE,
            }
        )

    def test_the_loaded_model_predicts_exactly_as_trained(self):
        loaded = ml_export.load_model(self.files[ml_export.MODEL_FILE])
        original = self.result["trained_models"][self.best]
        rows = self.result["X_test"]

        self.assertEqual(
            list(loaded.predict(rows)),
            list(original.predict(rows))
        )

    def test_the_card_describes_the_model(self):
        card = self.card

        self.assertEqual(card["model"], self.best)
        self.assertTrue(card["chosen_as_best"])
        self.assertEqual(card["task"], "classification")
        self.assertEqual(card["target_column"], "churned")
        self.assertEqual(sorted(card["class_labels"]), ["no", "yes"])
        self.assertEqual(card["rows"]["training"], 160)
        self.assertEqual(card["cross_validation"]["folds"], 3)
        self.assertIn("f1_macro", card["scores"])
        self.assertIn("cv_f1_macro", card["scores"])
        self.assertIn("scikit-learn", card["library_versions"])
        self.assertTrue(card["training_data"].startswith("sha256:"))

    def test_the_card_lists_every_type_the_file_needs(self):
        for name in self.card["trusted_types"]:
            with self.subTest(type=name):
                self.assertTrue(
                    name.startswith(ml_export.TRUSTED_TYPE_PREFIXES)
                )

    def test_the_guide_names_the_feature_columns(self):
        guide = self.files[ml_export.GUIDE_FILE].decode()

        for column in ["amount", "tickets", "plan", "region"]:
            self.assertIn(column, guide)

        self.assertIn("get_untrusted_types", guide)

    def test_any_trained_model_can_be_exported(self):
        for model_name in self.result["trained_models"]:
            with self.subTest(model=model_name):
                files = unzip(ml_export.export_model(self.result, model_name))
                card = json.loads(files[ml_export.CARD_FILE])

                self.assertEqual(card["chosen_as_best"], model_name == self.best)


class DateFeatureCardTest(unittest.TestCase):

    def test_converted_date_columns_are_listed_for_the_reader(self):
        dataset_store.DATAFRAME_CACHE.clear()
        key = dataset_store.make_dataset_key("export", "messy.csv")
        dataset_store.register_dataset(key, messy_frame())

        result = ml.train_classification_model(
            name=key,
            target_column="churned",
            cv_folds=0
        )

        card = json.loads(
            unzip(ml_export.export_model(result, result["best_model_name"]))[
                ml_export.CARD_FILE
            ]
        )

        self.assertEqual(
            card["features"]["dates_converted_to_seconds"],
            ["signup_date"]
        )
        self.assertEqual(
            card["features"]["removed"]["identifier_like"],
            ["customer_id"]
        )


class SafeLoadingTest(unittest.TestCase):

    def test_a_file_with_an_unfamiliar_type_is_refused(self):
        import skops.io as sio

        data = sio.dumps({"payload": Unfamiliar()})

        with self.assertRaises(ml_export.UntrustedModelError) as caught:
            ml_export.load_model(data)

        self.assertIn("Unfamiliar", str(caught.exception))


class FingerprintTest(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        self.y = pd.Series(["p", "q", "p"])

    def test_the_same_rows_give_the_same_fingerprint(self):
        self.assertEqual(
            ml_export.training_fingerprint(self.X, self.y),
            ml_export.training_fingerprint(self.X.copy(), self.y.copy())
        )

    def test_the_index_does_not_change_the_fingerprint(self):
        moved = self.X.set_axis([10, 11, 12])

        self.assertEqual(
            ml_export.training_fingerprint(self.X, self.y),
            ml_export.training_fingerprint(moved, self.y.set_axis([10, 11, 12]))
        )

    def test_one_changed_value_changes_the_fingerprint(self):
        changed = self.X.copy()
        changed.loc[1, "a"] = 2.5

        self.assertNotEqual(
            ml_export.training_fingerprint(self.X, self.y),
            ml_export.training_fingerprint(changed, self.y)
        )

    def test_a_renamed_column_changes_the_fingerprint(self):
        renamed = self.X.rename(columns={"a": "c"})

        self.assertNotEqual(
            ml_export.training_fingerprint(self.X, self.y),
            ml_export.training_fingerprint(renamed, self.y)
        )


if __name__ == "__main__":
    unittest.main()
