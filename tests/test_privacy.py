"""
The contract between local computation and the language model.

These tests are the reason the guarantee can be stated in the README. If
a future tool returns rows, or forgets the guard, the suite fails here
rather than the leak reaching a prompt unnoticed.

An API key is not required. The tools are called directly; only agent
construction needs a key, and that is not exercised.
"""

import os
import unittest

os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

import agent_data_analysis  # noqa: E402
import dataset_store  # noqa: E402
import privacy  # noqa: E402
import profiling  # noqa: E402
import schema  # noqa: E402
from tests.fixtures import (  # noqa: E402
    SECRET_NAMES,
    personal_data_frame,
)

# Every registered tool, with arguments that make it do real work.
TOOL_CALLS = [
    ("list_csv_file", {}),
    ("get_column_schema", {"name": "{key}"}),
    ("get_data_quality_report", {"name": "{key}"}),
    ("get_dataset_summary", {"dataset_paths": ["{key}"]}),
    ("get_stat_overview", {"name": "{key}"}),
    ("get_stat_overview", {"name": "{key}", "n_cate": 10}),
    ("recommend_visualisations", {"name": "{key}"}),
    ("analyse_target_for_ml", {"name": "{key}", "target_column": "band"}),
    (
        "train_classification_model_tool",
        {"name": "{key}", "target_column": "band"},
    ),
]


def resolve(arguments: dict, key: str) -> dict:
    """Substitute the dataset handle into a call's arguments."""

    resolved = {}

    for name, value in arguments.items():
        if isinstance(value, str):
            resolved[name] = value.replace("{key}", key)

        elif isinstance(value, list):
            resolved[name] = [
                item.replace("{key}", key)
                if isinstance(item, str)
                else item
                for item in value
            ]

        else:
            resolved[name] = value

    return resolved


class PrivacyContractTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dataset_store.DATAFRAME_CACHE.clear()
        cls.frame = personal_data_frame()
        cls.key = dataset_store.make_dataset_key("contract", "hr.csv")
        dataset_store.register_dataset(cls.key, cls.frame)
        cls.by_name = {tool.name: tool for tool in agent_data_analysis.tools}

    def test_row_preview_is_not_a_registered_tool(self):
        self.assertNotIn("dataset_preview", self.by_name)

    def test_schema_tool_replaces_the_row_preview(self):
        self.assertIn("get_column_schema", self.by_name)

    def test_every_registered_tool_is_guarded(self):
        for name, tool in self.by_name.items():
            with self.subTest(tool=name):
                self.assertTrue(
                    getattr(tool.func, "__model_safe__", False),
                    f"{name} is not wrapped with privacy.model_safe"
                )

    def test_no_tool_returns_row_level_keys(self):
        for name, arguments in TOOL_CALLS:
            with self.subTest(tool=name):
                payload = self.by_name[name].invoke(
                    resolve(arguments, self.key)
                )

                self.assertEqual(
                    privacy.find_row_level_keys(payload),
                    [],
                    f"{name} returned row level content"
                )

    def test_no_tool_returns_personal_values(self):
        for name, arguments in TOOL_CALLS:
            with self.subTest(tool=name):
                payload = self.by_name[name].invoke(
                    resolve(arguments, self.key)
                )

                self.assertEqual(
                    privacy.contains_any_value(payload, SECRET_NAMES),
                    [],
                    f"{name} exposed personal values"
                )

    def test_identifier_columns_are_withheld_from_the_model(self):
        payload = self.by_name["get_stat_overview"].invoke(
            {"name": self.key}
        )

        categories = payload["categorical_statistics"]

        self.assertIsNone(categories["employee"]["most_common_values"])
        self.assertIsNone(categories["reference"]["most_common_values"])
        self.assertIn(
            "employee",
            payload["identifier_columns_withheld"]
        )

    def test_distinct_counts_survive_the_masking(self):
        payload = self.by_name["get_stat_overview"].invoke(
            {"name": self.key}
        )

        self.assertEqual(
            payload["categorical_statistics"]["employee"]
            ["unique_values"],
            3
        )

    def test_ordinary_labels_remain_available_to_the_model(self):
        payload = self.by_name["get_stat_overview"].invoke(
            {"name": self.key}
        )

        categories = payload["categorical_statistics"]

        self.assertIsNotNone(categories["region"]["most_common_values"])
        self.assertIsNotNone(
            categories["product_name"]["most_common_values"]
        )

    def test_local_interface_keeps_full_detail(self):
        local = profiling.stat_overview(
            name=self.key,
            n_cate=5,
            mask_identifier_labels=False
        )

        self.assertIsNotNone(
            local["categorical_statistics"]["employee"]
            ["most_common_values"]
        )

    def test_local_preview_returns_rows_but_is_not_reachable(self):
        preview = profiling.local_dataset_preview(self.key, 3)

        self.assertIn("records", preview)
        self.assertTrue(
            privacy.contains_any_value(preview, SECRET_NAMES)
        )
        self.assertNotIn(
            "local_dataset_preview",
            {tool.name for tool in agent_data_analysis.tools}
        )


class ScrubberTest(unittest.TestCase):

    def test_nested_row_content_is_removed(self):
        dirty = {
            "success": True,
            "summary": {"mean": 5, "records": [{"a": 1}]},
            "pages": [{"rows": [1, 2, 3], "count": 3}],
            "sample_values": ["x"],
        }

        clean, removed = privacy.scrub_model_payload(dirty)

        self.assertEqual(
            removed,
            ["records", "rows", "sample_values"]
        )
        self.assertEqual(privacy.find_row_level_keys(clean), [])

    def test_legitimate_neighbours_are_kept(self):
        dirty = {"summary": {"mean": 5, "records": [{"a": 1}]}}

        clean, _ = privacy.scrub_model_payload(dirty)

        self.assertEqual(clean["summary"]["mean"], 5)

    def test_a_scalar_count_under_a_forbidden_name_is_kept(self):
        # "rows" is ambiguous: a count is harmless, a list of records is
        # not. Shape decides, so a row count must survive.
        payload = {"rows": 20, "columns": 4}

        clean, removed = privacy.scrub_model_payload(payload)

        self.assertEqual(clean["rows"], 20)
        self.assertEqual(removed, [])

    def test_redaction_is_announced_in_the_payload(self):
        clean, _ = privacy.scrub_model_payload({"records": [{"a": 1}]})

        self.assertIn("privacy_note", clean)

    def test_policy_forbids_row_values(self):
        self.assertFalse(privacy.SHARE_ROW_VALUES)
        self.assertFalse(privacy.SHARE_IDENTIFIER_LABELS)

    def test_policy_statement_is_specific(self):
        statement = privacy.describe_policy()

        self.assertIn("never sent", statement)
        self.assertIn("identifier", statement)


class IdentityHeuristicTest(unittest.TestCase):

    def test_personal_and_account_names_are_recognised(self):
        for name in [
            "employee",
            "customer_name",
            "Customer Name",
            "email",
            "work_email",
            "phone_number",
            "home_address",
            "date_of_birth",
            "iban",
            "creditCard",
        ]:
            with self.subTest(column=name):
                self.assertTrue(schema.name_suggests_identity(name))

    def test_ordinary_labels_are_left_alone(self):
        # These must stay visible: masking them would cost real analytical
        # value for no privacy gain.
        for name in [
            "product_name",
            "model_name",
            "region",
            "band",
            "candidate_id",
            "paid_amount",
            "contact_channel",
        ]:
            with self.subTest(column=name):
                self.assertFalse(schema.name_suggests_identity(name))


if __name__ == "__main__":
    unittest.main()
