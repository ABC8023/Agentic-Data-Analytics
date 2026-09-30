"""
Session isolation and the filesystem allow-list.

One Streamlit process serves every browser session, so two people who
upload a file with the same name must not see each other's data. Dataset
handles carry a session token to prevent that. Separately, the agent can
name a dataset, so path resolution is confined to one directory.
"""

import os
import tempfile
import unittest

import pandas as pd

os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

import dataset_store  # noqa: E402
import profiling  # noqa: E402


class SessionIsolationTest(unittest.TestCase):

    def setUp(self):
        dataset_store.DATAFRAME_CACHE.clear()

    def test_the_same_file_name_yields_different_handles(self):
        first = dataset_store.make_dataset_key("session_one", "data.csv")
        second = dataset_store.make_dataset_key("session_two", "data.csv")

        self.assertNotEqual(first, second)

    def test_neither_session_can_read_the_other(self):
        first = dataset_store.make_dataset_key("session_one", "data.csv")
        second = dataset_store.make_dataset_key("session_two", "data.csv")

        dataset_store.register_dataset(
            first,
            pd.DataFrame({"only_in_one": [1, 2, 3]})
        )
        dataset_store.register_dataset(
            second,
            pd.DataFrame({"only_in_two": [9, 9, 9]})
        )

        self.assertEqual(
            profiling.get_data_quality_report.invoke({"name": first})
            ["column_names"],
            ["only_in_one"]
        )
        self.assertEqual(
            profiling.get_data_quality_report.invoke({"name": second})
            ["column_names"],
            ["only_in_two"]
        )

    def test_a_bare_file_name_resolves_to_nobody(self):
        dataset_store.register_dataset(
            dataset_store.make_dataset_key("session_one", "data.csv"),
            pd.DataFrame({"secret": [1]})
        )

        result = profiling.get_data_quality_report.invoke(
            {"name": "data.csv"}
        )

        self.assertFalse(result["success"])

    def test_the_display_name_survives_the_handle(self):
        handle = dataset_store.make_dataset_key("token", "quarterly.csv")

        self.assertEqual(
            dataset_store.dataset_display_name(handle),
            "quarterly.csv"
        )

    def test_releasing_one_session_leaves_the_others(self):
        first = dataset_store.make_dataset_key("session_one", "data.csv")
        second = dataset_store.make_dataset_key("session_two", "data.csv")

        dataset_store.register_dataset(first, pd.DataFrame({"a": [1]}))
        dataset_store.register_dataset(second, pd.DataFrame({"b": [1]}))

        released = dataset_store.release_session_datasets("session_one")

        self.assertEqual(released, [first])
        self.assertNotIn(first, dataset_store.DATAFRAME_CACHE)
        self.assertIn(second, dataset_store.DATAFRAME_CACHE)

    def test_the_cache_is_bounded(self):
        for index in range(dataset_store.MAX_CACHED_DATASETS + 6):
            dataset_store.register_dataset(
                dataset_store.make_dataset_key(f"token{index}", "d.csv"),
                pd.DataFrame({"v": [index]})
            )

        self.assertLessEqual(
            len(dataset_store.DATAFRAME_CACHE),
            dataset_store.MAX_CACHED_DATASETS
        )

    def test_the_newest_dataset_survives_eviction(self):
        for index in range(dataset_store.MAX_CACHED_DATASETS + 6):
            newest = dataset_store.make_dataset_key(f"token{index}", "d.csv")
            dataset_store.register_dataset(
                newest,
                pd.DataFrame({"v": [index]})
            )

        self.assertIn(newest, dataset_store.DATAFRAME_CACHE)


class PathAllowListTest(unittest.TestCase):

    def setUp(self):
        dataset_store.DATAFRAME_CACHE.clear()

    def test_an_absolute_path_outside_the_root_is_refused(self):
        outside = os.path.join(
            tempfile.gettempdir(),
            "not_for_the_agent.csv"
        )
        pd.DataFrame({"secret": [1]}).to_csv(outside, index=False)

        result = profiling.get_data_quality_report.invoke(
            {"name": outside}
        )

        self.assertFalse(result["success"])

    def test_parent_traversal_is_refused(self):
        result = profiling.get_data_quality_report.invoke(
            {"name": os.path.join("..", "..", "secrets.csv")}
        )

        self.assertFalse(result["success"])
        self.assertIn("outside", result["error"])

    def test_a_non_csv_name_is_refused(self):
        result = profiling.get_data_quality_report.invoke(
            {"name": "app.py"}
        )

        self.assertFalse(result["success"])
        self.assertIn("not a CSV", result["error"])

    def test_an_empty_name_is_refused(self):
        result = profiling.get_data_quality_report.invoke({"name": "   "})

        self.assertFalse(result["success"])

    def test_a_registered_handle_bypasses_the_filesystem(self):
        # Uploaded data never touches disk, so the allow-list must not
        # stand in the way of a cached handle.
        handle = dataset_store.make_dataset_key("token", "uploaded.csv")
        dataset_store.register_dataset(handle, pd.DataFrame({"n": [1, 2]}))

        result = profiling.get_data_quality_report.invoke(
            {"name": handle}
        )

        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
