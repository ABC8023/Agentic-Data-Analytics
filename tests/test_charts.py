"""Chart recommendation and figure construction."""

import os
import unittest

import numpy as np
import pandas as pd

os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

import charts  # noqa: E402
import dataset_store  # noqa: E402
import profiling  # noqa: E402

ALL_CHART_TYPES = {
    "histogram",
    "box",
    "bar",
    "scatter",
    "correlation_heatmap",
    "line",
}


def rich_frame(rows: int = 120) -> pd.DataFrame:
    """A dataset that deserves every chart type the app can draw."""

    rng = np.random.default_rng(31)

    return pd.DataFrame({
        "day": pd.date_range("2024-01-01", periods=rows).astype(str),
        "revenue": rng.normal(1000, 100, rows),
        "cost": rng.normal(600, 80, rows),
        "units": rng.normal(50, 10, rows),
        "region": rng.choice(["n", "s", "e", "w"], rows),
        "channel": rng.choice(["web", "store", "phone"], rows),
        "segment": rng.choice(["smb", "mid", "ent"], rows),
        "tier": rng.choice(["gold", "silver"], rows),
    })


class RecommendationTest(unittest.TestCase):

    def setUp(self):
        dataset_store.DATAFRAME_CACHE.clear()
        self.key = dataset_store.make_dataset_key("charts", "rich.csv")
        dataset_store.register_dataset(self.key, rich_frame())

    def test_every_family_appears_at_the_default_limit(self):
        # Before family quotas, histograms and bars consumed the whole
        # budget and the heatmap and line chart were unreachable.
        result = charts.recommend_visualisations.invoke(
            {"name": self.key}
        )

        self.assertEqual(
            set(result["recommended_chart_types"]),
            ALL_CHART_TYPES
        )

    def test_a_small_limit_still_spreads_across_families(self):
        for limit in [1, 2, 3, 4, 6]:
            with self.subTest(limit=limit):
                result = charts.recommend_visualisations.invoke({
                    "name": self.key,
                    "max_recommendation": limit
                })

                self.assertEqual(
                    result["recommendation_count"],
                    limit
                )
                self.assertEqual(
                    len(set(result["recommended_chart_types"])),
                    limit
                )

    def test_the_limit_is_never_exceeded(self):
        for limit in [1, 5, 8, 12, 15]:
            with self.subTest(limit=limit):
                result = charts.recommend_visualisations.invoke({
                    "name": self.key,
                    "max_recommendation": limit
                })

                self.assertLessEqual(
                    result["recommendation_count"],
                    limit
                )

    def test_an_out_of_range_limit_is_refused(self):
        for limit in [0, -1, 16]:
            with self.subTest(limit=limit):
                result = charts.recommend_visualisations.invoke({
                    "name": self.key,
                    "max_recommendation": limit
                })

                self.assertFalse(result["success"])

    def test_every_recommendation_can_actually_be_drawn(self):
        result = charts.recommend_visualisations.invoke(
            {"name": self.key}
        )

        for specification in result["recommendations"]:
            with self.subTest(chart=specification["chart_type"]):
                figure = charts.create_plotly_figure(
                    name=self.key,
                    chart_specification=specification
                )

                self.assertIsNotNone(figure)


class DegradedDatasetTest(unittest.TestCase):

    def setUp(self):
        dataset_store.DATAFRAME_CACHE.clear()

    def register(self, frame, name):
        key = dataset_store.make_dataset_key("charts", name)
        dataset_store.register_dataset(key, frame)
        return key

    def test_two_numeric_columns_only(self):
        rng = np.random.default_rng(32)
        key = self.register(
            pd.DataFrame({
                "a": rng.normal(0, 1, 60),
                "b": rng.normal(0, 1, 60),
            }),
            "plain.csv"
        )

        result = charts.recommend_visualisations.invoke({"name": key})

        self.assertNotIn("line", result["recommended_chart_types"])

        for specification in result["recommendations"]:
            charts.create_plotly_figure(
                name=key,
                chart_specification=specification
            )

    def test_a_single_column(self):
        rng = np.random.default_rng(33)
        key = self.register(
            pd.DataFrame({"only": rng.normal(0, 1, 40)}),
            "single.csv"
        )

        result = charts.recommend_visualisations.invoke({"name": key})

        self.assertNotIn("scatter", result["recommended_chart_types"])

        for specification in result["recommendations"]:
            charts.create_plotly_figure(
                name=key,
                chart_specification=specification
            )


class FigureValidationTest(unittest.TestCase):

    def setUp(self):
        dataset_store.DATAFRAME_CACHE.clear()
        self.key = dataset_store.make_dataset_key("charts", "rich.csv")
        dataset_store.register_dataset(self.key, rich_frame())

    def test_an_unknown_chart_type_is_refused(self):
        with self.assertRaises(ValueError):
            charts.create_plotly_figure(
                name=self.key,
                chart_specification={
                    "chart_type": "sunburst",
                    "x": "revenue",
                }
            )

    def test_a_missing_column_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            charts.create_plotly_figure(
                name=self.key,
                chart_specification={
                    "chart_type": "histogram",
                    "x": "not_a_column",
                }
            )

        self.assertIn("not_a_column", str(caught.exception))

    def test_a_histogram_needs_a_numeric_column(self):
        with self.assertRaises(ValueError):
            charts.create_plotly_figure(
                name=self.key,
                chart_specification={
                    "chart_type": "histogram",
                    "x": "region",
                }
            )

    def test_a_heatmap_needs_two_numeric_columns(self):
        with self.assertRaises(ValueError):
            charts.create_plotly_figure(
                name=self.key,
                chart_specification={
                    "chart_type": "correlation_heatmap",
                    "x": ["revenue"],
                }
            )

    def test_a_date_column_is_detected_for_the_line_chart(self):
        detected = profiling.data_time_column(rich_frame())

        self.assertIn("day", detected)


if __name__ == "__main__":
    unittest.main()
