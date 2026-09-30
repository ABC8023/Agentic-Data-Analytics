"""
The application script actually runs.

These tests execute app.py through Streamlit's own harness. They catch
the class of failure that unit tests cannot see: a page that raises while
rendering, a widget whose state was discarded between reruns, or a
control offered for an action that cannot succeed.

No API key is needed. The AI tab must degrade rather than fail.
"""

import os
import unittest

os.environ["GOOGLE_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.fixtures import (  # noqa: E402
    classification_frame,
    messy_frame,
    regression_frame,
    timeline_frame,
)

TIMEOUT = 600

# AppTest resolves a relative path against the file that calls it, which
# would look for app.py inside tests/. Point at the repository root.
APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "app.py"
)

CHART_TYPES = [
    "Histogram",
    "Box plot",
    "Bar chart",
    "Scatter plot",
    "Correlation heatmap",
    "Time-series line chart",
]


def start(frame=None, name="fixture.csv"):
    """Run app.py, optionally with a dataset already loaded."""

    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)

    if frame is not None:
        app.session_state["dataframe"] = frame
        app.session_state["original_dataframe"] = frame.copy()
        app.session_state["name"] = name
        app.session_state["upload_signature"] = (name, len(frame), None)

    app.run()

    return app


def errors(app):
    return [exception.value for exception in app.exception]


def selectbox(app, suffix):
    """
    Find a select box by the stable tail of its key.

    Widget keys are prefixed with the dataset handle, which changes every
    session. Looking widgets up by label stopped working once two tabs
    offered a control called "Measure": the first match won, so a test
    aimed at one tab silently drove the other.
    """

    return app.selectbox(
        key=f"{app.session_state['dataset_key']}{suffix}"
    )


def toggle(app, suffix):
    """Find a toggle by the stable tail of its key."""

    return app.toggle(
        key=f"{app.session_state['dataset_key']}{suffix}"
    )


class LandingTest(unittest.TestCase):

    def test_the_page_renders_without_a_dataset_or_a_key(self):
        app = start()

        self.assertEqual(errors(app), [])

    def test_the_uploader_advertises_every_supported_format(self):
        app = start()

        labels = [
            uploader.label
            for uploader in app.get("file_uploader")
        ]

        self.assertTrue(labels)
        self.assertIn("CSV", labels[0])
        self.assertIn("XLSX", labels[0])

    def test_a_sample_dataset_can_be_opened_without_a_file(self):
        app = start()

        app.button(key="sample_store_orders").click().run()

        self.assertEqual(errors(app), [])
        self.assertEqual(app.session_state["name"], "store_orders.csv")
        self.assertEqual(len(app.session_state["dataframe"]), 3660)
        self.assertEqual(app.tabs[0].label, "Dashboard")

    def test_a_sample_can_be_opened_from_the_link(self):
        app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        app.query_params["sample"] = "iris"
        app.run()

        self.assertEqual(errors(app), [])
        self.assertEqual(app.session_state["name"], "iris.csv")

    def test_an_unknown_sample_in_the_link_is_ignored(self):
        app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        app.query_params["sample"] = "../.env"
        app.run()

        self.assertEqual(errors(app), [])
        self.assertIsNone(app.session_state["dataframe"])

    def test_the_name_is_not_a_link_on_the_start_page(self):
        app = start()

        keys = [button.key for button in app.button]
        markdown = " ".join(block.value for block in app.markdown)

        self.assertNotIn("brand_home", keys)
        self.assertIn('aria-current="page"', markdown)
        self.assertIn("Answers you can check", markdown)

    def test_the_bar_says_whether_a_model_is_set_up(self):
        app = start()

        markdown = " ".join(block.value for block in app.markdown)

        self.assertIn("AI questions off", markdown)

    def test_the_name_leads_back_to_the_start_page(self):
        app = start()
        app.button(key="sample_store_orders").click().run()

        self.assertEqual(app.session_state["name"], "store_orders.csv")

        markdown = " ".join(block.value for block in app.markdown)
        self.assertIn("Answers you can check", markdown)

        app.button(key="brand_home").click().run()

        self.assertEqual(errors(app), [])
        self.assertIsNone(app.session_state["dataframe"])
        self.assertEqual(app.session_state["agent_chat_messages"], [])
        self.assertEqual(len(app.tabs), 0)
        self.assertTrue(any(b.key == "sample_store_orders" for b in app.button))

    def test_going_home_from_a_sample_link_does_not_reload_it(self):
        app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        app.query_params["sample"] = "iris"
        app.run()

        self.assertEqual(app.session_state["name"], "iris.csv")

        app.button(key="brand_home").click().run()

        self.assertIsNone(app.session_state["dataframe"])
        self.assertNotIn("sample", app.query_params)

    def test_the_header_has_no_menus_or_demo_button(self):
        app = start()

        keys = [button.key for button in app.button]

        self.assertEqual(list(app.get("popover")), [])
        self.assertNotIn("topbar_cta", keys)

    def test_the_capability_cards_are_shown(self):
        app = start()

        markdown = " ".join(block.value for block in app.markdown)

        self.assertIn("What opens once a table is loaded", markdown)


class LoadedDatasetTest(unittest.TestCase):

    def test_all_six_tabs_render(self):
        app = start(classification_frame())

        self.assertEqual(errors(app), [])
        self.assertEqual(len(app.tabs), 6)

    def test_tab_labels_are_sentence_case_and_spelled_correctly(self):
        app = start(classification_frame())

        labels = [tab.label for tab in app.tabs]

        self.assertEqual(
            labels,
            [
                "Dataset overview",
                "Statistics",
                "Data visualisation",
                "Machine learning",
                "AI analysis",
                "Data cleaning",
            ]
        )

    def test_the_privacy_policy_is_visible(self):
        app = start(classification_frame())

        captions = [caption.value for caption in app.caption]

        self.assertTrue(
            any("never sent" in caption for caption in captions)
        )

    def test_the_chat_input_works_without_a_key(self):
        # Questions about figures, periods and segments are answered by
        # arithmetic here, so a missing key is a reduced service rather
        # than no service.
        app = start(classification_frame())

        self.assertFalse(app.chat_input[0].disabled)


class ChartBuilderTest(unittest.TestCase):

    def test_every_chart_type_renders_or_declines_cleanly(self):
        app = start(messy_frame())

        for chart in CHART_TYPES:
            with self.subTest(chart=chart):
                selector = next(
                    box for box in app.selectbox
                    if box.label == "Select chart type"
                )
                selector.select(chart).run()

                self.assertEqual(errors(app), [])

    def test_recommendations_cover_every_family(self):
        app = start(messy_frame())

        generate = next(
            button for button in app.button
            if button.label == "Generate recommended visualisations"
        )
        generate.click().run()

        self.assertEqual(errors(app), [])

        recommended = app.session_state["recommendation_result"]

        self.assertIn(
            "correlation_heatmap",
            recommended["recommended_chart_types"]
        )
        self.assertIn("line", recommended["recommended_chart_types"])

    def test_the_recommendation_limit_reaches_the_engine(self):
        app = start(messy_frame())

        slider = next(
            control for control in app.slider
            if control.label == "Maximum number of recommended charts"
        )
        slider.set_value(3).run()

        generate = next(
            button for button in app.button
            if button.label == "Generate recommended visualisations"
        )
        generate.click().run()

        self.assertEqual(
            app.session_state["recommendation_result"]
            ["recommendation_count"],
            3
        )


class MachineLearningTabTest(unittest.TestCase):

    def test_classification_trains_and_reports(self):
        app = start(classification_frame())

        app.selectbox(key="ml_target_column").select("churned").run()
        app.button(key="train_ml_models").click().run()

        self.assertEqual(errors(app), [])
        self.assertTrue(
            any(
                "trained successfully" in message.value
                for message in app.success
            )
        )

    def test_regression_is_recommended_and_trains(self):
        app = start(regression_frame())

        app.selectbox(key="ml_target_column").select("target").run()

        metrics = {metric.label: metric.value for metric in app.metric}

        self.assertEqual(metrics.get("Recommended task"), "Regression")

        app.button(key="train_ml_models").click().run()

        self.assertEqual(errors(app), [])

    def test_feature_handling_is_disclosed(self):
        app = start(messy_frame())

        app.selectbox(key="ml_target_column").select("churned").run()
        app.button(key="train_ml_models").click().run()

        warnings = " ".join(warning.value for warning in app.warning)
        notes = " ".join(note.value for note in app.info)

        self.assertIn("customer_id", warnings)
        self.assertIn("signup_date", notes)
        self.assertIn("source", notes)

    def test_feature_importance_runs(self):
        app = start(classification_frame())

        app.selectbox(key="ml_target_column").select("churned").run()
        app.button(key="train_ml_models").click().run()
        app.button(key="calculate_feature_importance").click().run()

        self.assertEqual(errors(app), [])

    def test_an_identifier_target_disables_training(self):
        app = start(messy_frame())

        app.selectbox(key="ml_target_column").select("customer_id").run()

        self.assertEqual(errors(app), [])
        self.assertTrue(app.button(key="train_ml_models").disabled)
        self.assertTrue(
            any(
                "identifier" in message.value
                for message in app.error
            )
        )

    def test_choosing_a_usable_target_re_enables_training(self):
        app = start(messy_frame())

        app.selectbox(key="ml_target_column").select("customer_id").run()
        app.selectbox(key="ml_target_column").select("churned").run()

        self.assertFalse(app.button(key="train_ml_models").disabled)

    def test_the_ranking_is_explained_and_the_model_can_be_downloaded(self):
        app = start(classification_frame())

        app.selectbox(key="ml_target_column").select("churned").run()
        app.button(key="train_ml_models").click().run()

        captions = " ".join(caption.value for caption in app.caption)
        downloads = [button.label for button in app.get("download_button")]

        self.assertEqual(errors(app), [])
        self.assertIn("averaged over 5 folds", captions)
        self.assertIn("Download best model", downloads)

    def test_tuning_shows_the_settings_it_chose(self):
        app = start(regression_frame())

        app.selectbox(key="ml_target_column").select("target").run()
        app.checkbox(key="ml_tune").check().run()
        app.button(key="train_ml_models").click().run()

        captions = " ".join(caption.value for caption in app.caption)

        self.assertEqual(errors(app), [])
        self.assertTrue(app.session_state["ml_result"]["tuned_parameters"])
        self.assertIn("slightly optimistic", captions)
        self.assertIn("Linear Regression kept the default", captions)

    def test_switching_cross_validation_off_ranks_on_the_test_rows(self):
        app = start(classification_frame())

        app.selectbox(key="ml_target_column").select("churned").run()
        app.checkbox(key="ml_cross_validate").uncheck().run()

        self.assertTrue(app.checkbox(key="ml_tune").disabled)

        app.button(key="train_ml_models").click().run()

        captions = " ".join(caption.value for caption in app.caption)

        self.assertEqual(errors(app), [])
        self.assertIn("because cross-validation didn't run", captions)
        self.assertEqual(
            app.session_state["ml_result"]["ranked_by"],
            "f1_macro"
        )


class CleaningTabTest(unittest.TestCase):

    def test_preview_apply_and_restore_survive_the_rerun(self):
        # Applying a cleaned dataset calls st.rerun() from a tab that
        # renders before the machine learning tab, which once discarded
        # the task radio's state and crashed the next run.
        app = start(messy_frame())

        app.button(key="preview_cleaned_dataset").click().run()
        self.assertEqual(errors(app), [])

        app.button(key="apply_cleaned_dataset").click().run()
        self.assertEqual(errors(app), [])

        app.button(key="restore_original_dataset").click().run()
        self.assertEqual(errors(app), [])

    def test_applying_a_clean_dataset_changes_the_reported_shape(self):
        app = start(messy_frame())

        before = {
            metric.label: metric.value
            for metric in app.metric
        }["Rows"]

        app.button(key="preview_cleaned_dataset").click().run()
        app.button(key="apply_cleaned_dataset").click().run()

        after = {
            metric.label: metric.value
            for metric in app.metric
        }["Rows"]

        self.assertNotEqual(before, after)

    def test_the_data_version_advances_so_caches_invalidate(self):
        app = start(messy_frame())

        before = app.session_state["data_version"]

        app.button(key="preview_cleaned_dataset").click().run()
        app.button(key="apply_cleaned_dataset").click().run()

        self.assertGreater(app.session_state["data_version"], before)


class CalculationTrailTest(unittest.TestCase):
    """
    The trail is rendered from stored chat history, so these tests need no
    API key and no model call.
    """

    def history(self, trail):
        return [
            {"role": "user", "content": "how many rows?"},
            {
                "role": "assistant",
                "content": "The dataset has 4,339 rows.",
                "trail": trail,
            },
        ]

    def test_a_populated_trail_names_the_tool_and_its_figures(self):
        app = start(classification_frame())
        app.session_state["agent_chat_messages"] = self.history([
            {
                "step": 1,
                "tool": "get_data_quality_report",
                "arguments": {"name": "fixture.csv"},
                "result": {"rows": 4339},
                "highlights": ["rows: 4,339", "duplicate_rows: 0"],
                "failed": False,
                "error": "",
            }
        ])
        app.run()

        self.assertEqual(errors(app), [])

        markdown = " ".join(block.value for block in app.markdown)

        self.assertIn("get_data_quality_report", markdown)
        self.assertIn("rows: 4,339", markdown)

    def test_an_empty_trail_warns_that_nothing_was_computed(self):
        app = start(classification_frame())
        app.session_state["agent_chat_messages"] = self.history([])
        app.run()

        self.assertEqual(errors(app), [])

        captions = [caption.value for caption in app.caption]

        self.assertTrue(
            any("No calculation ran" in caption for caption in captions)
        )

    def test_a_failed_step_is_surfaced_as_a_warning(self):
        app = start(classification_frame())
        app.session_state["agent_chat_messages"] = self.history([
            {
                "step": 1,
                "tool": "get_data_quality_report",
                "arguments": {"name": "missing.csv"},
                "result": {"success": False, "error": "file not found"},
                "highlights": [],
                "failed": True,
                "error": "file not found",
            }
        ])
        app.run()

        self.assertEqual(errors(app), [])

        warnings = " ".join(warning.value for warning in app.warning)

        self.assertIn("file not found", warnings)

    def test_an_answer_stored_without_a_trail_key_still_renders(self):
        # Conversations saved before the trail existed must not crash the
        # history rendering.
        app = start(classification_frame())
        app.session_state["agent_chat_messages"] = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        app.run()

        self.assertEqual(errors(app), [])

    def test_several_answers_each_get_their_own_trail(self):
        app = start(classification_frame())
        app.session_state["agent_chat_messages"] = [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "one",
                "trail": [{
                    "step": 1,
                    "tool": "get_column_schema",
                    "arguments": {},
                    "result": {"rows": 1},
                    "highlights": ["rows: 1"],
                    "failed": False,
                    "error": "",
                }],
            },
            {"role": "user", "content": "second"},
            {
                "role": "assistant",
                "content": "two",
                "trail": [{
                    "step": 1,
                    "tool": "get_stat_overview",
                    "arguments": {},
                    "result": {"rows": 2},
                    "highlights": ["rows: 2"],
                    "failed": False,
                    "error": "",
                }],
            },
        ]
        app.run()

        self.assertEqual(errors(app), [])

        markdown = " ".join(block.value for block in app.markdown)

        self.assertIn("get_column_schema", markdown)
        self.assertIn("get_stat_overview", markdown)


class TimeSeriesTabTest(unittest.TestCase):

    def test_the_tab_is_absent_when_the_data_has_no_dates(self):
        app = start(classification_frame())

        self.assertNotIn(
            "Time series",
            [tab.label for tab in app.tabs]
        )

    def test_the_tab_appears_when_the_data_has_dates(self):
        app = start(timeline_frame())

        self.assertEqual(errors(app), [])
        self.assertIn(
            "Time series",
            [tab.label for tab in app.tabs]
        )

    def test_the_tab_sits_after_the_visualisation_tab(self):
        app = start(timeline_frame())

        labels = [tab.label for tab in app.tabs]

        self.assertEqual(
            labels.index("Time series"),
            labels.index("Data visualisation") + 1
        )

    def test_the_timeline_renders_a_chart(self):
        app = start(timeline_frame())

        self.assertEqual(errors(app), [])
        self.assertGreater(len(app.get("plotly_chart")), 0)

    def test_the_period_summary_is_shown(self):
        app = start(timeline_frame())

        labels = [metric.label for metric in app.metric]

        self.assertIn("Periods", labels)
        self.assertIn("Latest period", labels)
        self.assertIn("Direction", labels)

    def test_a_planted_spike_is_listed_as_unusual(self):
        app = start(timeline_frame(spike_day=400))

        self.assertEqual(errors(app), [])

        tables = [
            frame.value
            for frame in app.dataframe
            if hasattr(frame.value, "columns")
            and "Deviations from trend" in list(frame.value.columns)
        ]

        self.assertTrue(tables, "no anomaly table was rendered")
        self.assertGreaterEqual(len(tables[0]), 1)

    def test_unequal_month_lengths_are_called_out(self):
        # Totalling daily data by month produces a saw-tooth: February is
        # short, not weak. The tab has to say so, because otherwise the
        # anomaly detector flags the calendar.
        app = start(timeline_frame(spike_day=None))

        captions = " ".join(caption.value for caption in app.caption)

        self.assertIn("different numbers of days", captions)
        self.assertIn("sum_per_day", captions)

    def test_a_clean_series_reports_nothing_unusual_as_a_daily_rate(self):
        app = start(timeline_frame(spike_day=None))

        selectbox(app, "_ts_aggregation").select("sum_per_day").run()

        self.assertEqual(errors(app), [])

        text = " ".join(block.value for block in app.markdown)

        self.assertIn("inside the expected band", text)

    def test_the_projection_is_shown_with_its_verdict(self):
        app = start(timeline_frame())

        self.assertEqual(errors(app), [])

        spoken = " ".join(
            [message.value for message in app.success]
            + [message.value for message in app.warning]
        )

        self.assertIn("assuming no change", spoken)

    def test_the_measure_can_be_changed(self):
        app = start(timeline_frame())

        selectbox(app, "_ts_measure").select("units").run()

        self.assertEqual(errors(app), [])

    def test_counting_rows_needs_no_measure_column(self):
        app = start(timeline_frame())

        selectbox(app, "_ts_measure").select("Number of rows").run()

        self.assertEqual(errors(app), [])

    def test_the_period_grain_can_be_overridden(self):
        app = start(timeline_frame())

        selectbox(app, "_ts_grain").select("quarter").run()

        self.assertEqual(errors(app), [])

    def test_the_aggregation_can_be_changed(self):
        app = start(timeline_frame())

        selectbox(app, "_ts_aggregation").select("mean").run()

        self.assertEqual(errors(app), [])

    def test_a_short_history_explains_itself_instead_of_failing(self):
        # Ten days cannot support a band or a projection. The tab has to
        # say why rather than render an empty chart.
        app = start(timeline_frame(days=10, spike_day=None))

        self.assertEqual(errors(app), [])

        spoken = " ".join(
            [message.value for message in app.info]
            + [message.value for message in app.error]
        )

        self.assertIn("at least", spoken)


class DashboardTabTest(unittest.TestCase):

    def test_the_tab_is_absent_without_a_timeline(self):
        # With no date there is no movement to report, and the tab would
        # only restate the overview.
        app = start(classification_frame())

        self.assertNotIn("Dashboard", [tab.label for tab in app.tabs])

    def test_the_tab_leads_when_the_data_supports_one(self):
        app = start(timeline_frame())

        self.assertEqual(errors(app), [])
        self.assertEqual(app.tabs[0].label, "Dashboard")

    def test_a_headline_finding_is_shown(self):
        app = start(timeline_frame())

        markdown = " ".join(block.value for block in app.markdown)

        self.assertIn("class='headline'", markdown)

    def test_column_names_cannot_inject_html(self):
        # Column names come from the uploaded file, so they are untrusted.
        # Any block rendered with raw HTML allowed must carry them escaped.
        hostile = "<img src=x onerror=alert(1)>"
        frame = timeline_frame().rename(columns={
            "region": f"region{hostile}",
            "revenue": f"revenue{hostile}",
        })

        app = start(frame)

        self.assertEqual(errors(app), [])

        raw_html_blocks = [
            block.value
            for block in app.markdown
            if block.proto.allow_html
        ]

        self.assertTrue(
            any("&lt;img" in value for value in raw_html_blocks),
            "the hostile column name should appear, escaped"
        )

        for value in raw_html_blocks:
            self.assertNotIn(hostile, value)

    def test_the_detected_roles_are_stated(self):
        app = start(timeline_frame())

        markdown = " ".join(block.value for block in app.markdown)

        self.assertIn("Reading this as", markdown)

    def test_four_headline_figures_are_shown(self):
        app = start(timeline_frame())

        labels = [metric.label for metric in app.metric]

        self.assertIn("Change on previous", labels)
        self.assertTrue(
            any(label.startswith("Average per") for label in labels)
        )

    def test_every_finding_carries_its_calculation(self):
        app = start(timeline_frame())

        captions = [caption.value for caption in app.caption]
        stated = [
            caption for caption in captions
            if caption.startswith("Calculation:")
        ]

        self.assertTrue(stated, "no calculation was shown")

    def test_the_planted_spike_reaches_the_ledger(self):
        app = start(timeline_frame(spike_day=400))

        text = " ".join(block.value for block in app.markdown)

        self.assertIn("Unusual periods", text)

    def test_recommendations_are_marked_as_investigations(self):
        app = start(timeline_frame(spike_day=400))

        captions = " ".join(caption.value for caption in app.caption)

        self.assertIn("not established causes", captions)

    def test_a_waterfall_and_a_heatmap_are_drawn(self):
        app = start(timeline_frame())

        self.assertEqual(errors(app), [])

        keys = [chart.proto.id for chart in app.get("plotly_chart")]

        self.assertTrue(
            any(key.endswith("_bi_waterfall") for key in keys),
            f"no waterfall was drawn: {keys}"
        )
        self.assertTrue(
            any(key.endswith("_bi_heatmap") for key in keys),
            f"no heatmap was drawn: {keys}"
        )

    def test_the_breakdown_can_be_switched_off(self):
        app = start(timeline_frame())

        selectbox(app, "_bi_segment").select("No breakdown").run()

        self.assertEqual(errors(app), [])

        notes = " ".join(message.value for message in app.info)

        self.assertIn("Break down by", notes)

    def test_the_measure_can_be_changed(self):
        app = start(timeline_frame())

        selectbox(app, "_bi_measure").select("units").run()

        self.assertEqual(errors(app), [])

    def test_counting_rows_needs_no_measure_column(self):
        app = start(timeline_frame())

        selectbox(app, "_bi_measure").select("Number of rows").run()

        self.assertEqual(errors(app), [])

    def test_the_period_grain_can_be_overridden(self):
        app = start(timeline_frame())

        selectbox(app, "_bi_grain").select("quarter").run()

        self.assertEqual(errors(app), [])

    def test_a_non_additive_aggregation_withholds_the_shares(self):
        app = start(timeline_frame())

        selectbox(app, "_bi_aggregation").select("mean").run()

        self.assertEqual(errors(app), [])
        self.assertTrue(toggle(app, "_bi_shares").disabled)

    def test_the_share_view_can_be_turned_off(self):
        app = start(timeline_frame())

        toggle(app, "_bi_shares").set_value(False).run()

        self.assertEqual(errors(app), [])

    def test_a_part_can_be_opened_up_by_another_dimension(self):
        frame = timeline_frame()
        frame["channel"] = [
            "web" if index % 2 else "retail"
            for index in range(len(frame))
        ]

        app = start(frame)

        self.assertEqual(errors(app), [])

        keys = [chart.proto.id for chart in app.get("plotly_chart")]

        self.assertTrue(
            any(key.endswith("_bi_inner_heatmap") for key in keys),
            f"no drill-down was drawn: {keys}"
        )

    def test_a_drill_down_says_it_does_not_tie_to_the_total(self):
        frame = timeline_frame()
        frame["channel"] = [
            "web" if index % 2 else "retail"
            for index in range(len(frame))
        ]

        app = start(frame)

        captions = " ".join(caption.value for caption in app.caption)

        self.assertIn("rather than to the headline total", captions)

    def test_the_brief_can_be_downloaded(self):
        app = start(timeline_frame(), name="sales.csv")

        buttons = [
            button for button in app.get("download_button")
            if button.label == "Download the brief"
        ]

        self.assertTrue(buttons, "no download button was offered")

    def test_the_reading_of_the_columns_is_explained(self):
        app = start(timeline_frame())

        labels = [
            expander.label
            for expander in app.get("expander")
        ]

        self.assertIn("How the columns were read", labels)

    def test_a_dataset_with_a_date_but_no_measure_still_works(self):
        frame = timeline_frame(days=200, spike_day=None).drop(
            columns=["revenue", "units"]
        )

        app = start(frame)

        self.assertEqual(errors(app), [])
        self.assertIn("Dashboard", [tab.label for tab in app.tabs])

    def test_a_short_history_explains_itself(self):
        app = start(timeline_frame(days=10, spike_day=None))

        self.assertEqual(errors(app), [])


class DeterministicQuestionTest(unittest.TestCase):
    """
    Asking a question with no API key configured.

    These run the whole path: the question is parsed into a plan, executed
    in pandas and rendered. No model is reachable, which is the point.
    """

    def ask(self, question, frame=None):
        app = start(frame if frame is not None else timeline_frame())
        app.chat_input[0].set_value(question).run()

        return app

    def test_a_total_is_answered_without_a_key(self):
        app = self.ask("what was total revenue")

        self.assertEqual(errors(app), [])

        stored = app.session_state["agent_chat_messages"]

        self.assertEqual(stored[-1]["planned_by"], "rules")
        self.assertIn("revenue", stored[-1]["content"].lower())

    def test_the_figure_matches_the_arithmetic(self):
        frame = timeline_frame(spike_day=None)
        app = self.ask("what was total revenue", frame)

        expected = f"{frame['revenue'].sum():,.2f}"
        answer = app.session_state["agent_chat_messages"][-1]["content"]

        self.assertIn(expected.split(".")[0], answer)

    def test_the_session_usage_counts_the_question(self):
        app = self.ask("what was total revenue")

        usage = app.session_state["session_usage"]
        labels = [expander.label for expander in app.get("expander")]

        self.assertEqual(usage.questions, 1)
        self.assertEqual(usage.by_route, {"rules": 1})
        self.assertEqual(usage.model_calls, 0)
        self.assertTrue(
            any(label.startswith("This session: 1 question") for label in labels),
            labels
        )

        markdown = " ".join(block.value for block in app.markdown)

        self.assertIn('class="usage-list"', markdown)
        self.assertIn("<dt>Answered locally</dt><dd>100%</dd>", markdown)

    def test_an_example_question_can_be_asked_in_one_click(self):
        app = start(timeline_frame())

        app.button(key="example_question_0").click().run()

        stored = app.session_state["agent_chat_messages"]

        self.assertEqual(errors(app), [])
        self.assertEqual(stored[0], {"role": "user", "content": "total revenue"})
        self.assertEqual(stored[-1]["planned_by"], "rules")

    def test_examples_disappear_once_a_conversation_starts(self):
        app = self.ask("what was total revenue")

        keys = [button.key for button in app.button]

        self.assertNotIn("example_question_0", keys)

    def test_a_glossary_upload_is_offered(self):
        app = start(timeline_frame())

        labels = [uploader.label for uploader in app.get("file_uploader")]

        self.assertTrue(
            any(label.startswith("Business glossary") for label in labels),
            labels
        )
        self.assertIsNone(app.session_state["glossary"])

    def test_the_equivalent_sql_is_offered(self):
        app = self.ask("what was total revenue")

        labels = [expander.label for expander in app.get("expander")]
        stored = app.session_state["agent_chat_messages"][-1]

        self.assertIn("Equivalent SQL", labels)
        self.assertIn('SUM(TRY_CAST("revenue" AS DOUBLE))', stored["sql"])

    def test_the_answer_says_how_the_question_was_read(self):
        app = self.ask("what was total revenue")

        captions = " ".join(caption.value for caption in app.caption)

        self.assertIn("Read as:", captions)

    def test_a_locally_read_answer_is_badged_as_such(self):
        app = self.ask("what was total revenue")

        markdown = " ".join(block.value for block in app.markdown)

        self.assertIn("Read locally", markdown)

    def test_the_working_is_available(self):
        app = self.ask("total revenue in north last 3 months")

        labels = [
            expander.label for expander in app.get("expander")
        ]

        self.assertTrue(
            any("Show the calculation" in label for label in labels),
            labels
        )

    def test_a_breakdown_returns_a_table(self):
        app = self.ask("revenue by region")

        self.assertEqual(errors(app), [])

        stored = app.session_state["agent_chat_messages"][-1]

        self.assertIsNotNone(stored["table"])

    def test_words_the_parser_did_not_use_are_reported(self):
        app = self.ask("total revenue for our best customers")

        captions = " ".join(caption.value for caption in app.caption)

        self.assertIn("Not used from the question", captions)

    def test_a_question_with_no_signal_explains_itself(self):
        app = self.ask("tell me something interesting")

        self.assertEqual(errors(app), [])

        stored = app.session_state["agent_chat_messages"][-1]

        self.assertEqual(stored["planned_by"], "nobody")
        self.assertIn("Gemini API key", stored["content"])

    def test_a_trend_question_is_answered(self):
        app = self.ask("is revenue growing")

        self.assertEqual(errors(app), [])

        stored = app.session_state["agent_chat_messages"][-1]

        self.assertEqual(stored["planned_by"], "rules")

    def test_a_second_question_keeps_the_first_answer(self):
        app = self.ask("total revenue")
        app.chat_input[0].set_value("revenue by region").run()

        self.assertEqual(errors(app), [])

        stored = app.session_state["agent_chat_messages"]

        self.assertEqual(len(stored), 4)
        self.assertEqual(stored[0]["role"], "user")
        self.assertEqual(stored[1]["role"], "assistant")

    def test_history_re_renders_without_error(self):
        app = self.ask("revenue by region")
        app.run()

        self.assertEqual(errors(app), [])

    def test_a_dateless_dataset_still_answers(self):
        app = self.ask("total amount by plan", classification_frame())

        self.assertEqual(errors(app), [])

        stored = app.session_state["agent_chat_messages"][-1]

        self.assertEqual(stored["planned_by"], "rules")

    def test_the_missing_key_notice_says_what_still_works(self):
        app = start(timeline_frame())

        notes = " ".join(message.value for message in app.info)

        self.assertIn("still", notes)


class SessionScopingTest(unittest.TestCase):

    def test_two_sessions_with_the_same_file_name_stay_separate(self):
        first = start(classification_frame(), name="data.csv")
        second = start(regression_frame(), name="data.csv")

        self.assertNotEqual(
            first.session_state["dataset_key"],
            second.session_state["dataset_key"]
        )

    def test_a_dataset_handle_is_assigned(self):
        app = start(classification_frame(), name="data.csv")

        self.assertIn("data.csv", app.session_state["dataset_key"])


if __name__ == "__main__":
    unittest.main()
