"""
Reconstructing the calculation behind an assistant answer.

Real message objects from langchain_core are used, so the extractor is
tested against the shape LangGraph actually produces rather than a guess
at it. No API key and no model call are involved.
"""

import unittest
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import audit


def tool_call(name, arguments, call_id):
    return {
        "name": name,
        "args": arguments,
        "id": call_id,
        "type": "tool_call",
    }


def conversation(*messages):
    return {"messages": list(messages)}


class ExtractionTest(unittest.TestCase):

    def test_a_single_call_is_reconstructed(self):
        response = conversation(
            HumanMessage(content="how many rows?"),
            AIMessage(
                content="",
                tool_calls=[tool_call(
                    "get_data_quality_report",
                    {"name": "sales.csv"},
                    "call_1"
                )]
            ),
            ToolMessage(
                content=str({
                    "success": True,
                    "rows": 4339,
                    "columns": 12,
                }),
                tool_call_id="call_1",
                name="get_data_quality_report"
            ),
            AIMessage(content="It has 4,339 rows."),
        )

        trail = audit.extract_calculation_trail(response)

        self.assertEqual(len(trail), 1)
        self.assertEqual(trail[0]["tool"], "get_data_quality_report")
        self.assertEqual(trail[0]["arguments"], {"name": "sales.csv"})
        self.assertEqual(trail[0]["result"]["rows"], 4339)
        self.assertFalse(trail[0]["failed"])

    def test_the_reported_figures_appear_in_the_highlights(self):
        response = conversation(
            AIMessage(
                content="",
                tool_calls=[tool_call("t", {}, "c1")]
            ),
            ToolMessage(
                content=str({"success": True, "rows": 4339}),
                tool_call_id="c1",
                name="t"
            ),
        )

        trail = audit.extract_calculation_trail(response)

        self.assertIn("rows: 4,339", trail[0]["highlights"])

    def test_the_session_token_is_hidden_from_the_arguments(self):
        token = uuid.uuid4().hex
        handle = f"{token}/quarterly.csv"

        response = conversation(
            AIMessage(
                content="",
                tool_calls=[tool_call("t", {"name": handle}, "c1")]
            ),
            ToolMessage(content="{}", tool_call_id="c1", name="t"),
        )

        trail = audit.extract_calculation_trail(response)

        self.assertEqual(
            trail[0]["arguments"]["name"],
            "quarterly.csv"
        )

    def test_the_session_token_is_absent_from_the_whole_entry(self):
        # The token also arrives inside the result, under keys such as
        # file_name, and must not be retained in the conversation.
        token = uuid.uuid4().hex
        handle = f"{token}/quarterly.csv"

        response = conversation(
            AIMessage(
                content="",
                tool_calls=[tool_call("t", {"name": handle}, "c1")]
            ),
            ToolMessage(
                content=str({
                    "success": True,
                    "file_name": handle,
                    "rows": 5,
                }),
                tool_call_id="c1",
                name="t"
            ),
        )

        trail = audit.extract_calculation_trail(response)

        self.assertNotIn(token, str(trail))
        self.assertEqual(trail[0]["result"]["file_name"], "quarterly.csv")

    def test_a_real_path_is_left_alone(self):
        response = conversation(
            AIMessage(
                content="",
                tool_calls=[tool_call(
                    "t",
                    {"name": "sample_data/iris.csv"},
                    "c1"
                )]
            ),
            ToolMessage(content="{}", tool_call_id="c1", name="t"),
        )

        trail = audit.extract_calculation_trail(response)

        self.assertEqual(
            trail[0]["arguments"]["name"],
            "sample_data/iris.csv"
        )

    def test_calls_are_numbered_in_order(self):
        response = conversation(
            AIMessage(
                content="",
                tool_calls=[tool_call("first", {}, "c1")]
            ),
            ToolMessage(content="{}", tool_call_id="c1", name="first"),
            AIMessage(
                content="",
                tool_calls=[tool_call("second", {}, "c2")]
            ),
            ToolMessage(content="{}", tool_call_id="c2", name="second"),
        )

        trail = audit.extract_calculation_trail(response)

        self.assertEqual(
            [(entry["step"], entry["tool"]) for entry in trail],
            [(1, "first"), (2, "second")]
        )

    def test_parallel_calls_are_matched_by_identifier(self):
        # A model may request several tools before any result arrives, so
        # results must not be paired by position.
        response = conversation(
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("alpha", {}, "c1"),
                    tool_call("beta", {}, "c2"),
                ]
            ),
            ToolMessage(
                content=str({"value": 2}),
                tool_call_id="c2",
                name="beta"
            ),
            ToolMessage(
                content=str({"value": 1}),
                tool_call_id="c1",
                name="alpha"
            ),
        )

        trail = audit.extract_calculation_trail(response)

        by_tool = {entry["tool"]: entry["result"] for entry in trail}

        self.assertEqual(by_tool["alpha"]["value"], 1)
        self.assertEqual(by_tool["beta"]["value"], 2)

    def test_a_failed_call_is_flagged_with_its_reason(self):
        response = conversation(
            AIMessage(
                content="",
                tool_calls=[tool_call("t", {"name": "x.csv"}, "c1")]
            ),
            ToolMessage(
                content=str({
                    "success": False,
                    "error": "file x.csv not found",
                }),
                tool_call_id="c1",
                name="t"
            ),
        )

        trail = audit.extract_calculation_trail(response)

        self.assertTrue(trail[0]["failed"])
        self.assertEqual(trail[0]["error"], "file x.csv not found")

    def test_a_call_without_a_result_is_still_recorded(self):
        response = conversation(
            AIMessage(
                content="",
                tool_calls=[tool_call("t", {}, "c1")]
            ),
        )

        trail = audit.extract_calculation_trail(response)

        self.assertEqual(len(trail), 1)
        self.assertIsNone(trail[0]["result"])
        self.assertEqual(trail[0]["highlights"], [])

    def test_a_result_without_an_identifier_is_matched_by_name(self):
        response = conversation(
            AIMessage(
                content="",
                tool_calls=[tool_call("named", {}, "c1")]
            ),
            ToolMessage(
                content=str({"rows": 7}),
                tool_call_id="",
                name="named"
            ),
        )

        trail = audit.extract_calculation_trail(response)

        self.assertEqual(trail[0]["result"]["rows"], 7)

    def test_an_answer_with_no_tool_calls_yields_an_empty_trail(self):
        response = conversation(
            HumanMessage(content="hello"),
            AIMessage(content="Hello. Upload a dataset to begin."),
        )

        trail = audit.extract_calculation_trail(response)

        self.assertEqual(trail, [])
        self.assertTrue(audit.trail_is_empty(trail))

    def test_a_bare_message_list_is_accepted(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[tool_call("t", {}, "c1")]
            ),
            ToolMessage(content="{}", tool_call_id="c1", name="t"),
        ]

        self.assertEqual(
            len(audit.extract_calculation_trail(messages)),
            1
        )

    def test_missing_or_empty_input_is_tolerated(self):
        for value in [{}, {"messages": None}, [], None]:
            with self.subTest(value=value):
                self.assertEqual(
                    audit.extract_calculation_trail(value),
                    []
                )

    def test_dictionary_shaped_messages_are_understood(self):
        # Defensive: a future version may hand back plain dictionaries.
        response = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [tool_call("t", {"a": 1}, "c1")],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "name": "t",
                    "content": str({"rows": 3}),
                },
            ]
        }

        trail = audit.extract_calculation_trail(response)

        self.assertEqual(len(trail), 1)
        self.assertEqual(trail[0]["result"]["rows"], 3)


class ResultParsingTest(unittest.TestCase):

    def test_a_python_repr_string_is_recovered(self):
        parsed = audit.parse_result("{'rows': 10, 'ok': True}")

        self.assertEqual(parsed, {"rows": 10, "ok": True})

    def test_an_already_parsed_value_passes_through(self):
        self.assertEqual(
            audit.parse_result({"rows": 1}),
            {"rows": 1}
        )

    def test_plain_prose_is_left_as_text(self):
        self.assertEqual(
            audit.parse_result("no datasets were found"),
            "no datasets were found"
        )

    def test_malformed_content_does_not_raise(self):
        self.assertEqual(
            audit.parse_result("{'rows': "),
            "{'rows': "
        )

    def test_a_list_result_is_recovered(self):
        self.assertEqual(
            audit.parse_result("[{'a': 1}]"),
            [{"a": 1}]
        )


class FormattingTest(unittest.TestCase):

    def test_large_integers_get_thousands_separators(self):
        self.assertEqual(audit.format_value(4339), "4,339")

    def test_booleans_read_as_words(self):
        self.assertEqual(audit.format_value(True), "yes")
        self.assertEqual(audit.format_value(False), "no")

    def test_whole_floats_lose_their_decimal_point(self):
        self.assertEqual(audit.format_value(20.0), "20")

    def test_fractional_floats_are_rounded(self):
        self.assertEqual(audit.format_value(0.95238095), "0.9524")

    def test_none_is_explicit(self):
        self.assertEqual(audit.format_value(None), "none")

    def test_a_short_list_is_named_in_full(self):
        self.assertEqual(
            audit.format_value(["a", "b"]),
            "2 (a, b)"
        )

    def test_a_long_list_is_counted_and_truncated(self):
        rendered = audit.format_value(
            ["a", "b", "c", "d", "e", "f"]
        )

        self.assertTrue(rendered.startswith("6 ("))
        self.assertIn("...", rendered)

    def test_an_empty_collection_reads_as_none(self):
        self.assertEqual(audit.format_value([]), "none")

    def test_long_text_is_truncated(self):
        rendered = audit.format_value("x" * 400)

        self.assertTrue(rendered.endswith("..."))
        self.assertLess(len(rendered), 400)


class SummaryTest(unittest.TestCase):

    def test_plumbing_keys_are_omitted(self):
        highlights = audit.summarise_result({
            "success": True,
            "privacy_note": "withheld",
            "file_name": "x.csv",
            "rows": 5,
        })

        self.assertEqual(highlights, ["rows: 5"])

    def test_an_error_is_shown_first(self):
        highlights = audit.summarise_result({
            "columns": 3,
            "error": "not found",
        })

        self.assertTrue(highlights[0].startswith("error:"))

    def test_empty_collections_are_omitted(self):
        highlights = audit.summarise_result({
            "constant_columns_removed": [],
            "rows": 5,
        })

        self.assertEqual(highlights, ["rows: 5"])

    def test_the_number_of_highlights_is_capped(self):
        payload = {f"key_{index}": index for index in range(40)}

        self.assertEqual(
            len(audit.summarise_result(payload)),
            audit.MAX_HIGHLIGHTS
        )

    def test_a_non_dictionary_result_is_rendered_directly(self):
        self.assertEqual(
            audit.summarise_result(["a.csv", "b.csv"]),
            ["2 (a.csv, b.csv)"]
        )


class DescriptionTest(unittest.TestCase):

    def test_no_calculation_is_stated_plainly(self):
        description = audit.describe_trail([])

        self.assertIn("No calculation", description)

    def test_one_call_is_singular(self):
        description = audit.describe_trail([
            {"failed": False}
        ])

        self.assertIn("1 calculation)", description)

    def test_several_calls_are_counted(self):
        description = audit.describe_trail([
            {"failed": False},
            {"failed": False},
            {"failed": False},
        ])

        self.assertIn("3 calculations)", description)

    def test_failures_are_surfaced_in_the_label(self):
        description = audit.describe_trail([
            {"failed": False},
            {"failed": True},
        ])

        self.assertIn("1 failed", description)


if __name__ == "__main__":
    unittest.main()
