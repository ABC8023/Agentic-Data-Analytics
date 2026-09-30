"""
The enforced reply schema matches the executor's vocabulary, and a
structured reply goes through the same gate as a text reply.
"""

import os
import unittest
from unittest import mock

import aggregation
import evaluation
import llm
import nlq
import nlq_answer
import nlq_model
import planner_schema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def enum_of(schema, name):
    definition = schema["properties"][name]

    for option in definition.get("anyOf", [definition]):
        if "enum" in option:
            return tuple(option["enum"])

    raise AssertionError(f"{name} has no enum")


class VocabularyTest(unittest.TestCase):

    def setUp(self):
        self.schema = planner_schema.PlannerReply.model_json_schema()

    def test_the_intents_are_the_executors(self):
        self.assertEqual(enum_of(self.schema, "intent"), tuple(nlq.INTENTS))

    def test_the_aggregations_are_the_executors(self):
        self.assertEqual(
            enum_of(self.schema, "aggregation"),
            tuple(aggregation.AGGREGATIONS)
        )

    def test_the_operators_are_the_executors(self):
        filter_schema = self.schema["$defs"]["PlanFilter"]

        self.assertEqual(
            tuple(filter_schema["properties"]["operator"]["enum"]),
            tuple(nlq.FILTER_OPERATORS)
        )

    def test_the_schema_has_no_field_the_plan_refuses(self):
        allowed = {
            "on_topic", "answerable", "intent", "measure", "aggregation",
            "date", "grain", "segment", "segment_value", "filters",
            "window", "comparison", "top_n", "direction",
        }

        self.assertEqual(set(self.schema["properties"]), allowed)


class RoundTripTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.frame = evaluation.load_dataset(
            os.path.join(ROOT, "sample_data", "store_orders.csv")
        )

    def plan(self, reply):
        text = planner_schema.reply_to_text(reply)

        return nlq_model.plan_with_model(
            "a question the rules cannot read", self.frame, lambda _: text
        )

    def test_unset_fields_are_left_out(self):
        text = planner_schema.reply_to_text(
            planner_schema.PlannerReply(on_topic=False)
        )

        self.assertEqual(text, '{"on_topic": false}')

    def test_a_structured_plan_with_a_numeric_filter_gives_the_right_figure(self):
        planned = self.plan(planner_schema.PlannerReply(
            on_topic=True,
            intent="aggregate",
            measure="revenue",
            aggregation="sum",
            filters=[planner_schema.PlanFilter(
                column="units", operator="greater_than", values=["20"],
            )],
        ))

        self.assertTrue(planned.understood, planned.reason)

        answer = nlq_answer.execute_plan(self.frame, planned.plan)

        self.assertAlmostEqual(answer.figures["value"], 888239.09, places=2)

    def test_a_structured_reply_naming_a_missing_column_is_still_refused(self):
        planned = self.plan(planner_schema.PlannerReply(
            on_topic=True, intent="breakdown", measure="revenue",
            segment="product",
        ))

        self.assertFalse(planned.understood)

    def test_a_structured_unanswerable_reply_is_not_executed(self):
        planned = self.plan(
            planner_schema.PlannerReply(on_topic=True, answerable=False)
        )

        self.assertTrue(planned.unanswerable)


class StructuredCompletionTest(unittest.TestCase):

    class Message:

        def __init__(self, content):
            self.content = content
            self.usage_metadata = {"input_tokens": 50, "output_tokens": 5}
            self.response_metadata = {"model_name": "fake-model-001"}

    class Model:

        def __init__(self, response):
            self.response = response
            self.schemas = []

        def with_structured_output(self, schema, include_raw=False):
            self.schemas.append((schema, include_raw))
            response = self.response

            class Runnable:

                def invoke(self, prompt):
                    return response

            return Runnable()

    def test_a_parsed_reply_is_written_back_as_text(self):
        model = self.Model({
            "raw": self.Message(""),
            "parsed": planner_schema.PlannerReply(on_topic=False),
            "parsing_error": None,
        })

        completion = llm.complete_structured(
            model, "p", planner_schema.PlannerReply,
            planner_schema.reply_to_text, "requested",
        )

        self.assertEqual(completion.text, '{"on_topic": false}')
        self.assertEqual(completion.input_tokens, 50)
        self.assertEqual(completion.model, "fake-model-001")
        self.assertEqual(model.schemas, [(planner_schema.PlannerReply, True)])

    def test_an_unparsed_reply_falls_back_to_the_raw_text(self):
        model = self.Model({
            "raw": self.Message('here you go {"on_topic": false}'),
            "parsed": None,
            "parsing_error": ValueError("bad"),
        })

        completion = llm.complete_structured(
            model, "p", planner_schema.PlannerReply,
            planner_schema.reply_to_text, "requested",
        )

        self.assertIn('{"on_topic": false}', completion.text)


class ModeTest(unittest.TestCase):

    def test_text_is_the_default(self):
        with mock.patch.dict(os.environ, {llm.PLANNER_OUTPUT_ENV: ""}):
            self.assertEqual(llm.planner_output_mode(), llm.PLANNER_OUTPUT_TEXT)

    def test_structured_can_be_chosen(self):
        with mock.patch.dict(os.environ, {llm.PLANNER_OUTPUT_ENV: "Structured"}):
            self.assertEqual(
                llm.planner_output_mode(), llm.PLANNER_OUTPUT_STRUCTURED
            )

    def test_an_unknown_mode_is_the_default(self):
        with mock.patch.dict(os.environ, {llm.PLANNER_OUTPUT_ENV: "yaml"}):
            self.assertEqual(llm.planner_output_mode(), llm.PLANNER_OUTPUT_TEXT)


if __name__ == "__main__":
    unittest.main()
