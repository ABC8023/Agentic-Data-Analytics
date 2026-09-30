"""
One event per question, with what it cost and why it ended as it did, and
nothing from the data itself unless asked for.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import llm
import nlq
import nlq_answer
import telemetry


def completion(input_tokens=100, output_tokens=20, model="fake-model-001"):
    return llm.Completion(
        text="{}",
        latency_ms=5.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )


def result(success=True, planned_by="rules", **flags):
    values = {
        "answer": nlq_answer.Answer(success=success, error="" if success else "no"),
        "planned_by": planned_by,
        "plan": nlq.QueryPlan(intent="aggregate", measure="revenue") if success else None,
        "ignored": [],
        "model_replies": [],
        "not_on_topic": False,
        "model_unanswerable": False,
        "model_unreachable": False,
    }
    values.update(flags)

    return values


class OutcomeTest(unittest.TestCase):

    def test_each_ending_has_its_own_name(self):
        self.assertEqual(telemetry.outcome_of(result()), "answered")
        self.assertEqual(
            telemetry.outcome_of(result(False, "nobody", model_unreachable=True)),
            "model_unreachable",
        )
        self.assertEqual(
            telemetry.outcome_of(result(False, "nobody", not_on_topic=True)),
            "off_topic",
        )
        self.assertEqual(
            telemetry.outcome_of(result(False, "nobody", model_unanswerable=True)),
            "unanswerable",
        )
        self.assertEqual(
            telemetry.outcome_of(result(False, "nobody", model_replies=["x"])),
            "model_plan_rejected",
        )
        self.assertEqual(
            telemetry.outcome_of(result(False, "nobody")),
            "not_understood",
        )


class QuestionEventTest(unittest.TestCase):

    def test_model_cost_is_summed_over_the_calls(self):
        event = telemetry.question_event(
            result(planned_by="model"),
            1234.56,
            [completion(100, 20), completion(150, 30, model="fake-model-002")],
            "abc",
        )

        self.assertEqual(event["model_calls"], 2)
        self.assertEqual(event["input_tokens"], 250)
        self.assertEqual(event["output_tokens"], 50)
        self.assertEqual(event["model"], "fake-model-002")
        self.assertEqual(event["latency_ms"], 1234.6)
        self.assertEqual(event["route"], "model")
        self.assertEqual(event["intent"], "aggregate")

    def test_the_question_text_is_left_out_by_default(self):
        with mock.patch.dict(os.environ, {telemetry.QUESTION_TEXT_ENV: ""}):
            event = telemetry.question_event(
                result(), 1.0, [], "abc", question="revenue for jane@example.com"
            )

        self.assertNotIn("question", event)
        self.assertNotIn("jane", json.dumps(event))

    def test_the_question_text_can_be_switched_on(self):
        with mock.patch.dict(os.environ, {telemetry.QUESTION_TEXT_ENV: "1"}):
            event = telemetry.question_event(
                result(), 1.0, [], "abc", question="total revenue"
            )

        self.assertEqual(event["question"], "total revenue")

    def test_every_value_is_serialisable(self):
        event = telemetry.question_event(result(), 1.0, [completion()], "abc")

        json.dumps(event)


class AgentEventTest(unittest.TestCase):

    class Message:

        def __init__(self, usage=None):
            self.usage_metadata = usage

    def test_usage_is_summed_over_the_model_turns_only(self):
        messages = [
            self.Message(),
            self.Message({"input_tokens": 400, "output_tokens": 30}),
            self.Message(),
            self.Message({"input_tokens": 600, "output_tokens": 80}),
        ]

        event = telemetry.agent_event("abc", 10.0, messages, 1, False)

        self.assertEqual(event["model_calls"], 2)
        self.assertEqual(event["input_tokens"], 1000)
        self.assertEqual(event["output_tokens"], 110)
        self.assertEqual(event["tool_calls"], 1)


class ConsoleLoggingTest(unittest.TestCase):

    def test_configuring_twice_adds_one_handler(self):
        before = list(telemetry.logger.handlers)
        self.addCleanup(setattr, telemetry.logger, "handlers", before)

        telemetry.configure_console_logging()
        telemetry.configure_console_logging()

        ours = [
            h for h in telemetry.logger.handlers
            if getattr(h, "_ai_analyst", False)
        ]

        self.assertEqual(len(ours), 1)


class EmitTest(unittest.TestCase):

    def test_events_are_appended_as_json_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "events.jsonl")

            with mock.patch.dict(os.environ, {telemetry.EVENT_LOG_ENV: path}):
                telemetry.emit({"event": "question", "request_id": "a"})
                telemetry.emit({"event": "question", "request_id": "b"})

            with open(path, encoding="utf-8") as handle:
                lines = [json.loads(line) for line in handle]

        self.assertEqual([line["request_id"] for line in lines], ["a", "b"])

    def test_an_unwritable_log_does_not_raise(self):
        with mock.patch.dict(
            os.environ,
            {telemetry.EVENT_LOG_ENV: os.path.join("no", "such", "dir", "x.jsonl")},
        ):
            telemetry.emit({"event": "question"})


class SessionUsageTest(unittest.TestCase):

    def test_the_local_share_counts_rules_answers(self):
        usage = telemetry.SessionUsage()

        for route in ("rules", "rules", "rules", "model"):
            usage.add_question({
                "route": route,
                "model_calls": 1 if route == "model" else 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 900.0,
            })

        self.assertEqual(usage.answered_locally_share, 0.75)
        self.assertEqual(usage.model_latency_ms, [900.0])

    def test_agent_calls_add_to_the_cost(self):
        usage = telemetry.SessionUsage()

        usage.add_agent({
            "model_calls": 3,
            "input_tokens": 1000,
            "output_tokens": 200,
            "latency_ms": 4000.0,
        })

        self.assertEqual(usage.handed_to_agent, 1)
        self.assertEqual(usage.model_calls, 3)
        self.assertEqual(usage.median_model_latency_ms, 4000.0)

    def test_an_empty_session_has_no_share(self):
        self.assertIsNone(telemetry.SessionUsage().answered_locally_share)


if __name__ == "__main__":
    unittest.main()
