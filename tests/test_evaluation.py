"""
The evaluation harness scores answers correctly, and the golden set holds.

The last class is the regression gate. Offline, every rules case must pass
and no question may be answered with a wrong figure. A change to the
parser that starts answering questions it only half read fails here.
"""

import json
import os
import tempfile
import unittest

import pandas as pd

import evaluation
import llm
import nlq
import nlq_answer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES_PATH = os.path.join(ROOT, "evals", "planner_cases.jsonl")


def result_with(answer, planned_by="rules", plan=None, not_on_topic=False):
    return {
        "answer": answer,
        "planned_by": planned_by,
        "plan": plan,
        "not_on_topic": not_on_topic,
    }


def aggregate_plan(measure="revenue"):
    return nlq.QueryPlan(intent="aggregate", measure=measure)


class LoadCasesTest(unittest.TestCase):

    def write(self, lines):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as handle:
            handle.write("\n".join(lines))

        self.addCleanup(os.unlink, handle.name)

        return handle.name

    def test_comments_and_blank_lines_are_skipped(self):
        path = self.write([
            "// a comment",
            "",
            json.dumps({
                "id": "x", "category": "rules",
                "question": "q", "expect": {},
            }),
        ])

        self.assertEqual(len(evaluation.load_cases(path)), 1)

    def test_an_unknown_category_is_refused(self):
        path = self.write([json.dumps({
            "id": "x", "category": "vibes", "question": "q", "expect": {},
        })])

        with self.assertRaises(ValueError):
            evaluation.load_cases(path)

    def test_a_repeated_id_is_refused(self):
        case = json.dumps({
            "id": "x", "category": "rules", "question": "q", "expect": {},
        })

        with self.assertRaises(ValueError):
            evaluation.load_cases(self.write([case, case]))

    def test_the_golden_set_loads(self):
        cases = evaluation.load_cases(CASES_PATH)

        self.assertGreaterEqual(len(cases), 60)
        self.assertEqual(
            {case["category"] for case in cases},
            set(evaluation.CATEGORIES)
        )


class ScoreCaseTest(unittest.TestCase):

    def test_a_right_figure_passes(self):
        case = {"category": "rules", "expect": {
            "figures": [{"key": "value", "equals": 100.0}],
        }}
        answer = nlq_answer.Answer(
            success=True, figures={"value": 100.004}, plan=aggregate_plan()
        )

        status, reasons, wrong = evaluation.score_case(
            case, result_with(answer, plan=aggregate_plan()), offline=True
        )

        self.assertEqual(status, evaluation.PASS, reasons)
        self.assertFalse(wrong)

    def test_a_wrong_figure_is_confidently_wrong(self):
        case = {"category": "rules", "expect": {
            "figures": [{"key": "value", "equals": 100.0}],
        }}
        answer = nlq_answer.Answer(success=True, figures={"value": 90.0})

        status, _, wrong = evaluation.score_case(
            case, result_with(answer), offline=True
        )

        self.assertEqual(status, evaluation.FAIL)
        self.assertTrue(wrong)

    def test_a_wrong_plan_field_is_confidently_wrong(self):
        case = {"category": "rules", "expect": {"measure": "units"}}
        answer = nlq_answer.Answer(success=True, figures={"value": 1})

        status, reasons, wrong = evaluation.score_case(
            case, result_with(answer, plan=aggregate_plan()), offline=True
        )

        self.assertEqual(status, evaluation.FAIL)
        self.assertTrue(wrong)
        self.assertIn("measure", reasons[0])

    def test_answering_a_question_that_should_be_declined_fails(self):
        case = {"category": "unanswerable", "expect": {"refuse": True}}
        answer = nlq_answer.Answer(success=True, figures={"value": 1})

        status, _, wrong = evaluation.score_case(
            case, result_with(answer), offline=True
        )

        self.assertEqual(status, evaluation.FAIL)
        self.assertTrue(wrong)

    def test_offline_a_model_case_is_skipped_not_failed(self):
        case = {"category": "model", "expect": {"intent": "share"}}
        answer = nlq_answer.Answer(success=False, error="not read")

        status, _, _ = evaluation.score_case(
            case, result_with(answer, planned_by="nobody"), offline=True
        )

        self.assertEqual(status, evaluation.SKIPPED)

    def test_live_an_unflagged_off_topic_question_fails(self):
        case = {"category": "off_topic", "expect": {
            "refuse": True, "off_topic": True,
        }}
        answer = nlq_answer.Answer(success=False, error="no plan")

        status, _, _ = evaluation.score_case(
            case, result_with(answer, planned_by="nobody"), offline=False
        )

        self.assertEqual(status, evaluation.FAIL)

    def test_forbidden_text_fails_even_in_a_refusal(self):
        case = {"category": "adversarial", "expect": {
            "refuse": True, "forbid": ["ORD-0"],
        }}
        answer = nlq_answer.Answer(success=False, error="see ORD-000001")

        status, _, _ = evaluation.score_case(
            case, result_with(answer, planned_by="nobody"), offline=True
        )

        self.assertEqual(status, evaluation.FAIL)

    def test_an_unreachable_model_is_an_error_not_a_failure(self):
        case = {"category": "model", "expect": {"intent": "share"}}
        answer = nlq_answer.Answer(
            success=False, error="The model could not be reached: 429"
        )
        result = result_with(answer, planned_by="nobody")
        result["model_unreachable"] = True

        status, _, wrong = evaluation.score_case(case, result, offline=False)

        self.assertEqual(status, evaluation.ERROR)
        self.assertFalse(wrong)

    def test_text_figures_match_case_insensitively(self):
        self.assertTrue(evaluation.figures_match("North", "north", {}))
        self.assertFalse(evaluation.figures_match("north", "south", {}))
        self.assertFalse(evaluation.figures_match(1.0, None, {}))


class RecordingInvokerTest(unittest.TestCase):

    class FakeMessage:

        def __init__(self, text):
            self.content = text
            self.usage_metadata = {"input_tokens": 10, "output_tokens": 2}
            self.response_metadata = {"model_name": "fake-model-001"}

    class FakeModel:

        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1

            return RecordingInvokerTest.FakeMessage('{"on_topic": false}')

    def setUp(self):
        self.config = llm.LLMConfig(
            provider="google_genai", model="fake-model",
            timeout=1.0, max_retries=0,
        )

    def test_each_call_is_recorded(self):
        completions = []
        invoke = evaluation.recording_invoker(
            self.FakeModel(), self.config, completions
        )

        invoke("prompt")

        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0].model, "fake-model-001")
        self.assertEqual(completions[0].input_tokens, 10)

    def test_a_cached_reply_is_replayed_without_a_call(self):
        model = self.FakeModel()
        completions = []
        cache = {}
        invoke = evaluation.recording_invoker(
            model, self.config, completions, cache
        )

        first = invoke("prompt")
        second = invoke("prompt")

        self.assertEqual(first, second)
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(completions), 1)

    def test_live_calls_are_spaced_out(self):
        slept = []
        invoke = evaluation.recording_invoker(
            self.FakeModel(), self.config, [],
            min_interval_seconds=10.0, sleep=slept.append,
        )

        invoke("first")
        invoke("second")

        self.assertEqual(len(slept), 1)
        self.assertGreater(slept[0], 9.0)


class SummariseTest(unittest.TestCase):

    def result(self, **overrides):
        values = dict(
            id="x", category="rules", question="q", status=evaluation.PASS,
            route="rules", expected_route="rules", answered=True,
            confident_wrong=False, not_on_topic=False,
        )
        values.update(overrides)

        return evaluation.CaseResult(**values)

    def test_errors_do_not_count_against_the_pass_rate(self):
        summary = evaluation.summarise([
            self.result(),
            self.result(id="y", status=evaluation.ERROR, answered=False),
        ], offline=False)

        self.assertEqual(summary["pass_rate"], 1.0)
        self.assertEqual(summary["errors"], 1)

    def test_skipped_cases_do_not_count_against_the_pass_rate(self):
        summary = evaluation.summarise([
            self.result(),
            self.result(
                id="y", category="model", status=evaluation.SKIPPED,
                route="nobody", expected_route="model", answered=False,
            ),
        ], offline=True)

        self.assertEqual(summary["pass_rate"], 1.0)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["route_accuracy"], 1.0)

    def test_cost_needs_both_prices(self):
        results = [self.result(input_tokens=1_000_000, output_tokens=500_000)]

        self.assertIsNone(
            evaluation.summarise(results, offline=False, input_price_per_million=1.0)[
                "estimated_cost"
            ]
        )
        self.assertEqual(
            evaluation.summarise(
                results, offline=False,
                input_price_per_million=1.0, output_price_per_million=4.0,
            )["estimated_cost"],
            3.0,
        )


class GoldenSetOfflineGateTest(unittest.TestCase):
    """The deterministic path, measured against known answers."""

    @classmethod
    def setUpClass(cls):
        cases = evaluation.load_cases(CASES_PATH)
        cls.results = evaluation.run(cases, ROOT, invoke=None)
        cls.summary = evaluation.summarise(cls.results, offline=True)

    def describe(self, results):
        return "\n".join(
            f"{r.id} {r.question!r}: {'; '.join(r.reasons)}" for r in results
        )

    def test_no_question_is_answered_with_a_wrong_figure(self):
        wrong = [r for r in self.results if r.confident_wrong]

        self.assertEqual(wrong, [], self.describe(wrong))

    def test_every_rules_case_passes(self):
        failed = [
            r for r in self.results
            if r.category == "rules" and r.status != evaluation.PASS
        ]

        self.assertEqual(failed, [], self.describe(failed))

    def test_nothing_offline_fails(self):
        failed = [r for r in self.results if r.status == evaluation.FAIL]

        self.assertEqual(failed, [], self.describe(failed))

    def test_no_model_is_called_offline(self):
        self.assertEqual(self.summary["model_calls"], 0)


class DatasetTest(unittest.TestCase):

    def test_the_sample_dataset_matches_the_facts_file(self):
        frame = evaluation.load_dataset(
            os.path.join(ROOT, "sample_data", "store_orders.csv")
        )
        revenue = pd.to_numeric(frame["revenue"], errors="coerce")

        self.assertEqual(len(frame), 3660)
        self.assertAlmostEqual(revenue.sum(), 1513129.01, places=2)


if __name__ == "__main__":
    unittest.main()
