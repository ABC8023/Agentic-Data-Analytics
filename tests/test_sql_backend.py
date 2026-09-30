"""
A compiled plan gives the same figures as the pandas executor, and nothing
a reader or a model typed can become part of the SQL text.
"""

import math
import os
import unittest

import numpy as np
import pandas as pd

import evaluation
import nlq
import nlq_answer
import sql_backend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES_PATH = os.path.join(ROOT, "evals", "planner_cases.jsonl")


def store_orders():
    return evaluation.load_dataset(
        os.path.join(ROOT, "sample_data", "store_orders.csv")
    )


def pandas_breakdown(frame, plan):
    table = nlq_answer.execute_plan(frame, plan).table

    return dict(zip(table.iloc[:, 0].astype(str), table.iloc[:, 1], strict=True))


def sql_breakdown(frame, plan):
    table = sql_backend.run_plan(plan, frame)

    return dict(zip(table["segment"].astype(str), table["total"], strict=True))


class ParityMixin:

    def assert_same_figures(self, frame, plan):
        if plan.intent == "aggregate":
            expected = nlq_answer.execute_plan(frame, plan).figures["value"]
            actual = sql_backend.run_plan(plan, frame)

            if expected is None:
                self.assertIsNone(actual, plan.describe())

            else:
                self.assertTrue(
                    math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6),
                    f"{plan.describe()}: sql {actual} pandas {expected}"
                )

            return

        expected = pandas_breakdown(frame, plan)
        actual = sql_breakdown(frame, plan)

        self.assertEqual(set(actual), set(expected), plan.describe())

        for part, value in expected.items():
            self.assertTrue(
                math.isclose(actual[part], value, rel_tol=1e-9, abs_tol=1e-6),
                f"{plan.describe()} [{part}]: sql {actual[part]} pandas {value}"
            )


class GoldenSetParityTest(ParityMixin, unittest.TestCase):
    """Every supported plan the rules produce on the evaluation set."""

    @classmethod
    def setUpClass(cls):
        cls.frame = store_orders()
        cls.plans = []

        for case in evaluation.load_cases(CASES_PATH):
            result = nlq_answer.answer_question(cls.frame, case["question"])
            plan = result["plan"]

            if plan is None:
                continue

            try:
                sql_backend.compile_plan(plan, cls.frame)

            except sql_backend.UnsupportedPlan:
                continue

            cls.plans.append((case["id"], plan))

    def test_enough_plans_are_covered_to_mean_something(self):
        self.assertGreaterEqual(len(self.plans), 15)

    def test_every_supported_plan_matches_pandas(self):
        for case_id, plan in self.plans:
            with self.subTest(case=case_id):
                self.assert_same_figures(self.frame, plan)


class OperatorParityTest(ParityMixin, unittest.TestCase):
    """Each filter operator and window kind, on data with awkward cells."""

    @classmethod
    def setUpClass(cls):
        cls.frame = store_orders()

    def plan(self, **fields):
        values = {"intent": "aggregate", "measure": "revenue", "date": "order_date"}
        values.update(fields)

        return nlq.QueryPlan(**values)

    def test_each_filter_operator(self):
        filters = [
            nlq.Filter("region", "equals", ("North",)),
            nlq.Filter("region", "not_equals", ("north",)),
            nlq.Filter("region", "in", ("north", "east")),
            nlq.Filter("region", "not_in", ("north", "east")),
            nlq.Filter("channel", "contains", ("sale",)),
            nlq.Filter("units", "greater_than", (20,)),
            nlq.Filter("units", "greater_or_equal", ("20",)),
            nlq.Filter("units", "less_than", (5,)),
            nlq.Filter("units", "less_or_equal", (5,)),
            nlq.Filter("units", "between", (5, 10)),
            nlq.Filter("region", "is_missing", ()),
            nlq.Filter("region", "is_present", ()),
            nlq.Filter("revenue", "greater_than", (1000,)),
        ]

        for entry in filters:
            with self.subTest(filter=entry.describe()):
                self.assert_same_figures(self.frame, self.plan(filters=(entry,)))

    def test_each_aggregation(self):
        for name in ("sum", "mean", "median", "min", "max", "count"):
            with self.subTest(aggregation=name):
                self.assert_same_figures(self.frame, self.plan(aggregation=name))

    def test_each_window_kind(self):
        windows = [
            nlq.Window("last_periods", count=1, grain="month"),
            nlq.Window("last_periods", count=3, grain="month"),
            nlq.Window("last_periods", count=1, grain="month", offset=1),
            nlq.Window("last_periods", count=2, grain="quarter"),
            nlq.Window("between_dates", start="2024-01-01", end="2024-06-30"),
            nlq.Window("between_dates", start="2025-01-01"),
            nlq.Window("period", start="2024-03-01", grain="month"),
        ]

        for window in windows:
            with self.subTest(window=window.describe()):
                self.assert_same_figures(self.frame, self.plan(window=window))

    def test_breakdowns_in_both_directions_with_a_limit(self):
        for direction in ("highest", "lowest"):
            for top_n in (None, 2):
                with self.subTest(direction=direction, top_n=top_n):
                    self.assert_same_figures(self.frame, self.plan(
                        intent="breakdown",
                        segment="channel",
                        direction=direction,
                        top_n=top_n,
                    ))

    def test_a_breakdown_counting_rows_keeps_the_blank_part(self):
        plan = self.plan(
            intent="breakdown", measure=None, aggregation="count",
            segment="region",
        )

        self.assert_same_figures(self.frame, plan)
        self.assertIn("(not set)", sql_breakdown(self.frame, plan))


class InjectionTest(unittest.TestCase):

    def setUp(self):
        self.frame = pd.DataFrame({
            'region"; DROP TABLE dataset; --': ["north", "south", "north"],
            "revenue": [10.0, 20.0, np.nan],
        })
        self.hostile = 'region"; DROP TABLE dataset; --'

    def test_a_hostile_column_name_is_quoted_as_one_identifier(self):
        plan = nlq.QueryPlan(
            intent="breakdown", measure="revenue", segment=self.hostile
        )

        compiled = sql_backend.compile_plan(plan, self.frame)

        self.assertIn('"region""; DROP TABLE dataset; --"', compiled.sql)
        self.assertEqual(
            sql_breakdown(self.frame, plan), {"south": 20.0, "north": 10.0}
        )

    def test_a_hostile_filter_value_is_a_parameter(self):
        value = "north' OR '1'='1"
        plan = nlq.QueryPlan(
            intent="aggregate", measure="revenue",
            filters=(nlq.Filter(self.hostile, "equals", (value,)),),
        )

        compiled = sql_backend.compile_plan(plan, self.frame)

        self.assertNotIn("OR '1'='1", compiled.sql)
        self.assertIn(value.lower(), compiled.parameters)
        self.assertIsNone(sql_backend.run_plan(plan, self.frame))

    def test_a_column_the_table_lacks_is_refused(self):
        plan = nlq.QueryPlan(intent="aggregate", measure="profit")

        with self.assertRaises(sql_backend.UnsupportedPlan):
            sql_backend.compile_plan(plan, self.frame)


class ScopeTest(unittest.TestCase):

    def setUp(self):
        self.frame = store_orders()

    def test_intents_without_one_query_are_refused(self):
        for intent in ("timeseries", "trend", "compare", "extreme", "share"):
            with self.subTest(intent=intent):
                plan = nlq.QueryPlan(intent=intent, measure="revenue")

                with self.assertRaises(sql_backend.UnsupportedPlan):
                    sql_backend.compile_plan(plan, self.frame)

    def test_a_daily_rate_is_refused(self):
        plan = nlq.QueryPlan(
            intent="aggregate", measure="revenue", aggregation="sum_per_day"
        )

        with self.assertRaises(sql_backend.UnsupportedPlan):
            sql_backend.compile_plan(plan, self.frame)

    def test_the_display_lists_the_parameters(self):
        plan = nlq.QueryPlan(
            intent="aggregate", measure="revenue",
            filters=(nlq.Filter("region", "equals", ("north",)),),
        )

        shown = sql_backend.compile_plan(plan, self.frame).display()

        self.assertIn("-- parameters: 'north'", shown)


if __name__ == "__main__":
    unittest.main()
