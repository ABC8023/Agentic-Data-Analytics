"""
The query plan, and the boundary model output has to cross.

Two properties matter here.

A plan is refused rather than repaired. Silently dropping a field a model
thought it was setting produces a confident answer to a question nobody
asked, which is worse than an error message.

A well formed plan is not an answerable one. The vocabulary check and the
dataset check are separate, and the second one is where "total of the
surname column" gets stopped.
"""

import unittest

import numpy as np
import pandas as pd

import aggregation
import nlq
import segments


def sales_frame(days: int = 400) -> pd.DataFrame:
    """A dated table with a measure, a segment and an identifier."""

    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    names = ["north", "south", "east"]

    return pd.DataFrame({
        "order_date": dates,
        "revenue": np.linspace(100.0, 200.0, days).round(2),
        "units": np.arange(1, days + 1),
        "region": [names[index % 3] for index in range(days)],
        "surname": [f"PERSON-{index:05d}" for index in range(days)],
        "note": ["shipped"] * days,
    })


class FilterTest(unittest.TestCase):

    def test_a_filter_round_trips(self):
        entry = nlq.Filter("region", "equals", ("north",))

        parsed, errors = nlq.filter_from_dict(entry.to_dict(), 0)

        self.assertEqual(errors, [])
        self.assertEqual(parsed, entry)

    def test_an_unknown_operator_is_refused(self):
        parsed, errors = nlq.filter_from_dict(
            {"column": "region", "operator": "sounds_like",
             "values": ["north"]},
            0
        )

        self.assertIsNone(parsed)
        self.assertIn("unknown operator", errors[0])

    def test_an_unexpected_field_is_refused(self):
        parsed, errors = nlq.filter_from_dict(
            {"column": "region", "operator": "equals", "values": ["n"],
             "case_sensitive": True},
            0
        )

        self.assertIsNone(parsed)
        self.assertIn("case_sensitive", errors[0])

    def test_between_needs_exactly_two_values(self):
        parsed, errors = nlq.filter_from_dict(
            {"column": "revenue", "operator": "between", "values": [10]},
            0
        )

        self.assertIsNone(parsed)
        self.assertIn("exactly 2", errors[0])

    def test_is_missing_takes_no_value(self):
        parsed, errors = nlq.filter_from_dict(
            {"column": "region", "operator": "is_missing",
             "values": ["north"]},
            0
        )

        self.assertIsNone(parsed)
        self.assertIn("takes no comparison value", errors[0])

    def test_in_needs_at_least_one_value(self):
        parsed, errors = nlq.filter_from_dict(
            {"column": "region", "operator": "in", "values": []},
            0
        )

        self.assertIsNone(parsed)
        self.assertIn("at least one", errors[0])

    def test_a_bare_value_is_accepted_as_a_single_value(self):
        parsed, errors = nlq.filter_from_dict(
            {"column": "region", "operator": "equals", "values": "north"},
            0
        )

        self.assertEqual(errors, [])
        self.assertEqual(parsed.values, ("north",))

    def test_a_missing_column_is_refused(self):
        parsed, errors = nlq.filter_from_dict(
            {"operator": "is_present"},
            0
        )

        self.assertIsNone(parsed)
        self.assertIn("no column name", errors[0])

    def test_the_error_names_which_filter_failed(self):
        parsed, errors = nlq.filter_from_dict({"operator": "nope"}, 2)

        self.assertIn("Filter 3", errors[0])

    def test_every_operator_can_be_described(self):
        for operator, arity in nlq.OPERATOR_ARITY.items():
            values = tuple(range(arity or 1))

            described = nlq.Filter("column", operator, values).describe()

            self.assertTrue(described)
            self.assertIn("column", described)


class WindowTest(unittest.TestCase):

    def test_no_window_means_everything(self):
        window, errors = nlq.window_from_dict(None)

        self.assertEqual(errors, [])
        self.assertEqual(window.kind, "all")

    def test_a_window_round_trips(self):
        window = nlq.Window(kind="last_periods", count=3, grain="month")

        parsed, errors = nlq.window_from_dict(window.to_dict())

        self.assertEqual(errors, [])
        self.assertEqual(parsed, window)

    def test_an_unknown_kind_is_refused(self):
        parsed, errors = nlq.window_from_dict({"kind": "recently"})

        self.assertIsNone(parsed)
        self.assertIn("unknown kind", errors[0])

    def test_last_periods_needs_a_count(self):
        parsed, errors = nlq.window_from_dict({
            "kind": "last_periods",
            "grain": "month",
        })

        self.assertIsNone(parsed)
        self.assertIn("how many", errors[0])

    def test_last_periods_needs_a_grain(self):
        parsed, errors = nlq.window_from_dict({
            "kind": "last_periods",
            "count": 3,
        })

        self.assertIsNone(parsed)
        self.assertIn("which period", errors[0])

    def test_a_date_range_needs_at_least_one_end(self):
        parsed, errors = nlq.window_from_dict({"kind": "between_dates"})

        self.assertIsNone(parsed)
        self.assertIn("start date, an end date or both", errors[0])

    def test_an_open_ended_range_is_allowed(self):
        # "Since March" is a real question, and the missing end is the end
        # of the data.
        parsed, errors = nlq.window_from_dict({
            "kind": "between_dates",
            "start": "2024-01-01",
        })

        self.assertEqual(errors, [])
        self.assertIn("onwards", parsed.describe())

    def test_an_offset_shifts_a_relative_window(self):
        parsed, errors = nlq.window_from_dict({
            "kind": "last_periods",
            "count": 1,
            "grain": "month",
            "offset": 1,
        })

        self.assertEqual(errors, [])
        self.assertEqual(parsed.offset, 1)
        self.assertIn("before the most recent", parsed.describe())

    def test_a_negative_offset_is_refused(self):
        parsed, errors = nlq.window_from_dict({
            "kind": "last_periods",
            "count": 1,
            "grain": "month",
            "offset": -1,
        })

        self.assertIsNone(parsed)
        self.assertIn("cannot be negative", errors[0])

    def test_an_offset_on_an_absolute_window_is_refused(self):
        result, errors = nlq.window_from_dict({
            "kind": "period",
            "grain": "month",
            "start": "2024-01-01",
            "offset": 2,
        })

        self.assertIsNone(result)
        self.assertIn("only means something", errors[0])

    def test_a_single_period_needs_its_start(self):
        parsed, errors = nlq.window_from_dict({
            "kind": "period",
            "grain": "month",
        })

        self.assertIsNone(parsed)
        self.assertIn("date the period starts", errors[0])

    def test_a_count_of_zero_is_refused(self):
        parsed, errors = nlq.window_from_dict({
            "kind": "last_periods",
            "count": 0,
            "grain": "month",
        })

        self.assertIsNone(parsed)
        self.assertIn("at least 1", errors[0])

    def test_a_count_that_is_not_a_number_is_refused(self):
        parsed, errors = nlq.window_from_dict({
            "kind": "last_periods",
            "count": "three",
            "grain": "month",
        })

        self.assertIsNone(parsed)
        self.assertIn("whole number", errors[0])

    def test_a_boolean_is_not_a_count(self):
        # True is an int in Python, which would slip through a naive check.
        parsed, errors = nlq.window_from_dict({
            "kind": "last_periods",
            "count": True,
            "grain": "month",
        })

        self.assertIsNone(parsed)
        self.assertIn("whole number", errors[0])

    def test_every_kind_can_be_described(self):
        windows = [
            nlq.Window(),
            nlq.Window("last_periods", count=1, grain="month"),
            nlq.Window("last_periods", count=3, grain="month"),
            nlq.Window("between_dates", start="2024-01-01",
                       end="2024-03-31"),
            nlq.Window("between_dates", start="2024-01-01"),
            nlq.Window("between_dates", end="2024-03-31"),
            nlq.Window("period", grain="month", start="2024-01-01"),
            nlq.Window("last_periods", count=1, grain="month", offset=1),
            nlq.Window("last_periods", count=3, grain="week", offset=2),
        ]

        for window in windows:
            self.assertTrue(window.describe())


class PlanParsingTest(unittest.TestCase):

    def minimal(self, **overrides):
        payload = {"intent": "aggregate", "measure": "revenue"}
        payload.update(overrides)

        return nlq.plan_from_dict(payload)

    def test_a_plan_round_trips(self):
        plan = nlq.QueryPlan(
            intent="breakdown",
            measure="revenue",
            aggregation="sum",
            date="order_date",
            grain="month",
            segment="region",
            filters=(nlq.Filter("units", "greater_than", (5,)),),
            window=nlq.Window("last_periods", count=3, grain="month"),
            top_n=5,
            direction="lowest",
            question="which region sells least",
            source="model"
        )

        parsed, errors = nlq.plan_from_dict(plan.to_dict())

        self.assertEqual(errors, [])
        self.assertEqual(parsed, plan)

    def test_an_unknown_intent_is_refused(self):
        parsed, errors = self.minimal(intent="forecast_next_year")

        self.assertIsNone(parsed)
        self.assertIn("not a supported intent", errors[0])

    def test_an_unknown_field_is_refused_not_ignored(self):
        # A model that invents a field believes it changed the question.
        # Ignoring it would answer a different one confidently.
        parsed, errors = self.minimal(sql="select 1")

        self.assertIsNone(parsed)
        self.assertIn("sql", errors[0])

    def test_an_unknown_aggregation_is_refused(self):
        parsed, errors = self.minimal(aggregation="geometric_mean")

        self.assertIsNone(parsed)
        self.assertIn("not a supported aggregation", errors[0])

    def test_an_unknown_grain_is_refused(self):
        parsed, errors = self.minimal(grain="fortnight")

        self.assertIsNone(parsed)
        self.assertIn("not a period", errors[0])

    def test_an_unknown_direction_is_refused(self):
        parsed, errors = self.minimal(direction="sideways")

        self.assertIsNone(parsed)
        self.assertIn("not a direction", errors[0])

    def test_an_unknown_source_is_refused(self):
        parsed, errors = self.minimal(source="intuition")

        self.assertIsNone(parsed)
        self.assertIn("not a plan source", errors[0])

    def test_a_column_name_must_be_text(self):
        parsed, errors = self.minimal(measure=42)

        self.assertIsNone(parsed)
        self.assertIn("must be text", errors[0])

    def test_a_top_n_beyond_the_cap_is_refused(self):
        parsed, errors = self.minimal(top_n=nlq.MAX_TOP_N + 1)

        self.assertIsNone(parsed)
        self.assertIn("cannot be more than", errors[0])

    def test_a_top_n_of_zero_is_refused(self):
        parsed, errors = self.minimal(top_n=0)

        self.assertIsNone(parsed)
        self.assertIn("at least 1", errors[0])

    def test_a_filter_error_is_reported_against_the_plan(self):
        parsed, errors = self.minimal(
            filters=[{"column": "region", "operator": "nope"}]
        )

        self.assertIsNone(parsed)
        self.assertIn("Filter 1", errors[0])

    def test_several_problems_are_all_reported(self):
        parsed, errors = nlq.plan_from_dict({
            "intent": "nope",
            "aggregation": "nope",
            "direction": "nope",
        })

        self.assertIsNone(parsed)
        self.assertGreaterEqual(len(errors), 3)

    def test_a_non_object_is_refused(self):
        parsed, errors = nlq.plan_from_dict("total revenue please")

        self.assertIsNone(parsed)
        self.assertIn("not an object", errors[0])

    def test_a_bare_intent_is_enough(self):
        parsed, errors = nlq.plan_from_dict({"intent": "aggregate"})

        self.assertEqual(errors, [])
        self.assertEqual(parsed.aggregation, "sum")
        self.assertEqual(parsed.window.kind, "all")
        self.assertEqual(parsed.source, "rules")

    def test_filters_must_be_a_list(self):
        parsed, errors = self.minimal(filters={"column": "region"})

        self.assertIsNone(parsed)
        self.assertTrue(
            any("filters must be a list" in error for error in errors)
        )


class PlanDescriptionTest(unittest.TestCase):
    """
    The description is what lets a reader catch a right answer to the
    wrong question, so every intent has to produce one.
    """

    def test_every_intent_describes_itself(self):
        for intent in nlq.INTENTS:
            plan = nlq.QueryPlan(
                intent=intent,
                measure="revenue",
                date="order_date",
                segment="region",
                segment_value="north"
            )

            described = plan.describe()

            self.assertTrue(described.endswith("."))
            self.assertIn("revenue", described)

    def test_counting_rows_is_described_as_counting_rows(self):
        plan = nlq.QueryPlan(intent="aggregate", aggregation="count")

        self.assertIn("number of rows", plan.describe())

    def test_the_window_is_stated(self):
        plan = nlq.QueryPlan(
            intent="aggregate",
            measure="revenue",
            window=nlq.Window("last_periods", count=3, grain="month")
        )

        self.assertIn("last 3 months", plan.describe())

    def test_filters_are_stated(self):
        plan = nlq.QueryPlan(
            intent="aggregate",
            measure="revenue",
            filters=(nlq.Filter("region", "equals", ("north",)),)
        )

        self.assertIn("'region' is north", plan.describe())

    def test_the_measure_label_falls_back_to_rows(self):
        plan = nlq.QueryPlan(intent="aggregate", aggregation="count")

        self.assertEqual(plan.measure_label, "Rows")


class ValidationTest(unittest.TestCase):

    def setUp(self):
        self.frame = sales_frame()

    def check(self, **overrides):
        payload = {
            "intent": "aggregate",
            "measure": "revenue",
            "aggregation": "sum",
        }
        payload.update(overrides)

        plan, errors = nlq.plan_from_dict(payload)

        self.assertEqual(errors, [], f"plan was malformed: {errors}")

        return nlq.validate_plan(plan, self.frame)

    def test_a_sound_plan_passes(self):
        result = self.check()

        self.assertTrue(result.valid)
        self.assertEqual(result.errors, [])

    def test_a_split_the_intent_would_ignore_is_refused(self):
        # Found by the live evaluation: "which territory brought in the
        # most" came back as a period ranking with a segment attached. The
        # period ranking ignores the segment, so the answer named a
        # quarter. Refusing the plan sends the reason back to the model.
        for intent in ("aggregate", "extreme", "timeseries", "trend"):
            with self.subTest(intent=intent):
                result = self.check(
                    intent=intent, date="order_date", segment="region"
                )

                self.assertFalse(result.valid)
                self.assertIn("breakdown", " ".join(result.errors))

    def test_a_split_is_accepted_where_it_is_used(self):
        self.assertTrue(self.check(intent="breakdown", segment="region").valid)

    def test_a_missing_measure_column_is_named(self):
        result = self.check(measure="turnover")

        self.assertFalse(result.valid)
        self.assertIn("turnover", result.errors[0])

    def test_a_text_column_cannot_be_totalled(self):
        result = self.check(measure="surname")

        self.assertFalse(result.valid)
        self.assertIn("cannot be measured", result.errors[0])

    def test_a_missing_date_column_is_named(self):
        result = self.check(
            intent="trend",
            date="when"
        )

        self.assertFalse(result.valid)
        self.assertIn("when", result.errors[0])

    def test_a_text_column_cannot_carry_the_timeline(self):
        result = self.check(intent="trend", date="note")

        self.assertFalse(result.valid)
        self.assertIn("cannot carry a timeline", result.errors[0])

    def test_a_trend_without_a_date_is_refused(self):
        result = self.check(intent="trend")

        self.assertFalse(result.valid)
        self.assertIn("needs a date column", result.errors[0])

    def test_a_breakdown_without_a_segment_is_refused(self):
        result = self.check(intent="breakdown")

        self.assertFalse(result.valid)
        self.assertIn("something to split by", result.errors[0])

    def test_an_identifier_cannot_be_a_segment(self):
        result = self.check(intent="breakdown", segment="surname")

        self.assertFalse(result.valid)
        self.assertIn("identifies rows", result.errors[0])

    def test_a_constant_column_cannot_be_a_segment(self):
        result = self.check(intent="breakdown", segment="note")

        self.assertFalse(result.valid)
        self.assertIn("same value in every row", result.errors[0])

    def test_a_window_without_a_date_is_refused(self):
        result = self.check(
            window={"kind": "last_periods", "count": 3, "grain": "month"}
        )

        self.assertFalse(result.valid)
        self.assertTrue(
            any("stretch of time" in error for error in result.errors)
        )

    def test_a_comparison_needs_two_periods(self):
        result = self.check(intent="compare", date="order_date")

        self.assertFalse(result.valid)
        self.assertIn("only one was given", result.errors[0])

    def test_a_comparison_on_another_intent_is_only_a_warning(self):
        result = self.check(
            date="order_date",
            comparison={"kind": "last_periods", "count": 1,
                        "grain": "month"}
        )

        self.assertTrue(result.valid)
        self.assertTrue(
            any("will be ignored" in note for note in result.warnings)
        )

    def test_counting_rows_refuses_a_measure_aggregation(self):
        # A plan with no measure counts rows however it labels itself, so
        # a plan claiming to average has to be stopped rather than
        # answered with a count.
        result = self.check(measure=None, aggregation="mean")

        self.assertFalse(result.valid)
        self.assertIn("needs a measure column", result.errors[0])

    def test_counting_rows_with_count_passes(self):
        result = self.check(measure=None, aggregation="count")

        self.assertTrue(result.valid)

    def test_an_empty_dataset_is_refused(self):
        plan, _ = nlq.plan_from_dict({
            "intent": "aggregate",
            "measure": "revenue",
        })

        result = nlq.validate_plan(plan, self.frame.iloc[0:0])

        self.assertFalse(result.valid)
        self.assertIn("no rows", result.errors[0])

    def test_a_dirty_measure_warns_about_the_excluded_rows(self):
        frame = self.frame.copy()
        frame["revenue"] = frame["revenue"].astype(object)
        frame.loc[0:19, "revenue"] = "n/a"

        plan, _ = nlq.plan_from_dict({
            "intent": "aggregate",
            "measure": "revenue",
        })

        result = nlq.validate_plan(plan, frame)

        self.assertTrue(result.valid)
        self.assertTrue(
            any("20 row(s)" in note for note in result.warnings),
            result.warnings
        )

    def test_a_single_unusable_value_is_still_reported(self):
        # A share would round to 0% here and read as nothing being wrong,
        # while the total quietly excludes a row.
        frame = self.frame.copy()
        frame["revenue"] = frame["revenue"].astype(object)
        frame.loc[0, "revenue"] = "n/a"

        plan, _ = nlq.plan_from_dict({
            "intent": "aggregate",
            "measure": "revenue",
        })

        result = nlq.validate_plan(plan, frame)

        self.assertTrue(result.valid)
        self.assertTrue(
            any("1 row(s)" in note for note in result.warnings),
            result.warnings
        )

    def test_a_clean_measure_warns_about_nothing(self):
        result = self.check()

        self.assertEqual(result.warnings, [])

    def test_unreadable_dates_are_reported_as_a_count(self):
        frame = self.frame.copy()
        frame["order_date"] = frame["order_date"].astype(object)
        frame.loc[0:2, "order_date"] = "not a date"

        plan, _ = nlq.plan_from_dict({
            "intent": "trend",
            "measure": "revenue",
            "date": "order_date",
        })

        result = nlq.validate_plan(plan, frame)

        self.assertTrue(result.valid, result.errors)
        self.assertTrue(
            any("3 row(s)" in note for note in result.warnings),
            result.warnings
        )


class ShareValidationTest(unittest.TestCase):
    """
    A share of a total is a proportion. A share of an average is nothing,
    and rendering one would invite a false reading.
    """

    def setUp(self):
        self.frame = sales_frame()

    def plan_for(self, **overrides):
        payload = {
            "intent": "share",
            "measure": "revenue",
            "aggregation": "sum",
            "segment": "region",
            "segment_value": "north",
        }
        payload.update(overrides)

        plan, errors = nlq.plan_from_dict(payload)

        self.assertEqual(errors, [])

        return nlq.validate_plan(plan, self.frame)

    def test_a_share_of_a_total_is_allowed(self):
        self.assertTrue(self.plan_for().valid)

    def test_a_share_of_an_average_is_refused(self):
        result = self.plan_for(aggregation="mean")

        self.assertFalse(result.valid)
        self.assertTrue(
            any("do not add up" in error for error in result.errors)
        )

    def test_a_share_needs_a_named_part(self):
        result = self.plan_for(segment_value=None)

        self.assertFalse(result.valid)
        self.assertIn("name which part", result.errors[0])

    def test_a_part_that_does_not_exist_is_refused(self):
        result = self.plan_for(segment_value="atlantis")

        self.assertFalse(result.valid)
        self.assertIn("atlantis", result.errors[0])

    def test_every_additive_aggregation_is_accepted(self):
        for name in sorted(segments.ADDITIVE_AGGREGATIONS):
            if name == "count":
                result = self.plan_for(measure=None, aggregation=name)

            else:
                result = self.plan_for(aggregation=name)

            with self.subTest(aggregation=name):
                self.assertTrue(result.valid, result.errors)


class FilterValidationTest(unittest.TestCase):

    def setUp(self):
        self.frame = sales_frame()

    def check(self, entry):
        plan, errors = nlq.plan_from_dict({
            "intent": "aggregate",
            "measure": "revenue",
            "filters": [entry],
        })

        self.assertEqual(errors, [], f"plan was malformed: {errors}")

        return nlq.validate_plan(plan, self.frame)

    def test_a_filter_on_a_real_column_passes(self):
        result = self.check({
            "column": "region",
            "operator": "equals",
            "values": ["north"],
        })

        self.assertTrue(result.valid)

    def test_a_filter_on_a_missing_column_is_named(self):
        result = self.check({
            "column": "country",
            "operator": "equals",
            "values": ["uk"],
        })

        self.assertFalse(result.valid)
        self.assertIn("country", result.errors[0])

    def test_a_numeric_test_on_text_is_refused(self):
        result = self.check({
            "column": "region",
            "operator": "greater_than",
            "values": [5],
        })

        self.assertFalse(result.valid)
        self.assertIn("not numeric", result.errors[0])

    def test_a_numeric_test_against_text_is_refused(self):
        result = self.check({
            "column": "revenue",
            "operator": "greater_than",
            "values": ["lots"],
        })

        self.assertFalse(result.valid)
        self.assertIn("not a number", result.errors[0])

    def test_a_contains_test_on_numbers_warns(self):
        result = self.check({
            "column": "units",
            "operator": "contains",
            "values": ["7"],
        })

        self.assertTrue(result.valid)
        self.assertTrue(
            any("written form" in note for note in result.warnings)
        )


class ResolutionTest(unittest.TestCase):
    """
    A question rarely names its columns. Detection fills them in, and the
    plan description is what makes the assumption visible.
    """

    def setUp(self):
        self.frame = sales_frame()

    def test_a_measure_is_filled_in(self):
        plan = nlq.resolve_plan_columns(
            nlq.QueryPlan(intent="aggregate"),
            self.frame
        )

        self.assertIsNotNone(plan.measure)

    def test_a_date_is_filled_in_for_a_trend(self):
        plan = nlq.resolve_plan_columns(
            nlq.QueryPlan(intent="trend"),
            self.frame
        )

        self.assertEqual(plan.date, "order_date")

    def test_a_segment_is_filled_in_for_a_breakdown(self):
        plan = nlq.resolve_plan_columns(
            nlq.QueryPlan(intent="breakdown"),
            self.frame
        )

        self.assertEqual(plan.segment, "region")

    def test_a_date_is_filled_in_when_a_window_needs_one(self):
        plan = nlq.resolve_plan_columns(
            nlq.QueryPlan(
                intent="aggregate",
                window=nlq.Window("last_periods", count=2, grain="month")
            ),
            self.frame
        )

        self.assertEqual(plan.date, "order_date")

    def test_no_date_is_added_when_nothing_needs_one(self):
        plan = nlq.resolve_plan_columns(
            nlq.QueryPlan(intent="aggregate", measure="revenue"),
            self.frame
        )

        self.assertIsNone(plan.date)

    def test_an_explicit_choice_is_never_overridden(self):
        plan = nlq.resolve_plan_columns(
            nlq.QueryPlan(intent="breakdown", measure="units",
                          segment="region"),
            self.frame
        )

        self.assertEqual(plan.measure, "units")
        self.assertEqual(plan.segment, "region")

    def test_a_dataset_with_nothing_to_measure_falls_back_to_counting(self):
        frame = pd.DataFrame({
            "order_date": pd.date_range("2024-01-01", periods=90),
            "region": ["north", "south", "east"] * 30,
        })

        plan = nlq.resolve_plan_columns(
            nlq.QueryPlan(intent="aggregate", aggregation="sum"),
            frame
        )

        self.assertIsNone(plan.measure)
        self.assertEqual(plan.aggregation, "count")

    def test_a_resolved_plan_validates(self):
        for intent in nlq.INTENTS:
            plan = nlq.resolve_plan_columns(
                nlq.QueryPlan(
                    intent=intent,
                    segment_value=(
                        "north" if intent == "share" else None
                    ),
                    comparison=(
                        nlq.Window("last_periods", count=1, grain="month")
                        if intent == "compare"
                        else None
                    ),
                    window=(
                        nlq.Window("last_periods", count=1, grain="month")
                        if intent == "compare"
                        else nlq.Window()
                    )
                ),
                self.frame
            )

            with self.subTest(intent=intent):
                result = nlq.validate_plan(plan, self.frame)

                self.assertTrue(result.valid, result.errors)


class VocabularyTest(unittest.TestCase):
    """
    The vocabularies are a contract with the model prompt, so a change
    here has to be deliberate.
    """

    def test_every_intent_needing_a_date_is_an_intent(self):
        for intent in nlq.INTENTS_NEEDING_DATE:
            self.assertIn(intent, nlq.INTENTS)

    def test_every_intent_needing_a_segment_is_an_intent(self):
        for intent in nlq.INTENTS_NEEDING_SEGMENT:
            self.assertIn(intent, nlq.INTENTS)

    def test_numeric_and_text_operators_are_operators(self):
        for operator in nlq.NUMERIC_OPERATORS | nlq.TEXT_OPERATORS:
            self.assertIn(operator, nlq.OPERATOR_ARITY)

    def test_the_aggregations_come_from_the_aggregation_module(self):
        plan, errors = nlq.plan_from_dict({
            "intent": "aggregate",
            "measure": "revenue",
            "aggregation": aggregation.AGGREGATIONS[0],
        })

        self.assertEqual(errors, [])
        self.assertIsNotNone(plan)


if __name__ == "__main__":
    unittest.main()
