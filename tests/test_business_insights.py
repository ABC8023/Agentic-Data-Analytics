"""
The evidence ledger, and the promises it makes.

Three of these promises are worth naming, because they are what separate a
ledger from a list of assertions.

Every card carries the arithmetic that produced it. A card with a finding
and no calculation is a claim the reader cannot check.

Every recommendation names evidence that exists. A recommendation pointing
at a card that is not in the ledger is unfalsifiable, and the interface
would render it with a dead link.

Nothing states a cause. A table of totals establishes what moved, never
why, and the prose has to keep that line.
"""

import unittest

import numpy as np
import pandas as pd

import aggregation
import business_insights
import forecasting
import formatting
import schema
import segments
import timeseries
from tests.fixtures import messy_frame, timeline_frame


def steady_frame(days: int = 730, value: float = 100.0) -> pd.DataFrame:
    """A clean daily series with no drift and no spike."""

    names = ["north", "south"]

    return pd.DataFrame({
        "order_date": pd.date_range("2023-01-01", periods=days, freq="D"),
        "revenue": [value] * days,
        "region": [names[index % len(names)] for index in range(days)],
    })


def declining_frame(days: int = 730) -> pd.DataFrame:
    """A daily series that falls steadily, then falls sharply at the end."""

    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    values = np.linspace(200.0, 100.0, days)
    values[-31:] = values[-31:] * 0.5

    names = ["north", "south", "east", "west"]

    return pd.DataFrame({
        "order_date": dates,
        "revenue": values,
        "region": [names[index % len(names)] for index in range(days)],
    })


def concentrated_frame(days: int = 400) -> pd.DataFrame:
    """One part carrying most of the total."""

    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    rows = []

    for name, value in (("whale", 90.0), ("minnow", 6.0), ("shrimp", 4.0)):
        for moment in dates:
            rows.append({
                "order_date": moment,
                "region": name,
                "revenue": value,
            })

    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame, aggregation_name: str = "sum"):
    """The headline series the evidence is built from."""

    return aggregation.aggregate_periods(
        frame,
        date_column="order_date",
        measure_column="revenue",
        aggregation=aggregation_name,
        grain="month"
    )


def period_frame(values, labels=None):
    """A minimal period series, for testing a card in isolation."""

    index = pd.date_range("2024-01-01", periods=len(values), freq="MS")

    return pd.DataFrame({
        "period_start": index,
        "value": values,
        "row_count": [10] * len(values),
        "period_days": [31] * len(values),
        "period_label": labels or [
            moment.strftime("%b %Y") for moment in index
        ],
    })


class EvidenceCardTest(unittest.TestCase):

    def test_a_card_carries_everything_the_interface_needs(self):
        card = business_insights.evidence_card(
            "example",
            "Title",
            "Finding.",
            "Calculation.",
            kind="test"
        )

        for key in (
            "id",
            "title",
            "finding",
            "calculation",
            "kind",
            "severity",
            "detail",
        ):
            self.assertIn(key, card)

    def test_severity_defaults_to_the_quietest_level(self):
        card = business_insights.evidence_card(
            "example", "T", "F", "C", kind="test"
        )

        self.assertEqual(card["severity"], "info")

    def test_detail_is_never_shared_between_cards(self):
        first = business_insights.evidence_card(
            "a", "T", "F", "C", kind="test"
        )
        second = business_insights.evidence_card(
            "b", "T", "F", "C", kind="test"
        )

        first["detail"]["x"] = 1

        self.assertEqual(second["detail"], {})


class CoverageTest(unittest.TestCase):

    def setUp(self):
        self.aggregated = aggregate(steady_frame())
        self.card = business_insights.coverage_evidence(self.aggregated)

    def test_it_states_the_period_count_and_the_rows(self):
        self.assertIn("periods", self.card["finding"])
        self.assertIn("rows", self.card["finding"])

    def test_it_names_the_date_column_in_the_calculation(self):
        self.assertIn("order_date", self.card["calculation"])

    def test_it_mentions_adjustments_when_there_were_any(self):
        frame = steady_frame(days=400)
        frame.loc[0:5, "order_date"] = None

        card = business_insights.coverage_evidence(aggregate(frame))

        self.assertIn("adjustment", card["calculation"])


class MovementTest(unittest.TestCase):

    def card_for(self, values, aggregated=None):
        aggregated = aggregated or {
            "periods": period_frame(values),
            "grain": "month",
        }

        return business_insights.movement_evidence(aggregated, "Revenue")

    def test_a_single_period_has_no_movement(self):
        self.assertIsNone(self.card_for([100.0]))

    def test_a_missing_latest_value_has_no_movement(self):
        self.assertIsNone(self.card_for([100.0, np.nan]))

    def test_a_rise_is_described_as_a_rise(self):
        card = self.card_for([100.0, 120.0])

        self.assertIn("rose", card["finding"])
        self.assertAlmostEqual(card["detail"]["relative_change"], 0.2)

    def test_a_fall_is_described_as_a_fall(self):
        card = self.card_for([100.0, 80.0])

        self.assertIn("fell", card["finding"])
        self.assertAlmostEqual(card["detail"]["change"], -20.0)

    def test_a_small_move_stays_quiet(self):
        card = self.card_for([100.0, 101.0])

        self.assertEqual(card["severity"], "info")

    def test_a_large_fall_asks_for_action(self):
        card = self.card_for([100.0, 70.0])

        self.assertEqual(card["severity"], "act")

    def test_a_large_rise_is_worth_watching_not_acting(self):
        card = self.card_for([100.0, 140.0])

        self.assertEqual(card["severity"], "watch")

    def test_the_calculation_shows_both_figures(self):
        card = self.card_for([100.0, 80.0])

        self.assertIn("100", card["calculation"])
        self.assertIn("80", card["calculation"])

    def test_a_zero_baseline_does_not_divide_by_zero(self):
        card = self.card_for([0.0, 50.0])

        self.assertIsNone(card["detail"]["relative_change"])


class TrendEvidenceTest(unittest.TestCase):

    def test_a_failed_fit_produces_no_card(self):
        self.assertIsNone(
            business_insights.trend_evidence(
                {"success": False},
                "month",
                "Revenue"
            )
        )

    def test_a_decline_is_worth_watching(self):
        aggregated = aggregate(declining_frame())
        periods = aggregated["periods"]
        trend = timeseries.fit_trend(
            periods["period_start"],
            periods["value"]
        )

        card = business_insights.trend_evidence(
            trend,
            aggregated["grain"],
            "Revenue"
        )

        self.assertEqual(card["detail"]["direction"], "falling")
        self.assertEqual(card["severity"], "watch")

    def test_the_calculation_reports_the_confidence_interval(self):
        aggregated = aggregate(declining_frame())
        periods = aggregated["periods"]
        trend = timeseries.fit_trend(
            periods["period_start"],
            periods["value"]
        )

        card = business_insights.trend_evidence(
            trend,
            aggregated["grain"],
            "Revenue"
        )

        self.assertIn("Theil-Sen", card["calculation"])
        self.assertIn("zero", card["calculation"])


class QualityEvidenceTest(unittest.TestCase):

    def setUp(self):
        self.cards = business_insights.quality_evidence(
            messy_frame(),
            "monthly_spend"
        )
        self.by_id = {card["id"]: card for card in self.cards}

    def test_gaps_in_the_measure_are_reported(self):
        self.assertIn("measure_gaps", self.by_id)

    def test_duplicate_rows_are_reported(self):
        self.assertIn("duplicates", self.by_id)
        self.assertEqual(self.by_id["duplicates"]["severity"], "act")

    def test_a_constant_column_is_reported(self):
        card = self.by_id["constant_columns"]

        self.assertIn("source", card["detail"]["columns"])

    def test_a_clean_frame_reports_nothing(self):
        cards = business_insights.quality_evidence(
            steady_frame(days=10).drop(columns=["revenue"]).assign(
                revenue=range(10),
                region=list("abcdefghij")
            ),
            "revenue"
        )

        self.assertEqual(cards, [])

    def test_a_tiny_share_of_gaps_is_watched_not_acted_on(self):
        frame = steady_frame(days=1000)
        frame.loc[0, "revenue"] = np.nan

        cards = business_insights.quality_evidence(frame, "revenue")
        by_id = {card["id"]: card for card in cards}

        self.assertEqual(by_id["measure_gaps"]["severity"], "watch")


class RelationshipEvidenceTest(unittest.TestCase):

    def test_no_measure_means_no_card(self):
        self.assertIsNone(
            business_insights.relationship_evidence(steady_frame(), None)
        )

    def test_a_single_numeric_column_means_no_card(self):
        frame = pd.DataFrame({"revenue": [1.0, 2.0, 3.0]})

        self.assertIsNone(
            business_insights.relationship_evidence(frame, "revenue")
        )

    def test_a_weak_relationship_is_not_reported(self):
        rng = np.random.default_rng(3)
        frame = pd.DataFrame({
            "revenue": rng.normal(0, 1, 400),
            "other": rng.normal(0, 1, 400),
        })

        self.assertIsNone(
            business_insights.relationship_evidence(frame, "revenue")
        )

    def test_a_strong_relationship_is_reported(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame({"units": rng.normal(50, 10, 300)})
        frame["revenue"] = frame["units"] * 3 + rng.normal(0, 1, 300)

        card = business_insights.relationship_evidence(frame, "revenue")

        self.assertEqual(card["detail"]["column"], "units")
        self.assertGreater(card["detail"]["correlation"], 0.9)

    def test_it_refuses_to_call_a_correlation_a_cause(self):
        rng = np.random.default_rng(5)
        frame = pd.DataFrame({"units": rng.normal(50, 10, 300)})
        frame["revenue"] = frame["units"] * 3 + rng.normal(0, 1, 300)

        card = business_insights.relationship_evidence(frame, "revenue")

        self.assertIn("not a", card["finding"])
        self.assertIn("cause", card["finding"])


class ConcentrationTest(unittest.TestCase):

    def breakdown_for(self, frame):
        aggregated = aggregate(frame)
        matrix = segments.segment_matrix(
            frame,
            date="order_date",
            segment="region",
            periods=aggregated["periods"],
            measure="revenue",
            aggregation_name="sum",
            grain="month"
        )["matrix"]

        return segments.segment_breakdown(matrix, additive=True)

    def test_a_dominant_part_is_named(self):
        breakdown = self.breakdown_for(concentrated_frame())

        card = business_insights.concentration_evidence(
            breakdown,
            "region",
            "Revenue"
        )

        self.assertEqual(card["detail"]["largest_segment"], "whale")
        self.assertEqual(card["severity"], "act")
        self.assertIn("action_hint", card["detail"])

    def test_an_even_split_stays_quiet(self):
        breakdown = self.breakdown_for(steady_frame())

        card = business_insights.concentration_evidence(
            breakdown,
            "region",
            "Revenue"
        )

        self.assertEqual(card["severity"], "info")

    def test_half_the_total_across_two_parts_is_not_concentration(self):
        # An even split of two parts puts 50% on the largest. Judging that
        # against a bare 50% threshold would warn about every two-part
        # breakdown in the product.
        breakdown = self.breakdown_for(steady_frame())

        self.assertAlmostEqual(breakdown["largest_share"], 0.5, places=6)
        self.assertAlmostEqual(breakdown["dominance"], 1.0, places=6)

        card = business_insights.concentration_evidence(
            breakdown,
            "region",
            "Revenue"
        )

        self.assertEqual(card["severity"], "info")

    def test_half_the_total_across_many_parts_is_concentration(self):
        index = pd.date_range("2024-01-01", periods=2, freq="MS")
        values = {"big": [50.0, 50.0]}

        for number in range(10):
            values[f"small_{number}"] = [5.0, 5.0]

        breakdown = segments.segment_breakdown(
            pd.DataFrame(values, index=index),
            additive=True
        )

        card = business_insights.concentration_evidence(
            breakdown,
            "region",
            "Revenue"
        )

        self.assertAlmostEqual(breakdown["largest_share"], 0.5, places=6)
        self.assertGreater(breakdown["dominance"], 5.0)
        self.assertEqual(card["severity"], "watch")

    def test_the_finding_compares_against_an_even_split(self):
        breakdown = self.breakdown_for(concentrated_frame())

        card = business_insights.concentration_evidence(
            breakdown,
            "region",
            "Revenue"
        )

        self.assertIn("even split", card["finding"])

    def test_a_card_is_produced_even_when_shares_are_withheld(self):
        index = pd.date_range("2024-01-01", periods=2, freq="MS")
        matrix = pd.DataFrame(
            {"good": [100.0, 110.0], "bad": [-30.0, -40.0]},
            index=index
        )
        breakdown = segments.segment_breakdown(matrix, additive=True)

        card = business_insights.concentration_evidence(
            breakdown,
            "region",
            "Profit"
        )

        self.assertEqual(card["id"], "concentration")
        self.assertEqual(card["severity"], "info")
        self.assertFalse(card["detail"]["shares_meaningful"])
        self.assertIn("negative", card["finding"])

    def test_the_calculation_names_the_grand_total(self):
        breakdown = self.breakdown_for(concentrated_frame())

        card = business_insights.concentration_evidence(
            breakdown,
            "region",
            "Revenue"
        )

        self.assertIn(
            formatting.format_number(breakdown["grand_total"]),
            card["calculation"]
        )


class SegmentMovementEvidenceTest(unittest.TestCase):

    def waterfall_for(self, values, totals=None):
        index = pd.date_range("2024-01-01", periods=2, freq="MS")
        matrix = pd.DataFrame(values, index=index)

        if totals is None:
            totals = matrix.sum(axis=1).to_numpy()

        periods = period_frame(list(totals))
        periods["period_start"] = index

        return segments.segment_waterfall(
            matrix,
            periods,
            additive=True,
            grain="month"
        )

    def test_a_failed_waterfall_produces_no_card(self):
        self.assertIsNone(
            business_insights.segment_movement_evidence(
                {"success": False},
                "region",
                "Revenue"
            )
        )

    def test_the_top_mover_is_named(self):
        card = business_insights.segment_movement_evidence(
            self.waterfall_for({
                "north": [100.0, 40.0],
                "south": [50.0, 52.0],
            }),
            "region",
            "Revenue"
        )

        self.assertEqual(card["detail"]["top_mover"], "north")
        self.assertIn("north", card["finding"])

    def test_a_dominant_faller_asks_for_action(self):
        card = business_insights.segment_movement_evidence(
            self.waterfall_for({
                "north": [100.0, 40.0],
                "south": [50.0, 50.0],
            }),
            "region",
            "Revenue"
        )

        self.assertEqual(card["severity"], "act")

    def test_offsetting_movements_are_disclosed(self):
        card = business_insights.segment_movement_evidence(
            self.waterfall_for({
                "north": [100.0, 40.0],
                "south": [50.0, 105.0],
            }),
            "region",
            "Revenue"
        )

        self.assertIn("rose while", card["finding"])

    def test_a_part_that_disappeared_is_disclosed(self):
        card = business_insights.segment_movement_evidence(
            self.waterfall_for({
                "north": [100.0, 100.0],
                "south": [50.0, 0.0],
            }),
            "region",
            "Revenue"
        )

        self.assertIn("recorded nothing", card["finding"])

    def test_the_calculation_reports_the_residual(self):
        card = business_insights.segment_movement_evidence(
            self.waterfall_for(
                {"north": [100.0, 90.0]},
                totals=[100.0, 80.0]
            ),
            "region",
            "Revenue"
        )

        self.assertIn("unaccounted", card["calculation"])
        self.assertFalse(card["detail"]["reconciles"])


class SegmentLabelEvidenceTest(unittest.TestCase):

    def test_a_fully_labelled_breakdown_produces_no_card(self):
        self.assertIsNone(
            business_insights.segment_label_evidence(
                {"unlabelled_rows": 0, "unlabelled_share": 0.0,
                 "rows_used": 100},
                "region"
            )
        )

    def test_a_large_share_of_blanks_asks_for_action(self):
        card = business_insights.segment_label_evidence(
            {"unlabelled_rows": 30, "unlabelled_share": 0.3,
             "rows_used": 100},
            "region"
        )

        self.assertEqual(card["severity"], "act")
        self.assertIn(segments.UNLABELLED_TEXT, card["finding"])

    def test_it_says_the_rows_were_kept_not_dropped(self):
        card = business_insights.segment_label_evidence(
            {"unlabelled_rows": 30, "unlabelled_share": 0.3,
             "rows_used": 100},
            "region"
        )

        self.assertIn("rather than dropped", card["finding"])


class SegmentEvidenceTest(unittest.TestCase):

    def setUp(self):
        self.frame = concentrated_frame()
        self.aggregated = aggregate(self.frame)
        self.result = business_insights.segment_evidence(
            self.frame,
            measure="revenue",
            date="order_date",
            segment="region",
            periods=self.aggregated["periods"],
            aggregation_name="sum",
            grain="month",
            measure_label="Revenue"
        )

    def test_it_produces_a_concentration_card(self):
        identifiers = {card["id"] for card in self.result["cards"]}

        self.assertIn("concentration", identifiers)

    def test_it_produces_a_movement_card(self):
        identifiers = {card["id"] for card in self.result["cards"]}

        self.assertIn("segment_movement", identifiers)

    def test_it_returns_the_matrix_for_charting(self):
        self.assertTrue(self.result["matrix"]["success"])

    def test_an_unusable_segment_returns_an_explanation(self):
        frame = self.frame.copy()
        frame["reference"] = [
            f"REF-{index}" for index in range(len(frame))
        ]

        result = business_insights.segment_evidence(
            frame,
            measure="revenue",
            date="order_date",
            segment="reference",
            periods=self.aggregated["periods"]
        )

        self.assertEqual(result["cards"], [])
        self.assertIn("identifies rows", result["error"])

    def test_a_broken_breakdown_is_flagged_as_a_quality_problem(self):
        # Hand the breakdown a period series whose totals do not match the
        # rows, which is what a stale cache would look like.
        periods = self.aggregated["periods"].copy()
        periods["value"] = periods["value"] * 2

        result = business_insights.segment_evidence(
            frame=self.frame,
            measure="revenue",
            date="order_date",
            segment="region",
            periods=periods,
            aggregation_name="sum",
            grain="month"
        )

        identifiers = {card["id"] for card in result["cards"]}

        self.assertIn("segment_reconciliation", identifiers)


class KpiTest(unittest.TestCase):

    def setUp(self):
        self.aggregated = aggregate(steady_frame())
        periods = self.aggregated["periods"]
        self.trend = timeseries.fit_trend(
            periods["period_start"],
            periods["value"]
        )

    def test_four_figures_are_produced(self):
        kpis = business_insights.build_kpis(
            self.aggregated,
            self.trend,
            "Revenue"
        )

        self.assertEqual(len(kpis), 4)

    def test_every_figure_has_a_label_a_value_and_a_note(self):
        kpis = business_insights.build_kpis(
            self.aggregated,
            self.trend,
            "Revenue"
        )

        for kpi in kpis:
            self.assertTrue(kpi["label"])
            self.assertTrue(kpi["value"])
            self.assertIsInstance(kpi["note"], str)

    def test_a_single_period_says_so_rather_than_showing_a_change(self):
        aggregated = {
            "periods": period_frame([100.0]),
            "grain": "month",
            "first_period": pd.Timestamp("2024-01-01"),
            "last_period": pd.Timestamp("2024-01-01"),
        }

        kpis = business_insights.build_kpis(
            aggregated,
            {"success": False},
            "Revenue"
        )
        change = next(
            kpi for kpi in kpis
            if kpi["label"] == "Change on previous"
        )

        self.assertIn("one period", change["note"])

    def test_the_change_is_signed(self):
        aggregated = {
            "periods": period_frame([100.0, 120.0]),
            "grain": "month",
            "first_period": pd.Timestamp("2024-01-01"),
            "last_period": pd.Timestamp("2024-02-01"),
        }

        kpis = business_insights.build_kpis(
            aggregated,
            {"success": False},
            "Revenue"
        )
        change = next(
            kpi for kpi in kpis
            if kpi["label"] == "Change on previous"
        )

        self.assertTrue(change["value"].startswith("+"))


class RecommendationTest(unittest.TestCase):

    def test_no_evidence_produces_no_recommendations(self):
        self.assertEqual(business_insights.recommend([]), [])

    def test_anomalies_produce_a_recommendation(self):
        evidence = [business_insights.evidence_card(
            "anomalies", "T", "F", "C",
            kind="anomaly",
            severity="act",
            detail={"anomaly_count": 2, "periods": ["Feb 2024"]}
        )]

        recommendations = business_insights.recommend(evidence)

        self.assertEqual(len(recommendations), 1)
        self.assertIn("Feb 2024", recommendations[0]["action"])

    def test_urgent_recommendations_come_first(self):
        evidence = [
            business_insights.evidence_card(
                "trend", "T", "F", "C",
                kind="trend",
                severity="watch",
                detail={"direction": "falling", "significant": True}
            ),
            business_insights.evidence_card(
                "anomalies", "T", "F", "C",
                kind="anomaly",
                severity="act",
                detail={"anomaly_count": 1, "periods": ["Feb 2024"]}
            ),
        ]

        recommendations = business_insights.recommend(evidence)

        self.assertEqual(recommendations[0]["severity"], "act")
        self.assertEqual(recommendations[0]["priority"], 1)

    def test_priorities_are_numbered_from_one_without_gaps(self):
        evidence = [
            business_insights.evidence_card(
                "anomalies", "T", "F", "C",
                kind="anomaly", severity="act",
                detail={"anomaly_count": 1, "periods": ["Feb 2024"]}
            ),
            business_insights.evidence_card(
                "duplicates", "Duplicate rows", "F", "C",
                kind="quality", severity="act", detail={"rows": 5}
            ),
            business_insights.evidence_card(
                "trend", "T", "F", "C",
                kind="trend", severity="watch",
                detail={"direction": "falling", "significant": True}
            ),
        ]

        recommendations = business_insights.recommend(evidence)

        self.assertEqual(
            [entry["priority"] for entry in recommendations],
            list(range(1, len(recommendations) + 1))
        )

    def test_every_recommendation_names_evidence_that_exists(self):
        evidence = [
            business_insights.evidence_card(
                "anomalies", "T", "F", "C",
                kind="anomaly", severity="act",
                detail={"anomaly_count": 1, "periods": ["Feb 2024"]}
            ),
            business_insights.evidence_card(
                "concentration", "T", "F", "C",
                kind="segment", severity="watch",
                detail={"action_hint": "Look at it."}
            ),
        ]
        available = {card["id"] for card in evidence}

        for entry in business_insights.recommend(evidence):
            for identifier in entry["rests_on"]:
                self.assertIn(identifier, available)

    def test_a_failed_backtest_warns_against_planning_on_it(self):
        recommendations = business_insights.recommend(
            [],
            {
                "success": True,
                "beat_naive": False,
                "verdict": "It did not beat the last value.",
            }
        )

        self.assertEqual(len(recommendations), 1)
        self.assertIn("projection", recommendations[0]["title"])

    def test_a_successful_backtest_produces_no_warning(self):
        recommendations = business_insights.recommend(
            [],
            {"success": True, "beat_naive": True, "verdict": "Fine."}
        )

        self.assertEqual(recommendations, [])

    def test_a_rise_does_not_trigger_the_fall_investigation(self):
        evidence = [business_insights.evidence_card(
            "movement", "T", "F", "C",
            kind="movement", severity="watch",
            detail={"latest_label": "Feb 2024"}
        )]

        self.assertEqual(business_insights.recommend(evidence), [])

    def test_recommendations_are_phrased_as_investigations(self):
        evidence = [business_insights.evidence_card(
            "anomalies", "T", "F", "C",
            kind="anomaly", severity="act",
            detail={"anomaly_count": 1, "periods": ["Feb 2024"]}
        )]

        for entry in business_insights.recommend(evidence):
            for text in (entry["title"], entry["action"], entry["why"]):
                self.assertNotIn("because of", text.lower())
                self.assertNotIn("caused by", text.lower())


class HeadlineTest(unittest.TestCase):

    def test_movement_leads_when_there_is_movement(self):
        evidence = [business_insights.evidence_card(
            "movement", "T", "Revenue fell 10%.", "C",
            kind="movement", detail={}
        )]

        self.assertEqual(
            business_insights.headline(evidence, "Revenue"),
            "Revenue fell 10%."
        )

    def test_the_trend_leads_when_there_is_no_movement(self):
        evidence = [business_insights.evidence_card(
            "trend", "T", "Revenue is falling.", "C",
            kind="trend", detail={}
        )]

        self.assertEqual(
            business_insights.headline(evidence, "Revenue"),
            "Revenue is falling."
        )

    def test_anomalies_are_appended(self):
        evidence = [
            business_insights.evidence_card(
                "movement", "T", "Revenue fell 10%.", "C",
                kind="movement", detail={}
            ),
            business_insights.evidence_card(
                "anomalies", "T", "F", "C",
                kind="anomaly", detail={"anomaly_count": 2}
            ),
        ]

        self.assertIn(
            "outside the expected band",
            business_insights.headline(evidence, "Revenue")
        )

    def test_an_empty_ledger_still_produces_a_sentence(self):
        sentence = business_insights.headline([], "Revenue")

        self.assertTrue(sentence.endswith("."))


class AnalysisTest(unittest.TestCase):

    def setUp(self):
        self.frame = timeline_frame()
        self.analysis = business_insights.analyse_business(
            self.frame,
            measure="revenue",
            date="order_date",
            segment="region",
            aggregation_name="sum",
            grain="month"
        )

    def test_the_analysis_succeeds(self):
        self.assertTrue(self.analysis["success"])

    def test_a_bad_date_column_fails_with_an_explanation(self):
        analysis = business_insights.analyse_business(
            self.frame,
            measure="revenue",
            date="nope"
        )

        self.assertFalse(analysis["success"])
        self.assertIn("nope", analysis["error"])

    def test_every_card_carries_its_arithmetic(self):
        for card in self.analysis["evidence"]:
            self.assertTrue(
                card["calculation"].strip(),
                f"{card['id']} has no calculation"
            )

    def test_every_recommendation_rests_on_evidence_in_the_ledger(self):
        available = {card["id"] for card in self.analysis["evidence"]}

        for entry in self.analysis["recommendations"]:
            for identifier in entry["rests_on"]:
                self.assertIn(identifier, available)

    def test_the_ledger_is_ordered_by_severity(self):
        ranks = [
            business_insights.SEVERITY_ORDER[card["severity"]]
            for card in self.analysis["evidence"]
        ]

        self.assertEqual(ranks, sorted(ranks))

    def test_the_spike_is_found(self):
        self.assertTrue(self.analysis["anomalies"]["success"])
        self.assertGreater(
            self.analysis["anomalies"]["anomaly_count"],
            0
        )

    def test_the_segment_breakdown_reconciles(self):
        matrix = self.analysis["segment_matrix"]["matrix"]

        np.testing.assert_allclose(
            matrix.sum(axis=1).to_numpy(),
            self.analysis["periods"]["value"].to_numpy()
        )

    def test_it_works_without_a_segment(self):
        analysis = business_insights.analyse_business(
            self.frame,
            measure="revenue",
            date="order_date"
        )

        self.assertTrue(analysis["success"])
        self.assertIsNone(analysis["segment_breakdown"])

    def test_it_works_counting_rows(self):
        analysis = business_insights.analyse_business(
            self.frame,
            measure=None,
            date="order_date",
            segment="region",
            aggregation_name="count"
        )

        self.assertTrue(analysis["success"])
        self.assertEqual(analysis["measure_label"], "Rows")

    def test_a_non_additive_aggregation_is_handled(self):
        analysis = business_insights.analyse_business(
            self.frame,
            measure="revenue",
            date="order_date",
            segment="region",
            aggregation_name="mean"
        )

        self.assertTrue(analysis["success"])

    def test_the_grain_is_detected_when_not_given(self):
        analysis = business_insights.analyse_business(
            self.frame,
            measure="revenue",
            date="order_date"
        )

        self.assertIn(analysis["grain"], aggregation.PERIOD_GRAINS)


class BriefTest(unittest.TestCase):

    def setUp(self):
        detected = schema.detect_business_schema(timeline_frame())
        self.analysis = business_insights.analyse_business(
            timeline_frame(),
            measure="revenue",
            date="order_date",
            segment="region",
            aggregation_name="sum",
            grain="month",
            schema_summary=schema.describe_business_schema(detected)
        )
        self.brief = self.analysis["brief"]

    def test_it_opens_with_a_heading(self):
        self.assertTrue(self.brief.startswith("# Executive brief"))

    def test_it_carries_the_headline(self):
        self.assertIn(self.analysis["headline"], self.brief)

    def test_it_lists_every_kpi(self):
        for kpi in self.analysis["kpis"]:
            self.assertIn(kpi["label"], self.brief)

    def test_it_shows_the_calculation_for_every_card(self):
        for card in self.analysis["evidence"]:
            self.assertIn(card["title"], self.brief)
            self.assertIn(card["calculation"], self.brief)

    def test_it_marks_recommendations_as_suggestions_not_causes(self):
        if self.analysis["recommendations"]:
            self.assertIn("not established causes", self.brief)

    def test_it_lists_the_timeline_adjustments(self):
        for note in self.analysis["notes"]:
            self.assertIn(note, self.brief)

    def test_it_never_leaks_a_placeholder(self):
        for leak in ("None", "nan", "{", "}"):
            self.assertNotIn(leak, self.brief)

    def test_a_horizon_is_respected(self):
        analysis = business_insights.analyse_business(
            timeline_frame(),
            measure="revenue",
            date="order_date",
            horizon=forecasting.DEFAULT_HORIZON
        )

        self.assertTrue(analysis["forecast"]["success"])


if __name__ == "__main__":
    unittest.main()
