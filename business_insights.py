"""
Evidence, and what to do about it.

Three layers live here, and the boundary between them is the point of the
module.

Evidence is calculated. Every card carries the arithmetic that produced
it, so a reader can check the number rather than trust it. Nothing in this
layer is an opinion.

Recommendations are interpretation. They rank the evidence by how much it
matters and say what to look at next. They are written as investigations,
never as causes, because a table of totals cannot tell anyone why
something moved.

The brief is presentation. It arranges the other two into something a
person can send to a colleague.

Keeping the layers apart is what lets the interface show a finding and its
calculation side by side, and what stops a guess about cause being
displayed with the same authority as a sum.
"""

from typing import Any

import numpy as np
import pandas as pd

import aggregation
import anomalies as anomaly_detection
import forecasting
import formatting
import segments
import timeseries

# A single part carrying more than this share of the total is worth
# naming, because the total then mostly describes that one part.
CONCENTRATION_THRESHOLD = 0.5

# A share alone cannot decide concentration. Half the total is dominance
# across ten parts and exactly even across two, so the largest part also
# has to stand this far above an even split before it is called out.
MINIMUM_DOMINANCE = 1.25

# Above this share, one part carries so much that the rest of the reporting
# is effectively invisible.
SEVERE_CONCENTRATION_THRESHOLD = 0.8

# A period-on-period move beyond this is called out even when it sits
# inside the anomaly band, because a reader will notice it anyway.
NOTABLE_MOVEMENT = 0.10

# Missing values in the measure above this share undermine every total
# built from it.
MISSING_MEASURE_THRESHOLD = 0.01

# Correlation strength worth mentioning at all.
NOTABLE_CORRELATION = 0.5

# How many segment parts to name in a breakdown.
TOP_SEGMENT_COUNT = 5

SEVERITY_ORDER = {"act": 0, "watch": 1, "info": 2}


def evidence_card(
    identifier: str,
    title: str,
    finding: str,
    calculation: str,
    kind: str,
    severity: str = "info",
    detail: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Build one evidence card.

    Args:
        identifier:
            Stable key, so a recommendation can point at this card.

        title:
            Short heading.

        finding:
            What the numbers say, in a sentence.

        calculation:
            The arithmetic behind the finding, in words.

        kind:
            Category, used for grouping in the interface.

        severity:
            One of act, watch or info.

        detail:
            Extra figures a caller may want.

    Returns:
        The card.
    """

    return {
        "id": identifier,
        "title": title,
        "finding": finding,
        "calculation": calculation,
        "kind": kind,
        "severity": severity,
        "detail": detail or {},
    }


def coverage_evidence(aggregated: dict[str, Any]) -> dict[str, Any]:
    """
    Describe what the analysis actually covers.

    Args:
        aggregated:
            Output of aggregation.aggregate_periods.

    Returns:
        An evidence card.
    """

    grain = aggregated["grain"]

    span = formatting.format_period_range(
        aggregated["first_period"],
        aggregated["last_period"],
        grain
    )

    adjustments = len(aggregated["notes"])

    finding = (
        f"{formatting.pluralise(aggregated['period_count'], 'period')} "
        f"of {formatting.describe_grain(grain)} data, {span}, built from "
        f"{formatting.pluralise(aggregated['rows_used'], 'row')}."
    )

    calculation = (
        f"Rows were grouped by {formatting.describe_grain(grain)[:-2]} "
        f"period on '{aggregated['date_column']}' and combined with "
        f"{aggregated['aggregation']}."
    )

    if adjustments:
        calculation += (
            f" {formatting.pluralise(adjustments, 'adjustment')} were "
            "made to keep the timeline comparable, listed under the "
            "chart."
        )

    return evidence_card(
        "coverage",
        "What this covers",
        finding,
        calculation,
        kind="coverage",
        detail={
            "period_count": aggregated["period_count"],
            "rows_used": aggregated["rows_used"],
            "grain": grain,
        }
    )


def movement_evidence(
    aggregated: dict[str, Any],
    measure_label: str
) -> dict[str, Any] | None:
    """
    Compare the most recent complete period with the one before it.

    Args:
        aggregated:
            Output of aggregation.aggregate_periods.

        measure_label:
            Name of the measure, for the sentence.

    Returns:
        An evidence card, or None when there is only one period.
    """

    periods = aggregated["periods"]

    if len(periods) < 2:
        return None

    latest = periods["value"].iloc[-1]
    previous = periods["value"].iloc[-2]

    if pd.isna(latest) or pd.isna(previous):
        return None

    latest_label = periods["period_label"].iloc[-1]
    previous_label = periods["period_label"].iloc[-2]

    change = float(latest - previous)
    relative = (
        float(change / abs(previous))
        if previous
        else None
    )

    direction = "rose" if change > 0 else ("fell" if change < 0 else "held")

    if relative is None:
        finding = (
            f"{measure_label} {direction} by "
            f"{formatting.format_number(abs(change))} in {latest_label}."
        )

    else:
        finding = (
            f"{measure_label} {direction} "
            f"{formatting.format_percent(abs(relative))} in "
            f"{latest_label}, a change of "
            f"{formatting.format_signed(change)}."
        )

    calculation = (
        f"{latest_label} {formatting.format_number(latest)} minus "
        f"{previous_label} {formatting.format_number(previous)}"
    )

    if relative is not None:
        calculation += (
            f", divided by {formatting.format_number(abs(previous))}"
        )

    calculation += "."

    severity = "info"

    if relative is not None and abs(relative) >= NOTABLE_MOVEMENT:
        severity = "watch" if change > 0 else "act"

    return evidence_card(
        "movement",
        "Latest movement",
        finding,
        calculation,
        kind="movement",
        severity=severity,
        detail={
            "latest": float(latest),
            "previous": float(previous),
            "change": change,
            "relative_change": relative,
            "latest_label": latest_label,
            "previous_label": previous_label,
        }
    )


def trend_evidence(
    trend: dict[str, Any],
    grain: str,
    measure_label: str
) -> dict[str, Any] | None:
    """
    Describe the direction of the whole series.

    Args:
        trend:
            Output of timeseries.fit_trend.

        grain:
            Period grain.

        measure_label:
            Name of the measure.

    Returns:
        An evidence card, or None when no trend could be fitted.
    """

    if not trend.get("success"):
        return None

    finding = f"{measure_label}: {timeseries.describe_trend(trend, grain)}"

    calculation = (
        "A Theil-Sen line was fitted across the periods, which resists "
        "being pulled by one unusual period. The slope's confidence "
        f"interval runs from "
        f"{formatting.format_number(trend['slope_low'], 4)} to "
        f"{formatting.format_number(trend['slope_high'], 4)} per day, "
        f"and {'excludes' if trend['trend_is_significant'] else 'includes'}"
        " zero."
    )

    severity = "info"

    if trend["trend_is_significant"] and trend["direction"] == "falling":
        severity = "watch"

    return evidence_card(
        "trend",
        "Direction over the whole period",
        finding,
        calculation,
        kind="trend",
        severity=severity,
        detail={
            "direction": trend["direction"],
            "slope_per_period": trend["slope_per_period"],
            "significant": trend["trend_is_significant"],
        }
    )


def anomaly_evidence(detected: dict[str, Any]) -> dict[str, Any] | None:
    """
    Report the periods that fell outside the expected band.

    Args:
        detected:
            Output of anomalies.detect_anomalies.

    Returns:
        An evidence card, or None when detection was not possible.
    """

    if not detected.get("success"):
        return None

    finding = anomaly_detection.describe_anomalies(detected)

    calculation = (
        f"Each period was compared with the trendline and the spread "
        f"around it, measured by median absolute deviation. A period is "
        f"flagged beyond {detected['multiplier']:.2f} deviations, a "
        f"threshold calibrated by simulation so a stable series raises a "
        f"false flag in about one analysis in "
        f"{int(round(1 / detected['false_alarm_rate']))}."
    )

    return evidence_card(
        "anomalies",
        "Unusual periods",
        finding,
        calculation,
        kind="anomaly",
        severity="act" if detected["anomaly_count"] else "info",
        detail={
            "anomaly_count": detected["anomaly_count"],
            "multiplier": detected["multiplier"],
            "periods": [
                entry["period_label"]
                for entry in detected["anomalies"]
            ],
        }
    )


def quality_evidence(
    frame: pd.DataFrame,
    measure: str | None
) -> list[dict[str, Any]]:
    """
    Report the data problems that undermine the totals above.

    Args:
        frame:
            The dataset.

        measure:
            The measure column, or None when counting rows.

    Returns:
        Evidence cards, one per problem found.
    """

    cards = []
    row_count = int(len(frame))

    if measure and measure in frame.columns:
        numeric = pd.to_numeric(frame[measure], errors="coerce")
        unusable = int(numeric.isna().sum())

        if unusable and row_count:
            share = unusable / row_count

            cards.append(evidence_card(
                "measure_gaps",
                "Gaps in the measure",
                (
                    f"{formatting.pluralise(unusable, 'row')} have no "
                    f"usable '{measure}', which is "
                    f"{formatting.format_percent(share)} of the data. "
                    "Every total above excludes them."
                ),
                (
                    f"Counted rows where '{measure}' is blank or not a "
                    f"number: {unusable:,} of {row_count:,}."
                ),
                kind="quality",
                severity=(
                    "act"
                    if share > MISSING_MEASURE_THRESHOLD
                    else "watch"
                ),
                detail={"rows": unusable, "share": share}
            ))

    duplicates = int(frame.duplicated().sum())

    if duplicates:
        cards.append(evidence_card(
            "duplicates",
            "Duplicate rows",
            (
                f"{formatting.pluralise(duplicates, 'row')} are exact "
                "copies of another row, so any total counts them twice."
            ),
            (
                "Counted rows identical to an earlier row across every "
                f"column: {duplicates:,} of {row_count:,}."
            ),
            kind="quality",
            severity="act",
            detail={"rows": duplicates}
        ))

    constant = [
        str(name)
        for name in frame.columns
        if frame[name].nunique(dropna=False) <= 1
    ]

    if constant:
        cards.append(evidence_card(
            "constant_columns",
            "Columns that never change",
            (
                f"{formatting.pluralise(len(constant), 'column')} hold "
                f"the same value in every row: "
                f"{', '.join(constant[:6])}."
            ),
            "Counted distinct values per column and kept those with one.",
            kind="quality",
            severity="info",
            detail={"columns": constant}
        ))

    return cards


def relationship_evidence(
    frame: pd.DataFrame,
    measure: str | None
) -> dict[str, Any] | None:
    """
    Find the numeric column that moves most closely with the measure.

    Args:
        frame:
            The dataset.

        measure:
            The measure column.

    Returns:
        An evidence card, or None when nothing correlates.
    """

    if not measure or measure not in frame.columns:
        return None

    numeric = frame.select_dtypes(include=np.number)

    if measure not in numeric.columns or numeric.shape[1] < 2:
        return None

    correlations = numeric.corr(numeric_only=True)[measure].drop(
        labels=[measure],
        errors="ignore"
    ).dropna()

    if correlations.empty:
        return None

    strongest = correlations.abs().idxmax()
    value = float(correlations[strongest])

    if abs(value) < NOTABLE_CORRELATION:
        return None

    together = "together" if value > 0 else "in opposite directions"

    return evidence_card(
        "relationship",
        "Strongest relationship",
        (
            f"'{measure}' and '{strongest}' move {together}, with a "
            f"correlation of {value:.2f}. This is an association, not a "
            "cause: either could drive the other, or something else "
            "could drive both."
        ),
        (
            f"Pearson correlation between '{measure}' and every other "
            f"numeric column, strongest by absolute value: "
            f"{value:.4f}."
        ),
        kind="relationship",
        severity="info",
        detail={"column": str(strongest), "correlation": value}
    )


def concentration_evidence(
    breakdown: dict[str, Any],
    segment: str,
    measure_label: str
) -> dict[str, Any]:
    """
    Report how much of the total rests on its largest part.

    Args:
        breakdown:
            Output of segments.segment_breakdown.

        segment:
            Name of the column split by.

        measure_label:
            Name of the measure.

    Returns:
        An evidence card. Always returned with the identifier
        "concentration", including when shares could not be computed, so
        the interface has something to show either way.
    """

    part_count = breakdown["part_count"]

    if not breakdown["shares_meaningful"]:
        return evidence_card(
            "concentration",
            "Concentration",
            (
                f"'{segment}' splits the data into "
                f"{formatting.pluralise(part_count, 'part')}, but how "
                "much rests on the largest cannot be stated here. "
                f"{breakdown['shares_withheld_because']}"
            ),
            (
                "Each part was totalled, then the share of the overall "
                "total was left out for the reason given."
            ),
            kind="segment",
            severity="info",
            detail={
                "part_count": part_count,
                "shares_meaningful": False,
                "largest_segment": breakdown["largest_segment"],
            }
        )

    largest = breakdown["largest_segment"]
    largest_share = breakdown["largest_share"]
    dominance = breakdown["dominance"] or 1.0
    top = breakdown["top"]

    named = ", ".join(
        f"{row.segment} "
        f"({formatting.format_percent(row.share)})"
        for row in top.itertuples()
    )

    # Both tests have to pass. A high share with no dominance is just a
    # small number of parts, and calling that concentration would put a
    # warning on every two-part breakdown in the product.
    concentrated = (
        largest_share >= CONCENTRATION_THRESHOLD
        and dominance >= MINIMUM_DOMINANCE
    )

    if concentrated and largest_share >= SEVERE_CONCENTRATION_THRESHOLD:
        severity = "act"

    elif concentrated:
        severity = "watch"

    else:
        severity = "info"

    if severity == "info":
        action_hint = (
            f"No single part dominates. The top "
            f"{breakdown['top_count']} together make up "
            f"{formatting.format_percent(breakdown['top_share'])}."
        )

    else:
        action_hint = (
            f"'{largest}' alone is "
            f"{formatting.format_percent(largest_share)} of "
            f"{measure_label.lower()}. Check whether that is intended, "
            f"and report it separately from the rest so movements in the "
            f"other {formatting.pluralise(part_count - 1, 'part')} are "
            "still visible."
        )

    finding = (
        f"The largest part of '{segment}' is '{largest}' at "
        f"{formatting.format_percent(largest_share)} of the total, "
        f"against {formatting.format_percent(breakdown['even_share'])} "
        f"for an even split across {part_count:,}. "
        f"{formatting.pluralise(breakdown['parts_for_80_percent'], 'part')} "
        f"account for the first 80%. "
        f"Largest first: {named}."
    )

    calculation = (
        f"Rows were grouped by '{segment}' and "
        f"{measure_label.lower()} totalled per part, then each part "
        f"divided by the overall total of "
        f"{formatting.format_number(breakdown['grand_total'])}."
    )

    return evidence_card(
        "concentration",
        "Concentration",
        finding,
        calculation,
        kind="segment",
        severity=severity,
        detail={
            "part_count": part_count,
            "shares_meaningful": True,
            "largest_segment": largest,
            "largest_share": largest_share,
            "even_share": breakdown["even_share"],
            "dominance": dominance,
            "top_share": breakdown["top_share"],
            "parts_for_80_percent": breakdown["parts_for_80_percent"],
            "action_hint": action_hint,
        }
    )


def segment_movement_evidence(
    waterfall: dict[str, Any],
    segment: str,
    measure_label: str
) -> dict[str, Any] | None:
    """
    Attribute the latest period-on-period change to the parts.

    Args:
        waterfall:
            Output of segments.segment_waterfall.

        segment:
            Name of the column split by.

        measure_label:
            Name of the measure.

    Returns:
        An evidence card, or None when the change could not be split.
    """

    if not waterfall.get("success"):
        return None

    table = waterfall["table"]

    if table.empty:
        return None

    total_change = waterfall["total_change"]
    latest_label = waterfall["latest_label"]
    previous_label = waterfall["previous_label"]

    movers = table.head(3)

    if waterfall["explains_total"] and total_change != 0:
        described = "; ".join(
            f"{row.segment} {formatting.format_signed(row.change)} "
            f"({formatting.format_percent(row.contribution)} of the "
            "change)"
            for row in movers.itertuples()
        )

        finding = (
            f"{measure_label} moved "
            f"{formatting.format_signed(total_change)} from "
            f"{previous_label} to {latest_label}. Largest movers by "
            f"'{segment}': {described}."
        )

    else:
        described = "; ".join(
            f"{row.segment} {formatting.format_signed(row.change)}"
            for row in movers.itertuples()
        )

        finding = (
            f"Largest movers by '{segment}' from {previous_label} to "
            f"{latest_label}: {described}."
        )

        if not waterfall["additive"]:
            finding += (
                " These changes do not add up to the overall change, "
                "because an average of the parts is not the average of "
                "the whole."
            )

    if waterfall["offsetting_movements"]:
        finding += (
            f" {waterfall['parts_moved_up']} part(s) rose while "
            f"{waterfall['parts_moved_down']} fell, so the net change is "
            "smaller than the movement underneath it."
        )

    if waterfall["exited_count"]:
        finding += (
            f" {formatting.pluralise(waterfall['exited_count'], 'part')} "
            f"present in {previous_label} recorded nothing in "
            f"{latest_label}."
        )

    calculation = (
        f"Each part's {latest_label} figure minus its "
        f"{previous_label} figure. These add to "
        f"{formatting.format_signed(waterfall['accounted_change'])} "
        f"against an overall change of "
        f"{formatting.format_signed(total_change)}, leaving "
        f"{formatting.format_signed(waterfall['residual'])} unaccounted "
        f"for."
    )

    severity = "info"

    if (
        waterfall["explains_total"]
        and total_change < 0
        and abs(total_change) > 0
    ):
        leader = table.iloc[0]

        if (
            pd.notna(leader["contribution"])
            and leader["contribution"] >= 0.5
        ):
            severity = "act"

    return evidence_card(
        "segment_movement",
        "What moved the latest period",
        finding,
        calculation,
        kind="segment",
        severity=severity,
        detail={
            "latest_label": latest_label,
            "previous_label": previous_label,
            "total_change": total_change,
            "residual": waterfall["residual"],
            "reconciles": waterfall["reconciles"],
            "offsetting_movements": waterfall["offsetting_movements"],
            "top_mover": str(table["segment"].iloc[0]),
            "top_mover_change": float(table["change"].iloc[0]),
            "entered_count": waterfall["entered_count"],
            "exited_count": waterfall["exited_count"],
        }
    )


def segment_label_evidence(
    matrix_result: dict[str, Any],
    segment: str
) -> dict[str, Any] | None:
    """
    Report rows that carry no segment label.

    Args:
        matrix_result:
            Output of segments.segment_matrix.

        segment:
            Name of the column split by.

    Returns:
        An evidence card, or None when every row is labelled.
    """

    unlabelled = matrix_result["unlabelled_rows"]

    if not unlabelled:
        return None

    share = matrix_result["unlabelled_share"]

    return evidence_card(
        "segment_labels",
        "Rows with no segment",
        (
            f"{formatting.pluralise(unlabelled, 'row')} have no "
            f"'{segment}' value, which is "
            f"{formatting.format_percent(share)} of the rows in the "
            f"breakdown. They are grouped under "
            f"'{segments.UNLABELLED_TEXT}' rather than dropped, so the "
            "parts still add up to the total."
        ),
        (
            f"Counted rows where '{segment}' is blank: "
            f"{unlabelled:,} of {matrix_result['rows_used']:,}."
        ),
        kind="quality",
        severity="act" if share > 0.05 else "watch",
        detail={"rows": unlabelled, "share": share}
    )


def segment_evidence(
    frame: pd.DataFrame,
    measure: str | None,
    date: str,
    segment: str,
    periods: pd.DataFrame,
    aggregation_name: str = "sum",
    grain: str = "month",
    measure_label: str = "Rows"
) -> dict[str, Any]:
    """
    Break the total down by segment and turn the result into evidence.

    Args:
        frame:
            The dataset.

        measure:
            Column being measured, or None to count rows.

        date:
            Column holding the dates.

        segment:
            Column to split by.

        periods:
            The headline period series, so the breakdown covers exactly
            the same periods as the chart above it.

        aggregation_name:
            One of aggregation.AGGREGATIONS.

        grain:
            Period grain.

        measure_label:
            Name of the measure, for the prose.

    Returns:
        The cards, the ranked breakdown, the change waterfall, the matrix
        for charting, and an error when the split was not possible.
    """

    empty: dict[str, Any] = {
        "cards": [],
        "breakdown": None,
        "movement": None,
        "matrix": None,
        "error": "",
    }

    matrix_result = segments.segment_matrix(
        frame,
        date=date,
        segment=segment,
        periods=periods,
        measure=measure,
        aggregation_name=aggregation_name,
        grain=grain
    )

    if not matrix_result.get("success"):
        empty["error"] = matrix_result.get(
            "error",
            "The breakdown could not be built."
        )

        return empty

    matrix = matrix_result["matrix"]

    breakdown = segments.segment_breakdown(
        matrix,
        additive=matrix_result["additive"],
        top_count=TOP_SEGMENT_COUNT
    )

    waterfall = segments.segment_waterfall(
        matrix,
        periods,
        additive=matrix_result["additive"],
        grain=grain
    )

    cards = [concentration_evidence(breakdown, segment, measure_label)]

    for card in (
        segment_movement_evidence(waterfall, segment, measure_label),
        segment_label_evidence(matrix_result, segment),
    ):
        if card:
            cards.append(card)

    if matrix_result["additive"] and not matrix_result["reconciles"]:
        cards.append(evidence_card(
            "segment_reconciliation",
            "Breakdown does not tie to the total",
            (
                "The parts of this breakdown do not add back to the "
                "period totals, so it should not be read as a "
                "decomposition. The largest gap is "
                f"{formatting.format_number(matrix_result['largest_discrepancy'])}."
            ),
            (
                "Summed the parts within each period and compared with "
                "the headline figure for that period."
            ),
            kind="quality",
            severity="act",
            detail={
                "largest_discrepancy": (
                    matrix_result["largest_discrepancy"]
                ),
            }
        ))

    return {
        "cards": cards,
        "breakdown": breakdown,
        "movement": waterfall if waterfall.get("success") else None,
        "matrix": matrix_result,
        "error": "",
    }


def build_kpis(
    aggregated: dict[str, Any],
    trend: dict[str, Any],
    measure_label: str
) -> list[dict[str, Any]]:
    """
    Reduce the analysis to four headline figures.

    Args:
        aggregated:
            Output of aggregation.aggregate_periods.

        trend:
            Output of timeseries.fit_trend.

        measure_label:
            Name of the measure.

    Returns:
        Four KPI dictionaries holding a label, a value and a note.
    """

    periods = aggregated["periods"]
    values = periods["value"].dropna()
    grain = aggregated["grain"]

    total = float(values.sum()) if not values.empty else 0.0
    latest = (
        float(periods["value"].iloc[-1])
        if not pd.isna(periods["value"].iloc[-1])
        else None
    )

    average = float(values.mean()) if not values.empty else None

    kpis = [
        {
            "label": f"Total {measure_label.lower()}",
            "value": formatting.format_compact(total),
            "note": formatting.format_period_range(
                aggregated["first_period"],
                aggregated["last_period"],
                grain
            ),
        },
        {
            "label": "Latest period",
            "value": formatting.format_compact(latest),
            "note": periods["period_label"].iloc[-1],
        },
    ]

    if len(periods) >= 2 and latest is not None:
        previous = periods["value"].iloc[-2]

        if pd.notna(previous) and previous:
            relative = (latest - previous) / abs(previous)

            kpis.append({
                "label": "Change on previous",
                "value": formatting.format_signed_percent(relative),
                "note": (
                    f"against {periods['period_label'].iloc[-2]}"
                ),
            })

        else:
            kpis.append({
                "label": "Change on previous",
                "value": formatting.format_signed(
                    latest - previous
                    if pd.notna(previous)
                    else None
                ),
                "note": "previous period unavailable",
            })

    else:
        kpis.append({
            "label": "Change on previous",
            "value": formatting.MISSING_TEXT,
            "note": "only one period available",
        })

    kpis.append({
        "label": f"Average per {formatting.describe_grain(grain)[:-2]}",
        "value": formatting.format_compact(average),
        "note": (
            timeseries.describe_trend(trend, grain)
            if trend.get("success")
            else "no trend could be fitted"
        ),
    })

    return kpis


def recommend(
    evidence: list[dict[str, Any]],
    backtest: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """
    Turn evidence into a ranked list of things to look at.

    Every entry names the evidence it rests on and is phrased as an
    investigation. A table of totals cannot establish why a number moved,
    so nothing here claims to.

    Args:
        evidence:
            Cards from the evidence layer.

        backtest:
            Output of forecasting.backtest, when a projection was made.

    Returns:
        Recommendations, most important first.
    """

    by_id = {card["id"]: card for card in evidence}
    recommendations: list[dict[str, Any]] = []

    anomaly_card = by_id.get("anomalies")

    if anomaly_card and anomaly_card["detail"].get("anomaly_count"):
        periods = anomaly_card["detail"]["periods"]

        recommendations.append({
            "title": "Find out what happened in the flagged periods",
            "action": (
                f"Look at {', '.join(periods[:3])} in the source system. "
                "These periods sit further from the trend than the "
                "normal spread explains."
            ),
            "why": (
                "An unusual period is either a real event worth "
                "understanding or a data problem worth fixing. Both are "
                "worth knowing about."
            ),
            "rests_on": ["anomalies"],
            "severity": "act",
        })

    movement_card = by_id.get("movement")

    if movement_card and movement_card["severity"] == "act":
        detail = movement_card["detail"]

        recommendations.append({
            "title": (
                f"Check the fall in {detail['latest_label']}"
            ),
            "action": (
                "Break the latest period down by segment to see whether "
                "the fall is broad or concentrated in one part."
            ),
            "why": (
                "A drop that comes from one segment points somewhere "
                "specific. A drop spread across all of them points at "
                "something shared, such as pricing, seasonality or "
                "collection."
            ),
            "rests_on": ["movement"],
            "severity": "act",
        })

    concentration_card = by_id.get("concentration")

    if concentration_card and concentration_card["severity"] in {
        "act",
        "watch",
    }:
        recommendations.append({
            "title": "Review how much rests on one part",
            "action": (
                concentration_card["detail"].get("action_hint")
                or "Look at whether this concentration is intended."
            ),
            "why": (
                "When most of the total comes from one part, the overall "
                "figure mostly reports that part's behaviour and hides "
                "everything else."
            ),
            "rests_on": ["concentration"],
            "severity": concentration_card["severity"],
        })

    trend_card = by_id.get("trend")

    if (
        trend_card
        and trend_card["detail"].get("direction") == "falling"
        and trend_card["detail"].get("significant")
    ):
        recommendations.append({
            "title": "Treat the decline as a pattern, not a blip",
            "action": (
                "Compare the earliest and latest thirds of the period to "
                "see whether the fall is steady or started at a point in "
                "time."
            ),
            "why": (
                "The slope is large enough relative to the noise that it "
                "is unlikely to be chance."
            ),
            "rests_on": ["trend"],
            "severity": "watch",
        })

    for quality_id in ("measure_gaps", "duplicates"):
        card = by_id.get(quality_id)

        if card and card["severity"] == "act":
            recommendations.append({
                "title": f"Fix the data first: {card['title'].lower()}",
                "action": (
                    "Correct this at source before acting on any total "
                    "above, because every figure here is computed from "
                    "the same rows."
                ),
                "why": card["finding"],
                "rests_on": [quality_id],
                "severity": "act",
            })

    if backtest and backtest.get("success") and not backtest["beat_naive"]:
        recommendations.append({
            "title": "Do not plan against the projection",
            "action": (
                "Use the last observed period as the working assumption. "
                "The projection was tested and did not beat that."
            ),
            "why": backtest["verdict"],
            "rests_on": [],
            "severity": "watch",
        })

    recommendations.sort(
        key=lambda entry: SEVERITY_ORDER.get(entry["severity"], 3)
    )

    for position, entry in enumerate(recommendations, start=1):
        entry["priority"] = position

    return recommendations


def headline(
    evidence: list[dict[str, Any]],
    measure_label: str
) -> str:
    """
    Write the one sentence a reader would take away.

    Args:
        evidence:
            Cards from the evidence layer.

        measure_label:
            Name of the measure.

    Returns:
        A sentence.
    """

    by_id = {card["id"]: card for card in evidence}

    movement = by_id.get("movement")
    trend = by_id.get("trend")
    anomaly = by_id.get("anomalies")

    if movement:
        sentence = movement["finding"]

    elif trend:
        sentence = trend["finding"]

    else:
        return f"{measure_label} was summarised over the loaded rows."

    if anomaly and anomaly["detail"].get("anomaly_count"):
        sentence += (
            f" {anomaly['detail']['anomaly_count']} period(s) also fell "
            "outside the expected band."
        )

    return sentence


def executive_brief(
    analysis: dict[str, Any]
) -> str:
    """
    Arrange the analysis into a Markdown brief.

    Args:
        analysis:
            Output of analyse_business.

    Returns:
        Markdown text, ready to download.
    """

    lines = [
        "# Executive brief",
        "",
        f"**{analysis['headline']}**",
        "",
        f"_{analysis['schema_summary']}_",
        "",
        "## Headline figures",
        "",
    ]

    for kpi in analysis["kpis"]:
        lines.append(
            f"- **{kpi['label']}**: {kpi['value']} "
            f"({kpi['note']})"
        )

    lines += ["", "## Evidence", ""]

    for card in analysis["evidence"]:
        lines += [
            f"### {card['title']}",
            "",
            card["finding"],
            "",
            f"_Calculation: {card['calculation']}_",
            "",
        ]

    if analysis["recommendations"]:
        lines += ["## What to look at next", ""]

        for entry in analysis["recommendations"]:
            lines += [
                f"{entry['priority']}. **{entry['title']}**",
                f"   - {entry['action']}",
                f"   - Why: {entry['why']}",
                "",
            ]

        lines += [
            (
                "_These are investigations suggested by the numbers, not "
                "established causes._"
            ),
            "",
        ]

    if analysis["notes"]:
        lines += ["## Adjustments made to the timeline", ""]

        for note in analysis["notes"]:
            lines.append(f"- {note}")

        lines.append("")

    return "\n".join(lines)


def analyse_business(
    frame: pd.DataFrame,
    measure: str | None,
    date: str,
    segment: str | None = None,
    aggregation_name: str = "sum",
    grain: str | None = None,
    horizon: int = forecasting.DEFAULT_HORIZON,
    schema_summary: str = ""
) -> dict[str, Any]:
    """
    Run the whole deterministic analysis for a dataset.

    Args:
        frame:
            The dataset.

        measure:
            Column being measured, or None to count rows.

        date:
            Column carrying the timeline.

        segment:
            Column splitting the business, when there is one.

        aggregation_name:
            How to combine rows within a period.

        grain:
            Period grain, detected when omitted.

        horizon:
            Periods to project.

        schema_summary:
            Sentence describing the detected roles, for the brief.

    Returns:
        The period series, every analysis, the KPIs, the evidence ledger,
        the recommendations, the headline and the Markdown brief.
    """

    aggregated = aggregation.aggregate_periods(
        frame,
        date_column=date,
        measure_column=measure,
        aggregation=aggregation_name,
        grain=grain
    )

    if not aggregated.get("success"):
        return {
            "success": False,
            "error": aggregated.get(
                "error",
                "The timeline could not be built."
            ),
        }

    periods = aggregated["periods"]
    resolved_grain = aggregated["grain"]
    measure_label = measure or "Rows"

    trend = timeseries.fit_trend(
        periods["period_start"],
        periods["value"]
    )

    detected = anomaly_detection.detect_anomalies(
        periods,
        grain=resolved_grain,
        trend=trend if trend.get("success") else None
    )

    projection = forecasting.baseline_forecast(
        periods,
        grain=resolved_grain,
        horizon=horizon
    )

    verification = forecasting.backtest(
        periods,
        grain=resolved_grain,
        horizon=horizon
    )

    evidence = [coverage_evidence(aggregated)]

    for card in (
        movement_evidence(aggregated, measure_label),
        trend_evidence(trend, resolved_grain, measure_label),
        anomaly_evidence(detected),
    ):
        if card:
            evidence.append(card)

    if segment:
        segment_cards = segment_evidence(
            frame,
            measure=measure,
            date=date,
            segment=segment,
            periods=periods,
            aggregation_name=aggregation_name,
            grain=resolved_grain,
            measure_label=measure_label
        )

        evidence.extend(segment_cards["cards"])

    else:
        segment_cards = {
            "cards": [],
            "breakdown": None,
            "movement": None,
            "matrix": None,
            "error": "",
        }

    evidence.extend(quality_evidence(frame, measure))

    relationship = relationship_evidence(frame, measure)

    if relationship:
        evidence.append(relationship)

    evidence.sort(
        key=lambda card: SEVERITY_ORDER.get(card["severity"], 3)
    )

    kpis = build_kpis(aggregated, trend, measure_label)
    recommendations = recommend(evidence, verification)

    analysis = {
        "success": True,
        "aggregated": aggregated,
        "periods": periods,
        "grain": resolved_grain,
        "trend": trend,
        "anomalies": detected,
        "forecast": projection,
        "backtest": verification,
        "segment_column": segment,
        "segment_breakdown": segment_cards["breakdown"],
        "segment_movement": segment_cards["movement"],
        "segment_matrix": segment_cards["matrix"],
        "segment_error": segment_cards["error"],
        "kpis": kpis,
        "evidence": evidence,
        "recommendations": recommendations,
        "measure_label": measure_label,
        "notes": aggregated["notes"],
        "schema_summary": schema_summary,
    }

    analysis["headline"] = headline(evidence, measure_label)
    analysis["brief"] = executive_brief(analysis)

    return analysis
