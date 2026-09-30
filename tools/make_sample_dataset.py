"""
Build a sample dataset with known planted facts.

A demonstration dataset is only useful if the answers can be checked. This
one is generated from a fixed seed and carries deliberate features, and the
script prints the figures it planted so a reader can compare them with what
the application reports.

    python tools/make_sample_dataset.py

Writes sample_data/store_orders.csv and sample_data/store_orders_facts.txt.
"""

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

OUTPUT_DIRECTORY = os.path.join(REPO, "sample_data")
DATA_PATH = os.path.join(OUTPUT_DIRECTORY, "store_orders.csv")
FACTS_PATH = os.path.join(OUTPUT_DIRECTORY, "store_orders_facts.txt")

# One part carrying half the business, so the concentration card has
# something true to report. Four parts means an even split is 25%, and
# north at 50% is twice that: enough to be called out.
REGION_WEIGHTS = {
    "north": 0.50,
    "south": 0.25,
    "east": 0.15,
    "west": 0.10,
}

CHANNELS = ("retail", "wholesale", "online")

# A month lifted well clear of the trend, so the anomaly band has a real
# event to find rather than noise.
ANOMALY_MONTH = "2024-03"
ANOMALY_MULTIPLIER = 2.6

# Gentle growth, applied per day, so the trendline has a direction that is
# genuinely distinguishable from flat.
DAILY_GROWTH = 0.0009

FIRST_DAY = "2023-01-01"
LAST_DAY = "2025-06-30"

# Deliberate mess, so the evidence ledger and the data-quality report have
# something real to report.
BLANK_REGION_ROWS = 40
UNREADABLE_REVENUE_ROWS = 25
DUPLICATE_ROWS = 12


def build() -> pd.DataFrame:
    """Generate the dataset."""

    rng = np.random.default_rng(20260827)
    days = pd.date_range(FIRST_DAY, LAST_DAY, freq="D")

    rows = []
    reference = 0

    for offset, day in enumerate(days):
        growth = (1.0 + DAILY_GROWTH) ** offset

        # A mild yearly shape, so a forecast has seasonality to find.
        seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * day.dayofyear / 365.0)

        spike = (
            ANOMALY_MULTIPLIER
            if day.strftime("%Y-%m") == ANOMALY_MONTH
            else 1.0
        )

        for region, weight in REGION_WEIGHTS.items():
            wobble = 1.0 + rng.normal(0.0, 0.06)
            revenue = 1000.0 * weight * growth * seasonal * spike * wobble

            rows.append({
                "order_reference": f"ORD-{reference:06d}",
                "order_date": day.strftime("%Y-%m-%d"),
                "region": region,
                "channel": CHANNELS[reference % len(CHANNELS)],
                "units": int(max(1, round(revenue / 25.0))),
                "revenue": round(max(0.0, revenue), 2),
                "source_system": "orders_export",
            })

            reference += 1

    frame = pd.DataFrame(rows)

    # Mess, applied at fixed positions so the facts below stay true.
    frame.loc[100:100 + BLANK_REGION_ROWS - 1, "region"] = None

    frame["revenue"] = frame["revenue"].astype(object)
    frame.loc[
        500:500 + UNREADABLE_REVENUE_ROWS - 1,
        "revenue"
    ] = "not recorded"

    frame = pd.concat(
        [frame, frame.iloc[2000:2000 + DUPLICATE_ROWS]],
        ignore_index=True
    )

    return frame


def facts(frame: pd.DataFrame) -> list[str]:
    """
    Work out the figures a reader can check the application against.

    Args:
        frame:
            The generated dataset.

    Returns:
        Lines describing what was planted, with the arithmetic behind each.
    """

    revenue = pd.to_numeric(frame["revenue"], errors="coerce")
    dates = pd.to_datetime(frame["order_date"])

    usable = revenue.notna()
    total = float(revenue[usable].sum())

    by_region = revenue[usable].groupby(
        frame.loc[usable, "region"].fillna("(not set)")
    ).sum().sort_values(ascending=False)

    monthly = revenue[usable].groupby(
        dates[usable].dt.to_period("M")
    ).sum()

    # The newest month in the data is complete, because the range ends on
    # the last day of June.
    last_month = monthly.index[-1]
    previous_month = monthly.index[-2]

    unreadable = int((~usable).sum())
    blank_regions = int(frame["region"].isna().sum())
    duplicates = int(frame.duplicated().sum())

    lines = [
        "Known facts about sample_data/store_orders.csv",
        "=" * 46,
        "",
        f"Rows in the file: {len(frame):,}",
        (
            f"Rows with an unreadable revenue: {unreadable:,} "
            "(written as 'not recorded')"
        ),
        f"Rows with a blank region: {blank_regions:,}",
        f"Exact duplicate rows: {duplicates:,}",
        "A column that never changes: source_system",
        "An identifier column: order_reference",
        "",
        f"Date range: {dates.min().date()} to {dates.max().date()}",
        f"Complete months of data: {len(monthly)}",
        "",
        "-- Totals you can check --",
        "",
        f"Total revenue, whole file: {total:,.2f}",
        "",
        "Total revenue by region, largest first:",
    ]

    for name, value in by_region.items():
        lines.append(
            f"  {name}: {value:,.2f} ({value / total:.1%} of the total)"
        )

    highest = monthly.idxmax().strftime("%b %Y")
    lowest = monthly.idxmin().strftime("%b %Y")
    latest_label = last_month.strftime("%b %Y")
    previous_label = previous_month.strftime("%b %Y")
    change = monthly.iloc[-1] - monthly.iloc[-2]
    relative = monthly.iloc[-1] / monthly.iloc[-2] - 1

    lines += [
        "",
        (
            f"  The highest month is {highest} at {monthly.max():,.2f}, "
            "which is the planted anomaly."
        ),
        f"  The lowest month is {lowest} at {monthly.min():,.2f}.",
        "",
        f"Latest complete month ({latest_label}): {monthly.iloc[-1]:,.2f}",
        f"Month before ({previous_label}): {monthly.iloc[-2]:,.2f}",
        f"Change: {change:+,.2f} ({relative:+.1%})",
        "",
        "-- What the application should say --",
        "",
        "Trend: rising, and distinguishable from flat.",
        (
            f"Unusual period: {ANOMALY_MONTH} should be flagged, because "
            f"it was multiplied by {ANOMALY_MULTIPLIER}."
        ),
        (
            "Concentration: north is the largest part at roughly 50%, "
            "which is well clear of an even split, so it is called out."
        ),
        (
            "Duplicates and blank regions should both appear as "
            "data-quality findings."
        ),
    ]

    return lines


def main() -> int:
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    frame = build()
    frame.to_csv(DATA_PATH, index=False)

    lines = facts(frame)

    with open(FACTS_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print(f"Wrote {DATA_PATH}")
    print(f"Wrote {FACTS_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
