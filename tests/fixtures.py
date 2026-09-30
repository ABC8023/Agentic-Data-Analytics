"""
Shared fixtures.

Datasets are built in code rather than read from disk so a test states
the shape it depends on, and so the suite runs with no data files
present.
"""

import numpy as np
import pandas as pd

# Distinctive values, so a privacy leak in a payload is unmistakable
# rather than something that has to be inferred.
SECRET_NAMES = [
    "Ingrid Halvorsen",
    "Tobias Mwangi",
    "Priya Raghunathan",
]

SECRET_SALARIES = [147311.0, 98214.0, 121977.0]


def classification_frame(rows: int = 200) -> pd.DataFrame:
    """
    Build a table with a clean categorical target.

    Args:
        rows:
            Number of rows to generate.

    Returns:
        A frame whose "churned" column is a two-class target.
    """

    rng = np.random.default_rng(11)

    return pd.DataFrame({
        "amount": rng.normal(180, 40, rows).round(2),
        "tickets": rng.integers(0, 9, rows),
        "plan": rng.choice(["basic", "plus", "premium"], rows),
        "region": rng.choice(["north", "south", "east", "west"], rows),
        "churned": rng.choice(["yes", "no"], rows),
    })


def regression_frame(rows: int = 200) -> pd.DataFrame:
    """
    Build a table with a continuous target carrying a known linear signal.

    Args:
        rows:
            Number of rows to generate.

    Returns:
        A frame whose "target" column depends mostly on "x1" and "x2".
    """

    rng = np.random.default_rng(12)

    frame = pd.DataFrame({
        "x1": rng.normal(0, 1, rows),
        "x2": rng.normal(5, 2, rows),
        "group": rng.choice(["a", "b"], rows),
    })

    frame["target"] = (
        3 * frame["x1"]
        - 2 * frame["x2"]
        + rng.normal(0, 0.5, rows)
    )

    return frame


def messy_frame(rows: int = 120) -> pd.DataFrame:
    """
    Build a table with the problems a real extract has.

    Contains an identifier column, a date column held as text, a constant
    column, missing values and duplicate rows.

    Args:
        rows:
            Number of unique rows before duplicates are appended.

    Returns:
        The messy frame.
    """

    rng = np.random.default_rng(13)

    frame = pd.DataFrame({
        "customer_id": [f"CUST-{index:05d}" for index in range(rows)],
        "signup_date": pd.date_range(
            "2024-01-01",
            periods=rows
        ).astype(str),
        "monthly_spend": rng.normal(180, 45, rows).round(2),
        "tickets": rng.integers(0, 9, rows),
        "plan": rng.choice(["basic", "plus", "premium"], rows),
        "source": "crm_export",
        "churned": rng.choice(["yes", "no"], rows),
    })

    frame.loc[0:9, "monthly_spend"] = np.nan
    frame.loc[5:12, "plan"] = None

    return pd.concat(
        [frame, frame.iloc[:8]],
        ignore_index=True
    )


def timeline_frame(
    days: int = 730,
    spike_day: int | None = 400
) -> pd.DataFrame:
    """
    Build a dated table long enough for a trend, a band and a forecast.

    Two years of daily rows aggregate to a monthly grain, which is the
    only grain that gets a seasonal adjustment, and gives enough periods
    for anomaly detection.

    Args:
        days:
            Number of daily rows.

        spike_day:
            Row to inflate, so there is something for the band to find.
            Pass None for a clean series.

    Returns:
        The frame.
    """

    rng = np.random.default_rng(15)

    frame = pd.DataFrame({
        "order_date": pd.date_range("2023-01-01", periods=days, freq="D"),
        "revenue": rng.normal(500, 40, days).round(2),
        "units": rng.integers(1, 20, days),
        "region": rng.choice(["north", "south", "east"], days),
    })

    if spike_day is not None and spike_day < days:
        frame.loc[spike_day, "revenue"] = 9000.0

    return frame


def personal_data_frame(rows: int = 20) -> pd.DataFrame:
    """
    Build a table mixing personal data with ordinary labels.

    Used by the privacy contract test. The personal columns are low
    cardinality on purpose, so the test proves the column-name heuristic
    is doing work that cardinality alone cannot.

    Args:
        rows:
            Number of rows to generate.

    Returns:
        The frame.
    """

    rng = np.random.default_rng(14)

    return pd.DataFrame({
        "employee": (SECRET_NAMES * 7)[:rows],
        "salary": (SECRET_SALARIES * 7)[:rows],
        "reference": [f"REF-{index:05d}" for index in range(rows)],
        "region": rng.choice(["emea", "apac"], rows),
        "band": rng.choice(["L3", "L4"], rows),
        "product_name": rng.choice(
            ["Atlas", "Borealis", "Cirrus"],
            rows
        ),
        "tenure_years": rng.normal(5, 2, rows).round(1),
    })
