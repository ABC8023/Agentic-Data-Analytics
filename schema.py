"""
Column schema description, and what the columns mean.

Two jobs live here. The first is a structural description of every
column, which is what the language model is given instead of a row
preview: enough to reason about a dataset with no cell values in it.

The second is business role detection. A dashboard needs to know which
column is the thing being measured, which one carries time, and which one
splits the business into parts. Guessing that from column names alone is
unreliable, and guessing from column shape alone is worse, so both are
used and the reasoning is reported so a reader can overrule it.

Detection can fail, and saying so is part of the job. A parts catalogue
has no revenue and no timeline, and a dashboard that renders empty cards
for it is less useful than one that explains why there is nothing to show.
"""

from typing import Any

import pandas as pd

import aggregation

# Share of a column's filled-in values that must read as numbers before it
# counts as something measurable. Held here so the schema, the question
# parser and the plan validator all use one rule.
NUMERIC_COERCION_RATE = 0.5

# A text column whose values are mostly distinct is an identifier rather
# than something to group by or predict.
IDENTIFIER_UNIQUE_RATIO = 0.5

# Minimum distinct values before the ratio rule above is trusted. Below
# this a small table would flag ordinary labels.
IDENTIFIER_MIN_DISTINCT = 20

# A column where essentially every value differs is an identifier at any
# size, so a short extract of reference codes is still recognised.
NEARLY_UNIQUE_RATIO = 0.95
NEARLY_UNIQUE_MIN_DISTINCT = 10

# Above this many distinct categories a column is too wide to one-hot
# encode or to chart as a bar.
HIGH_CARDINALITY_LEVELS = 50

# Column names that suggest the contents identify a person or an account.
#
# Cardinality alone cannot separate personal data from an ordinary label:
# in a small extract, three employee names repeat exactly as often as
# three region names would. The column name is the only other signal
# available, so it is used as a second, independent test.
#
# Matching is by substring against a separator-free version of the name,
# so "customerName" and "customer_name" both match "customer". Deliberately
# excluded are broad words like "name" and "contact" on their own, because
# "product_name" and "contact_channel" are ordinary labels whose values are
# useful and harmless to summarise. This is a heuristic and can be wrong in
# both directions.
IDENTITY_NAME_HINTS = (
    "email",
    "phone",
    "mobile",
    "telephone",
    "address",
    "postcode",
    "postalcode",
    "zipcode",
    "passport",
    "ssn",
    "socialsecurity",
    "iban",
    "bankaccount",
    "creditcard",
    "username",
    "surname",
    "fullname",
    "firstname",
    "lastname",
    "employee",
    "customer",
    "client",
    "patient",
    "dateofbirth",
    "dob",
)


def normalise_column_name(name: Any) -> str:
    """
    Reduce a column name to lowercase letters and digits.

    Args:
        name:
            Column name to normalise.

    Returns:
        The name with separators and punctuation removed, for example
        "Customer_Name" becomes "customername".
    """

    text = str(name).lower()

    return "".join(
        character
        for character in text
        if character.isalnum()
    )


def numeric_share(series: pd.Series) -> float:
    """
    Share of a column's filled-in values that read as numbers.

    Measured against the values that are present rather than every row, so
    a sparsely populated money column is not judged unmeasurable for being
    mostly blank.

    Args:
        series:
            The column.

    Returns:
        A proportion between zero and one.
    """

    present = series.dropna()

    if present.empty:
        return 0.0

    return float(
        pd.to_numeric(present, errors="coerce").notna().mean()
    )


def is_measurable(series: pd.Series) -> bool:
    """
    Report whether a column can be totalled, however it happens to be
    stored.

    Storage type is the wrong test. A real revenue column with a handful of
    cells reading "not recorded" arrives as text, and gating on the dtype
    excludes it from being a measure at all. The application then answers
    questions about revenue by quietly totalling some other column, which
    is the worst available outcome: a plausible figure for the wrong thing.

    Args:
        series:
            The column.

    Returns:
        True when most of the filled-in values read as numbers.
    """

    if pd.api.types.is_bool_dtype(series):
        return False

    if pd.api.types.is_datetime64_any_dtype(series):
        return False

    if pd.api.types.is_numeric_dtype(series):
        return True

    return numeric_share(series) >= NUMERIC_COERCION_RATE


def as_numbers(series: pd.Series) -> pd.Series:
    """
    Read a column as numbers, keeping its name.

    Args:
        series:
            The column.

    Returns:
        The column coerced to numbers, with anything unreadable as empty.
    """

    if pd.api.types.is_numeric_dtype(series):
        return series

    converted = pd.to_numeric(series, errors="coerce")
    converted.name = series.name

    return converted


def name_suggests_identity(name: Any) -> bool:
    """
    Report whether a column name suggests personal or account data.

    Args:
        name:
            Column name to test.

    Returns:
        True when the name contains one of the identity hints.
    """

    normalised = normalise_column_name(name)

    return any(
        hint in normalised
        for hint in IDENTITY_NAME_HINTS
    )


def column_kind(series: pd.Series) -> str:
    """
    Classify a column into a coarse kind.

    Args:
        series:
            The column to classify.

    Returns:
        One of "numeric", "datetime", "boolean" or "categorical".
    """

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    return "categorical"


def describe_column(
    series: pd.Series,
    row_count: int
) -> dict[str, Any]:
    """
    Describe one column without exposing any of its values.

    Args:
        series:
            The column to describe.

        row_count:
            Number of rows in the dataset, used for ratios.

    Returns:
        Name, kind, dtype, null and distinct counts, and flags for
        constant, identifier-like and high-cardinality columns.
    """

    non_missing = series.dropna()

    distinct = int(non_missing.nunique())

    unique_ratio = (
        distinct / len(non_missing)
        if len(non_missing)
        else 0.0
    )

    kind = column_kind(series)

    # Identity is a question about labels, not about measurements. A
    # continuous numeric column is nearly all distinct by nature, so
    # applying this test to numbers would label every measure an
    # identifier. Numeric spread is reported separately below, where
    # Phase 3 measure detection can use it.
    looks_like_identifier = (
        kind in {"categorical", "boolean"}
        and (
            (
                distinct > IDENTIFIER_MIN_DISTINCT
                and unique_ratio >= IDENTIFIER_UNIQUE_RATIO
            )
            or (
                distinct >= NEARLY_UNIQUE_MIN_DISTINCT
                and unique_ratio >= NEARLY_UNIQUE_RATIO
            )
        )
    )

    is_probably_continuous = (
        kind == "numeric"
        and distinct > IDENTIFIER_MIN_DISTINCT
        and unique_ratio >= IDENTIFIER_UNIQUE_RATIO
    )

    return {
        "name": str(series.name),
        "kind": kind,
        "dtype": str(series.dtype),
        "missing": int(series.isnull().sum()),
        "missing_share": round(
            float(series.isnull().sum() / row_count)
            if row_count
            else 0.0,
            4
        ),
        "distinct": distinct,
        "unique_ratio": round(float(unique_ratio), 4),
        "is_constant": distinct <= 1,
        "looks_like_identifier": bool(looks_like_identifier),
        "is_probably_continuous": bool(is_probably_continuous),
        "is_high_cardinality": distinct > HIGH_CARDINALITY_LEVELS,
        "name_suggests_identity": name_suggests_identity(series.name),
    }


def column_schema(dataframe: pd.DataFrame) -> dict[str, Any]:
    """
    Describe a whole dataset structurally, with no cell values.

    This is the payload handed to the model in place of a row preview.

    Args:
        dataframe:
            Dataset to describe.

    Returns:
        Row and column counts, a per-column description, and the column
        names grouped by kind.
    """

    row_count = int(len(dataframe))

    columns = [
        describe_column(dataframe[name], row_count)
        for name in dataframe.columns
    ]

    def names_of_kind(kind: str) -> list[str]:
        return [
            column["name"]
            for column in columns
            if column["kind"] == kind
        ]

    return {
        "rows": row_count,
        "columns": int(dataframe.shape[1]),
        "column_names": [str(name) for name in dataframe.columns],
        "numeric_columns": names_of_kind("numeric"),
        "categorical_columns": names_of_kind("categorical"),
        "datetime_columns": names_of_kind("datetime"),
        "boolean_columns": names_of_kind("boolean"),
        "identifier_like_columns": [
            column["name"]
            for column in columns
            if column["looks_like_identifier"]
        ],
        "constant_columns": [
            column["name"]
            for column in columns
            if column["is_constant"]
        ],
        "schema": columns,
    }


# ──────────────────────────────────────────────────────────────
#  BUSINESS ROLE DETECTION
# ──────────────────────────────────────────────────────────────

# Names that suggest a column is the thing being measured. Ordered, so a
# column called "revenue" is preferred over one called "units" when both
# are present: money is more often the headline than volume.
MEASURE_NAME_HINTS = (
    "revenue",
    "sales",
    "turnover",
    "gmv",
    "grossmargin",
    "margin",
    "profit",
    "income",
    "earnings",
    "amount",
    "total",
    "subtotal",
    "spend",
    "cost",
    "price",
    "value",
    "balance",
    "units",
    "quantity",
    "qty",
    "volume",
    "orders",
    "sessions",
    "clicks",
    "impressions",
    "visits",
    "duration",
    "weight",
)

# Names that suggest a column splits the business into comparable parts.
SEGMENT_NAME_HINTS = (
    "segment",
    "category",
    "subcategory",
    "product",
    "sku",
    "channel",
    "region",
    "country",
    "market",
    "territory",
    "department",
    "division",
    "team",
    "store",
    "branch",
    "outlet",
    "brand",
    "industry",
    "sector",
    "plan",
    "tier",
    "status",
    "stage",
    "source",
    "campaign",
    "platform",
    "device",
    "type",
    "group",
    "class",
    "city",
    "state",
    "province",
)

# Names that suggest a column carries the timeline.
DATE_NAME_HINTS = (
    "date",
    "day",
    "week",
    "month",
    "quarter",
    "period",
    "created",
    "ordered",
    "booked",
    "invoiced",
    "posted",
    "shipped",
    "signup",
    "timestamp",
    "datetime",
    "time",
)

# A column of four digit integers in this range is a calendar year, which
# is a period label rather than something to total.
YEAR_LOWER_BOUND = 1900
YEAR_UPPER_BOUND = 2100

# A segment with more parts than this is too wide to read as a breakdown.
MAX_SEGMENT_LEVELS = 50

# The range of segment sizes that produces a readable breakdown. Two parts
# is thin, forty is a wall of bars.
COMFORTABLE_SEGMENT_LEVELS = (2, 12)

# Low cardinality integer columns can be segments, such as a store number.
MAX_NUMERIC_SEGMENT_LEVELS = 12


def name_hint_rank(name: Any, hints: tuple[str, ...]) -> int | None:
    """
    Find where a column name matches a list of naming hints.

    Args:
        name:
            Column name.

        hints:
            Hints in preference order.

    Returns:
        The index of the first matching hint, or None when none match. A
        lower index means a stronger preference.
    """

    normalised = normalise_column_name(name)

    for index, hint in enumerate(hints):
        if hint in normalised:
            return index

    return None


def looks_like_year(series: pd.Series) -> bool:
    """
    Report whether a numeric column is really a calendar year.

    Totalling a column of years produces a number with no meaning, so a
    year has to be kept out of the measure candidates.

    Args:
        series:
            The column to test.

    Returns:
        True when every value is a whole number inside the plausible year
        range and there are few enough distinct values to be years.
    """

    if not pd.api.types.is_numeric_dtype(series):
        return False

    if pd.api.types.is_bool_dtype(series):
        return False

    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return False

    if not (values % 1 == 0).all():
        return False

    if values.min() < YEAR_LOWER_BOUND or values.max() > YEAR_UPPER_BOUND:
        return False

    return int(values.nunique()) <= 200


def measure_candidates(dataframe: pd.DataFrame) -> list[str]:
    """
    Rank the columns that could be the thing being measured.

    Args:
        dataframe:
            The dataset.

    Returns:
        Column names, best first.
    """

    row_count = int(len(dataframe))
    scored = []

    for position, name in enumerate(dataframe.columns):
        series = dataframe[name]

        if not is_measurable(series):
            continue

        # Described as numbers, so a money column stored as text is judged
        # on its figures rather than on its storage.
        series = as_numbers(series)
        description = describe_column(series, row_count)

        if description["is_constant"]:
            continue

        if description["name_suggests_identity"]:
            continue

        if looks_like_year(series):
            continue

        hint = name_hint_rank(name, MEASURE_NAME_HINTS)

        scored.append((
            0 if hint is not None else 1,
            hint if hint is not None else 0,
            0 if description["is_probably_continuous"] else 1,
            -description["distinct"],
            position,
            str(name),
        ))

    scored.sort()

    return [entry[-1] for entry in scored]


def date_candidates(dataframe: pd.DataFrame) -> list[str]:
    """
    Rank the columns that could carry the timeline.

    Args:
        dataframe:
            The dataset.

    Returns:
        Column names, best first.
    """

    usable = aggregation.usable_date_columns(dataframe)

    if not usable:
        return []

    positions = {
        str(name): index
        for index, name in enumerate(dataframe.columns)
    }

    scored = []

    for name in usable:
        hint = name_hint_rank(name, DATE_NAME_HINTS)
        distinct = int(dataframe[name].dropna().nunique())

        scored.append((
            0 if hint is not None else 1,
            hint if hint is not None else 0,
            -distinct,
            positions.get(name, 0),
            name,
        ))

    scored.sort()

    return [entry[-1] for entry in scored]


def segment_candidates(
    dataframe: pd.DataFrame,
    exclude: tuple[str, ...] = ()
) -> list[str]:
    """
    Rank the columns that could split the business into parts.

    Args:
        dataframe:
            The dataset.

        exclude:
            Columns already used for another role.

    Returns:
        Column names, best first.
    """

    row_count = int(len(dataframe))
    excluded = {str(name) for name in exclude}
    lowest, highest = COMFORTABLE_SEGMENT_LEVELS
    scored = []

    for position, name in enumerate(dataframe.columns):
        if str(name) in excluded:
            continue

        series = dataframe[name]
        description = describe_column(series, row_count)
        distinct = description["distinct"]

        if distinct < 2 or distinct > MAX_SEGMENT_LEVELS:
            continue

        if description["looks_like_identifier"]:
            continue

        if description["name_suggests_identity"]:
            continue

        if description["kind"] == "datetime":
            continue

        # A number can be a segment when there are few enough of them, as
        # with a store number, but not when it is a measurement.
        if description["kind"] == "numeric":
            if distinct > MAX_NUMERIC_SEGMENT_LEVELS:
                continue

            if name_hint_rank(name, MEASURE_NAME_HINTS) is not None:
                continue

        hint = name_hint_rank(name, SEGMENT_NAME_HINTS)

        comfortable = lowest <= distinct <= highest

        scored.append((
            0 if hint is not None else 1,
            hint if hint is not None else 0,
            0 if comfortable else 1,
            distinct,
            position,
            str(name),
        ))

    scored.sort()

    return [entry[-1] for entry in scored]


def _explain_choice(
    chosen: str | None,
    hints: tuple[str, ...],
    role: str,
    dataframe: pd.DataFrame
) -> str:
    """
    Say why a column was chosen for a role.

    Args:
        chosen:
            The selected column, or None.

        hints:
            The naming hints used for this role.

        role:
            Role name, for the sentence.

        dataframe:
            The dataset, for shape detail.

    Returns:
        A sentence explaining the choice.
    """

    if chosen is None:
        return f"No column looks like a {role}."

    if name_hint_rank(chosen, hints) is not None:
        return (
            f"'{chosen}' was chosen because its name reads like a "
            f"{role}."
        )

    distinct = int(dataframe[chosen].dropna().nunique())

    return (
        f"'{chosen}' was chosen from its shape: no column name looked "
        f"like a {role}, and this one has {distinct:,} distinct values "
        f"that fit the role."
    )


def detect_business_schema(
    dataframe: pd.DataFrame,
    measure: str | None = None,
    date: str | None = None,
    segment: str | None = None
) -> dict[str, Any]:
    """
    Work out which column is the measure, the date and the segment.

    Any of the three can be supplied to override detection, which is what
    the interface passes when a reader disagrees with the guess.

    Args:
        dataframe:
            The dataset.

        measure:
            Override for the measure column.

        date:
            Override for the date column.

        segment:
            Override for the segment column.

    Returns:
        The chosen roles, the reasoning, the ranked candidates for each,
        and flags describing what can and cannot be built from them.
    """

    measures = measure_candidates(dataframe)
    dates = date_candidates(dataframe)

    chosen_measure = (
        measure
        if measure in dataframe.columns
        else (measures[0] if measures else None)
    )
    chosen_date = (
        date
        if date in dataframe.columns
        else (dates[0] if dates else None)
    )

    segments = segment_candidates(
        dataframe,
        exclude=tuple(
            name
            for name in (chosen_measure, chosen_date)
            if name
        )
    )

    chosen_segment = (
        segment
        if segment in dataframe.columns
        else (segments[0] if segments else None)
    )

    row_count = int(len(dataframe))

    identifier_columns = [
        str(name)
        for name in dataframe.columns
        if describe_column(dataframe[name], row_count)[
            "looks_like_identifier"
        ]
    ]

    notes: list[str] = []

    if chosen_measure is None:
        notes.append(
            "No column reads as a measure, so there is no total to "
            "track. This happens with reference tables and catalogues, "
            "where every column is a label or an identifier. Counting "
            "rows is still available."
        )

    if chosen_date is None:
        notes.append(
            "No column reads as a date, so movement over time, unusual "
            "periods and projections are all unavailable."
        )

    if chosen_segment is None:
        notes.append(
            "No column reads as a segment, so the totals cannot be "
            "broken down. A segment needs between 2 and "
            f"{MAX_SEGMENT_LEVELS} repeated values."
        )

    return {
        "measure": chosen_measure,
        "measure_reason": _explain_choice(
            chosen_measure,
            MEASURE_NAME_HINTS,
            "measure",
            dataframe
        ),
        "measure_candidates": measures,
        "date": chosen_date,
        "date_reason": _explain_choice(
            chosen_date,
            DATE_NAME_HINTS,
            "date",
            dataframe
        ),
        "date_candidates": dates,
        "segment": chosen_segment,
        "segment_reason": _explain_choice(
            chosen_segment,
            SEGMENT_NAME_HINTS,
            "segment",
            dataframe
        ),
        "segment_candidates": segments,
        "identifier_columns": identifier_columns,
        "can_measure": chosen_measure is not None,
        "can_track_time": chosen_date is not None,
        "can_split": chosen_segment is not None,
        "can_build_dashboard": (
            chosen_measure is not None and chosen_date is not None
        ),
        "notes": notes,
    }


def describe_business_schema(detected: dict[str, Any]) -> str:
    """
    Summarise the detected roles in one sentence.

    Args:
        detected:
            Output of detect_business_schema.

    Returns:
        A sentence naming the roles that were found.
    """

    parts = []

    if detected["measure"]:
        parts.append(f"measuring '{detected['measure']}'")

    if detected["date"]:
        parts.append(f"over '{detected['date']}'")

    if detected["segment"]:
        parts.append(f"split by '{detected['segment']}'")

    if not parts:
        return (
            "No measure, date or segment could be identified in this "
            "dataset."
        )

    return "Reading this as " + ", ".join(parts) + "."
