"""
English into a plan, with no model involved.

Most questions a person asks of a table are simple, and sending "what was
total revenue last quarter" to a language model so it can decide the answer
is both slow and a category error. The words that matter are a small
closed set: an aggregation word, a period phrase, a column name, sometimes
a category value. Matching them is a parsing job.

The parser is built around consuming text. Every phrase it recognises is
marked as used, and whatever is left over is reported back as ignored.
That matters more than it sounds: a parser that quietly drops half a
question still produces a confident plan, and the reader has no way to see
that the answer is to a different question. Here the unused words are on
screen next to the answer.

Order of recognition is doing real work. Period phrases are consumed
before anything else, so "by month" is read as a monthly grain rather than
a breakdown by a column called month, and "last 3 months" is never
mistaken for a request for the top 3. Column names are matched before
category values, so a column called "region" is not confused with a value
inside it.

Where the question is genuinely ambiguous the parser refuses and says
why, rather than picking. Refusing hands the question to the model
fallback, which is the right escalation; guessing produces a number that
looks the same as a correct one.
"""

import calendar
import re
from dataclasses import dataclass, field

import pandas as pd

import aggregation
import nlq
import schema
import segments

# Phrases that decide the intent. Checked in the order given in
# choose_intent, because several of them can appear in one question.
COMPARISON_PHRASES = (
    "compared to",
    "compared with",
    "compare to",
    "compare with",
    "comparison with",
    " vs ",
    " vs. ",
    " versus ",
    "against last",
    "against the last",
    "against the previous",
    "than last",
    "than the last",
    "than the previous",
    "month on month",
    "year on year",
    "week on week",
    "quarter on quarter",
)

SHARE_PHRASES = (
    "share of",
    "proportion of",
    "percentage of",
    "percent of",
    "what fraction",
    "how much of",
    "what share",
)

TREND_PHRASES = (
    "growing",
    "growth",
    "declining",
    "decline",
    "shrinking",
    "trending",
    "trend",
    "direction",
    "going up",
    "going down",
    "increasing",
    "decreasing",
    "improving",
    "getting worse",
    "getting better",
)

OVER_TIME_PHRASES = (
    "over time",
    "by period",
    "time series",
    "timeline",
    "history",
)

QUANTITY_PHRASES = (
    "how much",
    "how many",
    "what is the",
    "what was the",
    "what's the",
    "give me",
    "tell me",
)

# Aggregation words. A phrase is matched before a shorter phrase it
# contains, so "average per day" does not first match "average".
AGGREGATION_PHRASES = (
    ("daily rate", "sum_per_day"),
    ("per day rate", "sum_per_day"),
    ("rate per day", "sum_per_day"),
    ("average per day", "sum_per_day"),
    ("number of", "count"),
    ("how many", "count"),
    ("row count", "count"),
    ("count of", "count"),
    ("count", "count"),
    ("total", "sum"),
    ("sum of", "sum"),
    ("sum", "sum"),
    ("combined", "sum"),
    ("altogether", "sum"),
    ("average", "mean"),
    ("avg", "mean"),
    ("mean", "mean"),
    ("typical", "median"),
    ("median", "median"),
    ("maximum", "max"),
    ("max", "max"),
    ("minimum", "min"),
    ("min", "min"),
)

# Words that pick a direction rather than an aggregation. "The highest
# month" asks which period was largest, which is a ranking; "the maximum
# revenue" asks for a figure, which is an aggregation. Keeping them apart
# is why "highest" does not become max.
DIRECTION_PHRASES = (
    ("highest", "highest"),
    ("largest", "highest"),
    ("biggest", "highest"),
    ("best", "highest"),
    ("most", "highest"),
    ("top", "highest"),
    ("peak", "highest"),
    ("lowest", "lowest"),
    ("smallest", "lowest"),
    ("worst", "lowest"),
    ("least", "lowest"),
    ("bottom", "lowest"),
)

GRAIN_WORDS = {
    "daily": "day",
    "day": "day",
    "days": "day",
    "weekly": "week",
    "week": "week",
    "weeks": "week",
    "monthly": "month",
    "month": "month",
    "months": "month",
    "quarterly": "quarter",
    "quarter": "quarter",
    "quarters": "quarter",
    "yearly": "year",
    "annually": "year",
    "annual": "year",
    "year": "year",
    "years": "year",
}

GRAIN_PATTERN = "|".join(sorted(GRAIN_WORDS, key=len, reverse=True))

MONTH_NUMBERS = {
    name.lower(): number
    for number, name in enumerate(calendar.month_name)
    if name
}

MONTH_NUMBERS.update({
    name.lower(): number
    for number, name in enumerate(calendar.month_abbr)
    if name
})

MONTH_PATTERN = "|".join(sorted(MONTH_NUMBERS, key=len, reverse=True))

# Numeric comparison phrases, longest first so "more than" is not read as
# a bare "than".
NUMERIC_PHRASES = (
    ("greater or equal to", "greater_or_equal"),
    ("greater than or equal to", "greater_or_equal"),
    ("at least", "greater_or_equal"),
    ("no less than", "greater_or_equal"),
    ("less than or equal to", "less_or_equal"),
    ("no more than", "less_or_equal"),
    ("at most", "less_or_equal"),
    ("greater than", "greater_than"),
    ("higher than", "greater_than"),
    ("more than", "greater_than"),
    ("exceeding", "greater_than"),
    ("over", "greater_than"),
    ("above", "greater_than"),
    ("less than", "less_than"),
    ("fewer than", "less_than"),
    ("lower than", "less_than"),
    ("under", "less_than"),
    ("below", "less_than"),
)

# Words never worth matching against a category value, because they are
# question scaffolding rather than data.
VALUE_STOP_WORDS = frozenset({
    "all",
    "and",
    "any",
    "average",
    "count",
    "for",
    "from",
    "how",
    "many",
    "much",
    "not",
    "row",
    "rows",
    "the",
    "total",
    "what",
    "which",
    "with",
    "yes",
})

# Words dropped from the ignored list, because reporting them as unused
# would be noise rather than information.
FILLER_WORDS = frozenset({
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "get",
    "give",
    "had",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "look",
    "looks",
    "make",
    "over",
    "many",
    "me",
    "much",
    "my",
    "of",
    "on",
    "our",
    "please",
    "show",
    "so",
    "tell",
    "that",
    "the",
    "then",
    "there",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "whats",
    "with",
    "you",
})

# Words that can be left unread without changing what is computed: the
# verbs and nouns people wrap a request for a figure in. Anything unread
# that is not listed here stops the rules from answering, because it may
# name a column, a period or a kind of question the rules did not see.
# Add to this list only with an evaluation case showing the word is safe.
HARMLESS_UNREAD_WORDS = frozenset({
    "all",
    "bring",
    "brings",
    "brought",
    "came",
    "come",
    "comes",
    "data",
    "dataset",
    "earn",
    "earned",
    "how",
    "made",
    "money",
    "order",
    "orders",
    "overall",
    "records",
    "rows",
    "sales",
    "total",
    "where",
    "which",
})

# In a comparison of two periods these only say which way the reader
# expects it to go. The comparison reports both directions either way.
HARMLESS_UNREAD_IN_A_COMPARISON = frozenset({
    "better",
    "bigger",
    "higher",
    "less",
    "lower",
    "more",
    "smaller",
    "worse",
})

# Phrases asking for a figure per period rather than one figure. Consumed
# before a split by column is looked for, so "by month" is a grain and not
# a breakdown by a column that happens to be called month.
GRAIN_PHRASES = tuple(
    (phrase, grain)
    for word, grain in (
        ("day", "day"),
        ("week", "week"),
        ("month", "month"),
        ("quarter", "quarter"),
        ("year", "year"),
    )
    for phrase in (
        f"by {word}",
        f"per {word}",
        f"each {word}",
        f"every {word}",
        f"{word} by {word}",
    )
) + (
    ("daily", "day"),
    ("weekly", "week"),
    ("monthly", "month"),
    ("quarterly", "quarter"),
    ("yearly", "year"),
    ("annually", "year"),
    ("annual", "year"),
)

# A bare four digit number is a year only inside a plausible range.
# Outside it the number is a quantity, and reading "between 1000 and 2000"
# as a pair of years would answer a question about money with a date
# filter.
EARLIEST_PLAUSIBLE_YEAR = 1900
LATEST_PLAUSIBLE_YEAR = 2100

# A category value shorter than this matches too much to be trusted.
MINIMUM_VALUE_LENGTH = 3

# Columns with more distinct values than this are not scanned for value
# matches, both for speed and because they are identifiers.
MAX_VALUES_PER_COLUMN = 60

# Total values scanned across all columns.
MAX_VALUES_SCANNED = 600

# Default number of parts a "which" question returns.
DEFAULT_WHICH_LIMIT = 1


class Scanner:
    """
    A string that remembers which characters have been used.

    Recognising a phrase consumes it, so a later rule cannot match the
    same words again, and whatever is never consumed is reported back to
    the reader as ignored.
    """

    def __init__(self, text: str):
        self.text = text
        self.used = [False] * len(text)

    def is_free(self, start: int, end: int) -> bool:
        """Report whether a span is still unconsumed."""

        return not any(self.used[start:end])

    def consume(self, start: int, end: int) -> None:
        """Mark a span as used."""

        for position in range(max(0, start), min(len(self.used), end)):
            self.used[position] = True

    def search(self, pattern: str) -> re.Match | None:
        """Find the first match that lies entirely in unused text."""

        for match in re.finditer(pattern, self.text):
            if self.is_free(match.start(), match.end()):
                return match

        return None

    def find_all(self, pattern: str) -> list[re.Match]:
        """Find every match lying entirely in unused text."""

        return [
            match
            for match in re.finditer(pattern, self.text)
            if self.is_free(match.start(), match.end())
        ]

    def take(self, pattern: str) -> re.Match | None:
        """Find the first unused match and consume it."""

        match = self.search(pattern)

        if match:
            self.consume(match.start(), match.end())

        return match

    def contains(self, phrase: str) -> int:
        """Return the position of an unused phrase, or -1."""

        start = 0

        while True:
            position = self.text.find(phrase, start)

            if position < 0:
                return -1

            if self.is_free(position, position + len(phrase)):
                return position

            start = position + 1

    def take_phrase(self, phrase: str) -> int:
        """Consume an unused phrase and return its position, or -1."""

        position = self.contains(phrase)

        if position >= 0:
            self.consume(position, position + len(phrase))

        return position

    def remaining_words(self) -> list[str]:
        """Words made only of characters nothing has claimed."""

        words = []

        for match in re.finditer(r"[a-z0-9_']+", self.text):
            if self.is_free(match.start(), match.end()):
                words.append(match.group())

        return words


@dataclass
class ParseResult:
    """
    What the parser made of a question.

    Attributes:
        plan:
            The plan, when one could be built and validated.

        understood:
            Whether the question produced an executable plan.

        reason:
            Why not, when it did not.

        matched:
            What the parser recognised, in reader-facing wording.

        ignored:
            Words the parser did not use. Shown to the reader, because a
            question half understood produces an answer to a different
            question.

        warnings:
            Things that change how the answer should be read.
    """

    plan: nlq.QueryPlan | None = None
    understood: bool = False
    reason: str = ""
    matched: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def normalise_question(question: str) -> str:
    """
    Reduce a question to lowercase text with tidy spacing.

    Args:
        question:
            What the reader typed.

    Returns:
        The question in lowercase, with punctuation that carries no
        meaning removed and single spaces between words. Spaces are added
        at both ends so a phrase pattern written with leading and trailing
        spaces, such as " vs ", matches at the edges too.
    """

    text = str(question).lower()
    text = text.replace("%", " percent ")
    text = re.sub(r"[^a-z0-9\-_/:.,']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return f" {text} "


def _column_variants(name: str) -> list[str]:
    """
    Ways a column name might be written in a sentence.

    Args:
        name:
            The column name.

    Returns:
        The lowercase name, and a version with separators as spaces, so
        both "order_date" and "order date" match.
    """

    lowered = str(name).lower()
    spaced = re.sub(r"[_\-./]+", " ", lowered)
    spaced = re.sub(r"\s+", " ", spaced).strip()

    variants = {lowered, spaced}

    return [
        variant
        for variant in sorted(variants, key=len, reverse=True)
        if variant
    ]


def _read_number(text: str) -> float | None:
    """Read a written number, tolerating thousands separators."""

    cleaned = text.replace(",", "").strip()

    try:
        return float(cleaned)

    except ValueError:
        return None


def _latest_year(frame: pd.DataFrame, date_column: str | None) -> int | None:
    """
    The most recent year in the dataset's timeline.

    Used to read a bare month name. "Revenue in March" has to mean some
    particular March, and the last one in the data is the only defensible
    choice; the parser reports the assumption rather than hiding it.

    Args:
        frame:
            The dataset.

        date_column:
            The column carrying the timeline.

    Returns:
        The year, or None when there is no usable date column.
    """

    if not date_column or date_column not in frame.columns:
        return None

    parsed = aggregation.parse_dates(frame[date_column]).dropna()

    if parsed.empty:
        return None

    return int(parsed.max().year)


def _quarter_start(year: int, quarter: int) -> str:
    """First day of a quarter, as an ISO date."""

    return f"{year:04d}-{3 * (quarter - 1) + 1:02d}-01"


def read_windows(
    scanner: Scanner,
    frame: pd.DataFrame,
    date_column: str | None,
    matched: list[str]
) -> tuple[nlq.Window | None, nlq.Window | None, str | None]:
    """
    Read the period phrases out of a question.

    Runs before everything else, so period words are never available to be
    read as column names or as a ranking limit.

    Args:
        scanner:
            The question, which this consumes from.

        frame:
            The dataset, for resolving a bare month name.

        date_column:
            The likely timeline column, for the same reason.

        matched:
            Collects reader-facing notes about what was recognised.

    Returns:
        The window, the comparison window when the question compares two
        periods, and the grain the phrasing implies.
    """

    window: nlq.Window | None = None
    comparison: nlq.Window | None = None
    grain: str | None = None

    # An explicit range first, so its dates are never read as years.
    date_pattern = (
        r"(\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2}|\d{4}-\d{2}|"
        rf"(?:{MONTH_PATTERN})\s+\d{{4}}|\d{{4}})"
    )

    # Each date branch searches before it consumes. A span is only claimed
    # once its dates actually read, so "between 1000 and 2000" is left for
    # the numeric reader instead of being swallowed as a pair of years.
    between = scanner.search(
        rf"\bbetween\s+{date_pattern}\s+and\s+{date_pattern}\b"
    )

    if between:
        start = _read_date(between.group(1))
        end = _read_date(between.group(2), end=True)

        if start and end:
            scanner.consume(between.start(), between.end())
            matched.append(f"read the range {start} to {end}")

            return (
                nlq.Window("between_dates", start=start, end=end),
                None,
                None
            )

    since = scanner.search(rf"\b(?:since|from|after)\s+{date_pattern}\b")

    if since:
        start = _read_date(since.group(1))

        if start:
            scanner.consume(since.start(), since.end())
            matched.append(
                f"read '{since.group(0).strip()}' as {start} onwards"
            )
            window = nlq.Window("between_dates", start=start)

    if window is None:
        until = scanner.search(
            rf"\b(?:before|until|up\s+to)\s+{date_pattern}\b"
        )

        if until:
            end = _read_date(until.group(1), end=True)

            if end:
                scanner.consume(until.start(), until.end())
                matched.append(
                    f"read '{until.group(0).strip()}' as up to {end}"
                )
                window = nlq.Window("between_dates", end=end)

    # "the last 3 months", and the period before it when comparing.
    if window is None:
        counted = scanner.take(
            rf"\blast\s+(\d+)\s+({GRAIN_PATTERN})\b"
        )

        if counted:
            count = int(counted.group(1))
            grain = GRAIN_WORDS[counted.group(2)]
            window = nlq.Window("last_periods", count=count, grain=grain)
            matched.append(f"read '{counted.group(0).strip()}' as a window")

    if window is None:
        single = scanner.take(
            rf"\b(?:last|previous|prior|this|current|latest)\s+"
            rf"({GRAIN_PATTERN})\b"
        )

        if single:
            grain = GRAIN_WORDS[single.group(1)]
            window = nlq.Window("last_periods", count=1, grain=grain)
            matched.append(f"read '{single.group(0).strip()}' as one {grain}")

    if window is None and scanner.take(r"\byear\s+to\s+date\b|\bytd\b"):
        grain = "year"
        window = nlq.Window("last_periods", count=1, grain=grain)
        matched.append("read 'year to date' as the most recent year")

    # A named quarter, month or year.
    if window is None:
        quarter = scanner.take(r"\bq([1-4])\s*(\d{4})\b")

        if quarter:
            grain = "quarter"
            start = _quarter_start(
                int(quarter.group(2)),
                int(quarter.group(1))
            )
            window = nlq.Window("period", grain=grain, start=start)
            matched.append(f"read '{quarter.group(0).strip()}' as a quarter")

    if window is None:
        named = scanner.take(
            rf"\b(?:in|during|for)?\s*({MONTH_PATTERN})\s+(\d{{4}})\b"
        )

        if named:
            grain = "month"
            month = MONTH_NUMBERS[named.group(1)]
            start = f"{int(named.group(2)):04d}-{month:02d}-01"
            window = nlq.Window("period", grain=grain, start=start)
            matched.append(f"read '{named.group(0).strip()}' as a month")

    if window is None:
        bare_month = scanner.take(
            rf"\b(?:in|during|for)\s+({MONTH_PATTERN})\b"
        )

        if bare_month:
            year = _latest_year(frame, date_column)

            if year is None:
                matched.append(
                    f"could not tell which year "
                    f"'{bare_month.group(1)}' means"
                )

            else:
                grain = "month"
                month = MONTH_NUMBERS[bare_month.group(1)]
                start = f"{year:04d}-{month:02d}-01"
                window = nlq.Window("period", grain=grain, start=start)
                matched.append(
                    f"read '{bare_month.group(1)}' as "
                    f"{calendar.month_name[month]} {year}, the latest "
                    f"year in the data"
                )

    if window is None:
        year_only = scanner.search(r"\b(?:in|during|for)\s+(\d{4})\b")

        if year_only and _plausible_year(year_only.group(1)):
            scanner.consume(year_only.start(), year_only.end())
            grain = "year"
            window = nlq.Window(
                "period",
                grain=grain,
                start=f"{int(year_only.group(1)):04d}-01-01"
            )
            matched.append(f"read '{year_only.group(1)}' as a year")

    return window, comparison, grain


def _plausible_year(text: str) -> bool:
    """
    Report whether a four digit number is plausibly a year.

    Args:
        text:
            The digits as written.

    Returns:
        True when the number falls in a range a dataset could cover.
    """

    try:
        year = int(text)

    except (TypeError, ValueError):
        return False

    return EARLIEST_PLAUSIBLE_YEAR <= year <= LATEST_PLAUSIBLE_YEAR


def read_grain(
    scanner: Scanner,
    matched: list[str]
) -> str | None:
    """
    Read a request for a figure per period.

    Runs before the split-by-column reader, so "by month" is understood as
    a monthly grain rather than a breakdown by a column called month.

    Args:
        scanner:
            The question, which this consumes from.

        matched:
            Collects reader-facing notes.

    Returns:
        A grain, or None when the question does not ask for one.
    """

    for phrase, grain in GRAIN_PHRASES:
        if scanner.take_phrase(f" {phrase} ") >= 0:
            matched.append(f"read '{phrase}' as a {grain} period")

            return grain

    return None


def _read_date(text: str, end: bool = False) -> str | None:
    """
    Turn a written date into an ISO date.

    Args:
        text:
            The date as written.

        end:
            When set, a partial date is read as the last day it covers
            rather than the first, so "until 2024" means the end of 2024.

    Returns:
        An ISO date, or None when the text could not be read.
    """

    cleaned = text.strip().replace("/", "-")

    if re.fullmatch(r"\d{4}", cleaned):
        if not _plausible_year(cleaned):
            return None

        year = int(cleaned)

        return f"{year:04d}-12-31" if end else f"{year:04d}-01-01"

    if re.fullmatch(r"\d{4}-\d{2}", cleaned):
        year, month = (int(part) for part in cleaned.split("-"))

        if end:
            last = calendar.monthrange(year, month)[1]

            return f"{year:04d}-{month:02d}-{last:02d}"

        return f"{year:04d}-{month:02d}-01"

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
        return cleaned

    named = re.fullmatch(rf"({MONTH_PATTERN})\s+(\d{{4}})", cleaned)

    if named:
        year = int(named.group(2))
        month = MONTH_NUMBERS[named.group(1)]

        if end:
            last = calendar.monthrange(year, month)[1]

            return f"{year:04d}-{month:02d}-{last:02d}"

        return f"{year:04d}-{month:02d}-01"

    return None


def read_column_mentions(
    scanner: Scanner,
    frame: pd.DataFrame
) -> list[tuple[str, int]]:
    """
    Find the columns a question names, and where it names them.

    Longer names are matched first, so a dataset with both "revenue" and
    "revenue per unit" does not have the second read as the first.

    Args:
        scanner:
            The question, which this consumes from.

        frame:
            The dataset.

    Returns:
        Each column named, with the position of the mention.
    """

    candidates = []

    for name in frame.columns:
        for variant in _column_variants(name):
            candidates.append((str(name), variant))

    candidates.sort(key=lambda entry: len(entry[1]), reverse=True)

    found: list[tuple[str, int]] = []
    seen: set[str] = set()

    for name, variant in candidates:
        if name in seen:
            continue

        # A trailing "s" is tolerated, because a question says "regions"
        # where the column is called "region".
        match = scanner.take(rf"\b{re.escape(variant)}s?\b")

        if match:
            found.append((name, match.start()))
            seen.add(name)

    return found


def read_value_mentions(
    scanner: Scanner,
    frame: pd.DataFrame,
    skip_columns: frozenset[str]
) -> list[tuple[str, str, int]]:
    """
    Find category values a question names.

    "Revenue in north" names no column at all, but "north" is a value of
    one, and finding it is what turns the question into a filter. Only
    low-cardinality columns are scanned, and only values long enough not
    to collide with ordinary words.

    Args:
        scanner:
            The question, which this consumes from.

        frame:
            The dataset.

        skip_columns:
            Columns already given a role, whose values should not become
            filters.

    Returns:
        Each match as the column, the value and the position.
    """

    found: list[tuple[str, str, int]] = []
    scanned = 0

    for name in frame.columns:
        if str(name) in skip_columns:
            continue

        if scanned >= MAX_VALUES_SCANNED:
            break

        series = frame[name]

        # A column of numbers holds no labels worth matching, whether it is
        # stored as numbers or as text that reads as numbers.
        if schema.is_measurable(series):
            continue

        try:
            values = series.dropna().unique()

        except TypeError:
            continue

        if len(values) > MAX_VALUES_PER_COLUMN:
            continue

        rendered = sorted(
            {
                str(value).strip().lower()
                for value in values
                if str(value).strip()
            },
            key=len,
            reverse=True
        )

        for value in rendered:
            scanned += 1

            if scanned >= MAX_VALUES_SCANNED:
                break

            if len(value) < MINIMUM_VALUE_LENGTH:
                continue

            if value in VALUE_STOP_WORDS:
                continue

            match = scanner.take(rf"\b{re.escape(value)}\b")

            if match:
                found.append((str(name), value, match.start()))
                break

    return found


@dataclass
class NumericCondition:
    """
    A numeric comparison found in a question, not yet tied to a column.

    Attributes:
        operator:
            One of nlq.FILTER_OPERATORS.

        values:
            The numbers compared against.

        position:
            Where the phrase sat in the question, used to attach it to the
            nearest column named.

        text:
            The phrase as written, for reader-facing notes.
    """

    operator: str
    values: tuple[float, ...]
    position: int
    text: str


def find_numeric_conditions(scanner: Scanner) -> list[NumericCondition]:
    """
    Find numeric comparisons, before anything else can eat their words.

    Recognition is separated from attachment for an ordering reason worth
    stating. "Least" is a direction word and "at least 100" is a
    comparison, and whichever reader runs first claims the word. Attaching
    a comparison needs the column positions, which are found later, so the
    phrase is recognised and consumed here and matched to a column
    afterwards.

    Args:
        scanner:
            The question, which this consumes from.

    Returns:
        The comparisons found, in the order they appear.
    """

    found: list[NumericCondition] = []

    ranged = scanner.take(
        r"\bbetween\s+([\d,]+(?:\.\d+)?)\s+and\s+([\d,]+(?:\.\d+)?)\b"
    )

    if ranged:
        low = _read_number(ranged.group(1))
        high = _read_number(ranged.group(2))

        if low is not None and high is not None:
            found.append(NumericCondition(
                "between",
                (min(low, high), max(low, high)),
                ranged.start(),
                ranged.group(0).strip()
            ))

    for phrase, operator in NUMERIC_PHRASES:
        while True:
            match = scanner.take(
                rf"\b{re.escape(phrase)}\s+([\d,]+(?:\.\d+)?)\b"
            )

            if not match:
                break

            value = _read_number(match.group(1))

            if value is None:
                continue

            found.append(NumericCondition(
                operator,
                (value,),
                match.start(),
                match.group(0).strip()
            ))

    return sorted(found, key=lambda entry: entry.position)


def attach_numeric_conditions(
    conditions: list[NumericCondition],
    frame: pd.DataFrame,
    column_positions: list[tuple[str, int]],
    matched: list[str],
    warnings: list[str]
) -> list[nlq.Filter]:
    """
    Tie each numeric comparison to the column it is about.

    A phrase like "over 100" is meaningless alone, so it is attached to the
    numeric column named nearest to it. When no numeric column is named,
    the comparison is reported as unused rather than guessed at: attaching
    it to whatever happened to be the measure would silently answer a
    different question.

    Args:
        conditions:
            Output of find_numeric_conditions.

        frame:
            The dataset.

        column_positions:
            Columns named in the question, with their positions.

        matched:
            Collects reader-facing notes.

        warnings:
            Collects things that change how to read the answer.

    Returns:
        The conditions that could be tied to a column.
    """

    numeric_columns = [
        (name, position)
        for name, position in column_positions
        if name in frame.columns
        and schema.is_measurable(frame[name])
    ]

    filters: list[nlq.Filter] = []

    for condition in conditions:
        column = _nearest_column(numeric_columns, condition.position)

        if column is None:
            warnings.append(
                f"'{condition.text}' was not used, because the question "
                "does not say which numeric column it applies to."
            )

            continue

        filters.append(nlq.Filter(
            column,
            condition.operator,
            condition.values
        ))
        matched.append(
            f"read '{condition.text}' as a condition on '{column}'"
        )

    return filters


def _nearest_column(
    columns: list[tuple[str, int]],
    position: int
) -> str | None:
    """Return the column named closest to a position in the question."""

    if not columns:
        return None

    return min(
        columns,
        key=lambda entry: abs(entry[1] - position)
    )[0]


def choose_intent(
    scanner: Scanner,
    grain: str | None,
    segment: str | None,
    matched: list[str]
) -> tuple[str, str | None, str]:
    """
    Decide what the question is asking for.

    Args:
        scanner:
            The question, which this consumes from.

        grain:
            Grain the question asked for a figure per, when any. This is
            deliberately not the grain of a window: "last month" narrows
            the rows, while "by month" asks for a row per month.

        segment:
            Column the question asked to split by, when any.

        matched:
            Collects reader-facing notes.

    Returns:
        The intent, the grain it implies, and a refusal reason when no
        intent could be read.
    """

    for phrase in COMPARISON_PHRASES:
        if scanner.take_phrase(phrase) >= 0:
            matched.append(f"read '{phrase.strip()}' as a comparison")

            return "compare", grain, ""

    for phrase in SHARE_PHRASES:
        if scanner.take_phrase(phrase) >= 0:
            matched.append(f"read '{phrase}' as a share")

            return "share", grain, ""

    # "Which month was worst" asks about a period. "Which region was
    # worst" asks about a part, and is a ranking.
    period_question = scanner.take(
        rf"\b(?:which|what)\s+({GRAIN_PATTERN})\b"
    )

    if period_question:
        matched.append(
            f"read '{period_question.group(0).strip()}' as a question "
            "about periods"
        )

        return "extreme", GRAIN_WORDS[period_question.group(1)], ""

    if scanner.take(r"\bwhen\s+(?:was|were|did)\b"):
        matched.append("read 'when was' as a question about periods")

        return "extreme", grain, ""

    for phrase in TREND_PHRASES:
        if scanner.take_phrase(phrase) >= 0:
            matched.append(f"read '{phrase}' as a question about direction")

            return "trend", grain, ""

    if segment:
        return "breakdown", grain, ""

    for phrase in OVER_TIME_PHRASES:
        if scanner.take_phrase(phrase) >= 0:
            matched.append(f"read '{phrase}' as a request for a series")

            return "timeseries", grain, ""

    if grain is not None:
        return "timeseries", grain, ""

    return "aggregate", grain, ""


def read_segment(
    scanner: Scanner,
    frame: pd.DataFrame,
    matched: list[str],
    allow_bare: bool = False
) -> tuple[str | None, str]:
    """
    Read a request to split the figures up.

    Three cases have to be told apart, and conflating any two of them
    produces a confident answer to a different question.

    "By region" names a dimension, and that is the split.

    "By revenue" names a number, which in "top 2 regions by revenue" says
    what to rank on rather than what to split by. The lead word alone is
    consumed so the column is still read as the measure.

    "By order reference" names a dimension that cannot group anything,
    because it holds one value per row. The reader asked for that split in
    so many words, so it is refused with the reason rather than dropped.

    Args:
        scanner:
            The question, which this consumes from.

        frame:
            The dataset.

        matched:
            Collects reader-facing notes.

        allow_bare:
            Accept a column named with no lead word. Set when a ranking
            was found, because "top 2 regions by revenue" names the
            dimension without saying "by".

    Returns:
        The column to split by, and a refusal reason when the question
        asked to split by a column that cannot group the data.
    """

    candidates = []

    for name in frame.columns:
        for variant in _column_variants(name):
            candidates.append((str(name), variant))

    candidates.sort(key=lambda entry: len(entry[1]), reverse=True)

    for lead in ("by", "per", "for each", "across", "which", "what"):
        for name, variant in candidates:
            match = scanner.search(
                rf"\b{lead}\s+{re.escape(variant)}s?\b"
            )

            if not match:
                continue

            # "By" before a number names what to rank on, not what to
            # split by: "top 2 regions by revenue" ranks regions. Only the
            # lead word is consumed, leaving the column to be read as the
            # measure.
            if schema.is_measurable(frame[name]):
                scanner.consume(
                    match.start(),
                    match.start() + len(lead)
                )

                continue

            suitability = segments.describe_segment_suitability(
                frame,
                name
            )

            if not suitability["usable"]:
                # The reader asked for this split in so many words. Saying
                # why it cannot be done beats quietly answering the
                # question without it.
                return None, suitability["reason"]

            scanner.consume(match.start(), match.end())
            matched.append(
                f"read '{match.group(0).strip()}' as a split by '{name}'"
            )

            return name, ""

    if allow_bare:
        for name, variant in candidates:
            if not segments.describe_segment_suitability(
                frame,
                name
            )["usable"]:
                continue

            match = scanner.take(rf"\b{re.escape(variant)}s?\b")

            if match:
                matched.append(
                    f"read '{match.group(0).strip()}' as the dimension "
                    "to rank"
                )

                return name, ""

    return None, ""


def read_ranking(
    scanner: Scanner,
    matched: list[str]
) -> tuple[int | None, str | None]:
    """
    Read a limit and a direction from the question.

    Args:
        scanner:
            The question, which this consumes from.

        matched:
            Collects reader-facing notes.

    Returns:
        The limit and the direction, either of which may be None.
    """

    limit = None
    direction = None

    counted = scanner.take(r"\b(top|bottom|highest|lowest|worst|best)\s+(\d+)\b")

    if counted:
        limit = int(counted.group(2))
        direction = (
            "lowest"
            if counted.group(1) in {"bottom", "lowest", "worst"}
            else "highest"
        )
        matched.append(
            f"read '{counted.group(0).strip()}' as a ranking"
        )

        return limit, direction

    for phrase, chosen in DIRECTION_PHRASES:
        if scanner.take_phrase(f" {phrase} ") >= 0:
            matched.append(f"read '{phrase}' as {chosen} first")

            return None, chosen

    return None, None


def read_aggregation(
    scanner: Scanner,
    matched: list[str]
) -> str | None:
    """
    Read how the rows should be combined.

    Args:
        scanner:
            The question, which this consumes from.

        matched:
            Collects reader-facing notes.

    Returns:
        An aggregation name, or None when the question does not say.
    """

    for phrase, name in AGGREGATION_PHRASES:
        if scanner.take_phrase(f" {phrase} ") >= 0:
            matched.append(f"read '{phrase}' as {name}")

            return name

    return None


def parse_question(
    question: str,
    frame: pd.DataFrame
) -> ParseResult:
    """
    Turn a question into a validated plan, using rules only.

    Args:
        question:
            What the reader typed.

        frame:
            The dataset the question is about.

    Returns:
        A ParseResult. When understood is false the question should be
        handed to the model planner, and reason says what stopped the
        rules.
    """

    if not str(question).strip():
        return ParseResult(
            understood=False,
            reason="The question was empty."
        )

    if frame.empty:
        return ParseResult(
            understood=False,
            reason="The dataset has no rows, so there is nothing to ask."
        )

    matched: list[str] = []
    warnings: list[str] = []

    scanner = Scanner(normalise_question(question))

    detected = schema.detect_business_schema(frame)

    # Period phrases first, so their words are never available to be read
    # as a column name or as a ranking limit.
    window, comparison, grain = read_windows(
        scanner,
        frame,
        detected["date"],
        matched
    )

    # Numeric comparisons next, so a direction word cannot claim the
    # "least" in "at least 100". They are attached to a column further
    # down, once the column positions are known.
    conditions = find_numeric_conditions(scanner)

    aggregation_name = read_aggregation(scanner, matched)
    limit, direction = read_ranking(scanner, matched)

    # A grain phrase before a split phrase, so "by month" is a period and
    # only "by region" is a breakdown.
    #
    # The two grains are kept apart deliberately. "Last month" names a
    # window and asks for one figure; "by month" asks for a figure per
    # month. Both mention a month, and treating them alike turns "total
    # revenue last month" into a one-row time series.
    window_grain = grain
    series_grain = read_grain(scanner, matched)

    segment, segment_refusal = read_segment(
        scanner,
        frame,
        matched,
        allow_bare=limit is not None
    )

    if segment_refusal:
        return ParseResult(
            understood=False,
            reason=segment_refusal,
            matched=matched,
            warnings=warnings
        )

    column_positions = read_column_mentions(scanner, frame)

    filters = attach_numeric_conditions(
        conditions,
        frame,
        column_positions,
        matched,
        warnings
    )

    intent, intent_grain, reason = choose_intent(
        scanner,
        series_grain,
        segment,
        matched
    )

    # A grain named by the question wins; otherwise the window's grain is
    # used, because the period arithmetic still needs one.
    grain = intent_grain or series_grain or window_grain

    if reason:
        return ParseResult(
            understood=False,
            reason=reason,
            matched=matched,
            warnings=warnings
        )

    named = [name for name, _ in column_positions]

    measure_columns = [
        name
        for name in named
        if schema.is_measurable(frame[name])
        and name not in {entry.column for entry in filters}
    ]

    date_columns = [
        name
        for name in named
        if name in set(schema.date_candidates(frame))
    ]

    other_columns = [
        name
        for name in named
        if name not in measure_columns
        and name not in date_columns
        and name != segment
    ]

    if len(measure_columns) > 1:
        return ParseResult(
            understood=False,
            reason=(
                "The question names more than one thing to measure "
                f"({', '.join(measure_columns)}), so it is not clear "
                "which one to total."
            ),
            matched=matched,
            warnings=warnings
        )

    measure = measure_columns[0] if measure_columns else None
    date = date_columns[0] if date_columns else None

    if segment is None and other_columns and intent in {
        "breakdown",
        "share",
    }:
        segment = other_columns[0]
        matched.append(f"read '{segment}' as the column to split by")

    # A category value names a filter, unless the question is asking for
    # that one part's share of the whole.
    used_columns = frozenset(
        name
        for name in (measure, date, segment)
        if name
    )

    value_mentions = read_value_mentions(scanner, frame, used_columns)
    segment_value = None

    for column, value, _ in value_mentions:
        if intent == "share" and segment in (None, column):
            segment = column
            segment_value = value
            matched.append(
                f"read '{value}' as a part of '{column}'"
            )

            continue

        filters.append(nlq.Filter(column, "equals", (value,)))
        matched.append(f"read '{value}' as a filter on '{column}'")

    if intent == "share" and segment_value is None:
        return ParseResult(
            understood=False,
            reason=(
                "The question asks for a share but does not name which "
                "part of the data it is about."
            ),
            matched=matched,
            warnings=warnings
        )

    # "Aggregate" is where a question lands when no phrase matched, so it
    # needs a signal of its own before it can be claimed. Without this
    # check every sentence typed into the box comes back as the total of
    # whatever column detection liked best, which is a confident answer to
    # a question nobody asked and leaves the model fallback unreachable.
    if intent == "aggregate" and not any((
        aggregation_name is not None,
        bool(column_positions),
        window is not None,
        bool(filters),
    )):
        return ParseResult(
            understood=False,
            reason=(
                "Nothing in the question names a figure, a column, a "
                "period or a condition, so there is no way to tell what "
                "it is asking for."
            ),
            matched=matched,
            ignored=[
                word
                for word in scanner.remaining_words()
                if word not in FILLER_WORDS
            ],
            warnings=warnings
        )

    # A column named but given no role was quietly dropped, and only the
    # reader can tell whether that matters.
    for name in other_columns:
        if name != segment:
            warnings.append(
                f"'{name}' was named in the question but not used, "
                "because it is not the measure, the timeline or the "
                "column being split by."
            )

    if intent == "compare":
        compare_grain = grain or "month"

        # "This month against last month" names two periods, and the first
        # was already consumed as the window. Consuming the second stops it
        # being reported as a word the parser ignored, which would read as
        # though half the question had been dropped.
        # "The month before" names the same second period the other way
        # round.
        second = scanner.take(
            rf"\b(?:(?:last|previous|prior|the\s+previous)\s+"
            rf"({GRAIN_PATTERN}|period)"
            rf"|(?:the\s+)?({GRAIN_PATTERN}|period)\s+before)\b"
        )

        second_grain = (
            (second.group(1) or second.group(2)) if second else None
        )

        if second_grain in GRAIN_WORDS:
            compare_grain = GRAIN_WORDS[second_grain]

        # "Than the previous period" loses "previous" to the comparison
        # phrase, leaving the period word on its own.
        if not second:
            scanner.take(r"\bperiod\b")

        window = nlq.Window("last_periods", count=1, grain=compare_grain)
        comparison = nlq.Window(
            "last_periods",
            count=1,
            grain=compare_grain,
            offset=1
        )
        grain = compare_grain

    # "Which region" asks for one part, not a table of them.
    asks_which = "which" in str(question).lower()

    if intent in {"breakdown", "extreme"} and limit is None and asks_which:
        limit = DEFAULT_WHICH_LIMIT

    plan = nlq.QueryPlan(
        intent=intent,
        measure=measure,
        aggregation=aggregation_name or "sum",
        date=date,
        grain=grain,
        segment=segment,
        segment_value=segment_value,
        filters=tuple(filters),
        window=window or nlq.EVERYTHING,
        comparison=comparison,
        top_n=limit,
        direction=direction or "highest",
        question=str(question).strip(),
        source="rules"
    )

    # Detection fills in the columns the question left implicit, and drops
    # to counting rows when the dataset has nothing measurable in it.
    plan = nlq.resolve_plan_columns(plan, frame)

    validation = nlq.validate_plan(plan, frame)

    ignored = [
        word
        for word in scanner.remaining_words()
        if word not in FILLER_WORDS
    ]

    # A question read only in part is not answered from the part. "Average
    # age of customers" read as "average" alone is the mean of whatever
    # measure detection picked, which is a confident answer to a question
    # nobody asked. The evaluation set found thirteen of these. The words
    # below are the ones that can go unread without changing what is
    # computed; anything else hands the question to the model planner, or,
    # with no model, back to the reader with the words that were not read.
    harmless = HARMLESS_UNREAD_WORDS | (
        HARMLESS_UNREAD_IN_A_COMPARISON if plan.intent == "compare" else set()
    )
    unread = [word for word in ignored if word not in harmless]

    if unread:
        return ParseResult(
            understood=False,
            reason=(
                "Part of the question was not read: "
                + ", ".join(f"'{word}'" for word in unread)
                + ". Answering from the rest could answer a different "
                "question."
            ),
            matched=matched,
            ignored=ignored,
            warnings=warnings
        )

    if not validation.valid:
        return ParseResult(
            understood=False,
            reason=" ".join(validation.errors),
            matched=matched,
            ignored=ignored,
            warnings=warnings + validation.warnings
        )

    return ParseResult(
        plan=validation.plan,
        understood=True,
        matched=matched,
        ignored=ignored,
        warnings=warnings + validation.warnings
    )
