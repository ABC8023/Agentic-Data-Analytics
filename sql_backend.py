"""
A query plan, compiled to parameterised SQL.

The model never writes SQL. It writes a plan in a fixed vocabulary, the
plan is validated against the dataset, and only then is it compiled here.
That keeps what text-to-SQL is useful for, answering questions against a
database, without letting generated text reach a query:

    column names are checked against the table's columns and quoted as
    identifiers, so a name cannot close the quote and start a clause

    every value a reader or a model supplied is a bound parameter, never
    part of the SQL text

The pandas executor in nlq_answer remains the reference. This compiler
covers the plans that map onto one grouped query, aggregate and breakdown,
with every filter operator and any window, and refuses the rest with
UnsupportedPlan rather than approximating them. tests/test_sql_backend.py
runs every supported plan in the evaluation set through both and requires
the same figures.

Two decisions keep the results identical to the pandas executor:

    Numbers are read with TRY_CAST, so a cell like "not recorded" is
    skipped rather than failing the query, as pandas.to_numeric does.

    A relative window ("last 3 months") is resolved to dates by the same
    function the pandas path uses, including leaving out a period still in
    progress, and passed in as parameters. The rule lives in one place.

Dates are read with TRY_CAST to TIMESTAMP, which reads ISO dates. In a
warehouse the column would already be typed; for loosely formatted text
dates the pandas path, which parses mixed formats, is the one to use.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import aggregation
import nlq
import nlq_answer
import segments

SUPPORTED_INTENTS = frozenset({"aggregate", "breakdown"})

SQL_AGGREGATES = {
    "sum": "SUM",
    "mean": "AVG",
    "median": "MEDIAN",
    "min": "MIN",
    "max": "MAX",
}

TABLE_NAME = "dataset"


class UnsupportedPlan(ValueError):
    """The plan has no single-query SQL form here. Use the pandas path."""


@dataclass(frozen=True)
class CompiledQuery:
    """
    SQL text and the values bound to it.

    Attributes:
        sql:
            The statement, with ? placeholders.

        parameters:
            Values for the placeholders, in order.

        shape:
            "scalar" for one figure, "table" for a part per row.
    """

    sql: str
    parameters: tuple[Any, ...] = field(default_factory=tuple)
    shape: str = "scalar"

    def display(self) -> str:
        """The statement with its parameters listed, for a reader."""

        if not self.parameters:
            return self.sql

        listed = ", ".join(repr(value) for value in self.parameters)

        return f"{self.sql}\n-- parameters: {listed}"


def quote_identifier(name: str, columns: set[str]) -> str:
    """
    Quote a column name as an SQL identifier, if the table has it.

    Args:
        name:
            The column.

        columns:
            The table's columns. A name outside this set is refused, so
            quoting is a second line of defence rather than the only one.

    Returns:
        The quoted identifier.

    Raises:
        UnsupportedPlan: If the column is not in the table.
    """

    if name not in columns:
        raise UnsupportedPlan(f"'{name}' is not a column of the table.")

    return '"' + str(name).replace('"', '""') + '"'


def _text(identifier: str) -> str:
    """A column as trimmed lowercase text, the way labels are compared."""

    return f"lower(trim(CAST({identifier} AS VARCHAR)))"


def _number(identifier: str) -> str:
    return f"TRY_CAST({identifier} AS DOUBLE)"


def _filter_clause(
    entry: nlq.Filter,
    columns: set[str]
) -> tuple[str, list[Any]]:
    """One filter as a WHERE condition and its parameters."""

    column = quote_identifier(entry.column, columns)
    operator = entry.operator

    if operator == "is_missing":
        return f"{column} IS NULL", []

    if operator == "is_present":
        return f"{column} IS NOT NULL", []

    if operator in nlq.NUMERIC_OPERATORS:
        values = [float(value) for value in entry.values]
        number = _number(column)

        comparisons = {
            "greater_than": ">",
            "greater_or_equal": ">=",
            "less_than": "<",
            "less_or_equal": "<=",
        }

        if operator in comparisons:
            # NULL, an unreadable number, compares as unknown and the row
            # is dropped, as it is on the pandas path.
            return f"{number} {comparisons[operator]} ?", values[:1]

        return f"{number} BETWEEN ? AND ?", values[:2]

    text = _text(column)
    wanted = [str(value).strip().lower() for value in entry.values]

    if operator == "contains":
        return f"coalesce(position(? IN {text}) > 0, FALSE)", wanted[:1]

    placeholders = ", ".join("?" for _ in wanted)

    if operator in {"equals", "in"}:
        return f"{text} IN ({placeholders})", wanted

    # A blank cell is not equal to any label, so "not north" keeps it, as
    # the pandas path does. Plain NOT IN would drop it as unknown.
    return f"coalesce({text} NOT IN ({placeholders}), TRUE)", wanted


def _window_bounds(
    frame: pd.DataFrame,
    plan: nlq.QueryPlan
) -> tuple[Any, Any]:
    """Resolve the plan's window to dates with the pandas path's rule."""

    if plan.window.kind == "all" or plan.date is None:
        return None, None

    dates = aggregation.parse_dates(frame[plan.date])
    resolved = nlq_answer.resolve_window(
        dates, plan.window, plan.grain or "month"
    )

    if not resolved.get("success"):
        raise UnsupportedPlan(resolved.get("error", "The window cannot be read."))

    return resolved["start"], resolved["end"]


def compile_plan(
    plan: nlq.QueryPlan,
    frame: pd.DataFrame,
    table: str = TABLE_NAME
) -> CompiledQuery:
    """
    Compile a validated plan to one parameterised statement.

    Args:
        plan:
            A plan that has passed nlq.validate_plan.

        frame:
            The dataset, for its column names and to anchor a relative
            window. No rows are read into the SQL.

        table:
            Table name to select from. Quoted as an identifier.

    Returns:
        The compiled query.

    Raises:
        UnsupportedPlan: For an intent or aggregation this compiler does
        not cover.
    """

    if plan.intent not in SUPPORTED_INTENTS:
        raise UnsupportedPlan(f"'{plan.intent}' plans are answered in pandas.")

    counting = plan.measure is None or plan.aggregation == "count"

    if not counting and plan.aggregation not in SQL_AGGREGATES:
        raise UnsupportedPlan(
            f"'{plan.aggregation}' is answered in pandas."
        )

    columns = {str(name) for name in frame.columns}
    table_identifier = '"' + table.replace('"', '""') + '"'

    conditions: list[str] = []
    parameters: list[Any] = []

    for entry in plan.filters:
        clause, values = _filter_clause(entry, columns)
        conditions.append(clause)
        parameters.extend(values)

    start, end = _window_bounds(frame, plan)

    if start is not None or end is not None:
        date = f"TRY_CAST({quote_identifier(plan.date, columns)} AS TIMESTAMP)"
        conditions.append(f"{date} IS NOT NULL")

        if start is not None:
            conditions.append(f"{date} >= ?")
            parameters.append(pd.Timestamp(start).to_pydatetime())

        if end is not None:
            conditions.append(f"{date} <= ?")
            parameters.append(pd.Timestamp(end).to_pydatetime())

    if counting:
        figure = "CAST(COUNT(*) AS DOUBLE)"

    else:
        measure = _number(quote_identifier(plan.measure, columns))
        figure = f"{SQL_AGGREGATES[plan.aggregation]}({measure})"

    where = f"\nWHERE {' AND '.join(conditions)}" if conditions else ""

    if plan.intent == "aggregate":
        return CompiledQuery(
            sql=f"SELECT {figure} AS value\nFROM {table_identifier}{where}",
            parameters=tuple(parameters),
            shape="scalar",
        )

    segment = quote_identifier(plan.segment, columns)
    label = (
        f"coalesce(nullif(trim(CAST({segment} AS VARCHAR)), ''), "
        f"'{segments.UNLABELLED_TEXT}')"
    )

    # Rows with an unreadable measure are left out of their part, as the
    # pandas path does, so a part can be missing when none of its rows
    # carry a number.
    if not counting:
        conditions.append(f"{_number(quote_identifier(plan.measure, columns))} IS NOT NULL")
        where = f"\nWHERE {' AND '.join(conditions)}"

    order = "ASC" if plan.direction == "lowest" else "DESC"
    limit = ""

    if plan.top_n:
        limit = "\nLIMIT ?"
        parameters.append(int(plan.top_n))

    return CompiledQuery(
        sql=(
            f"SELECT {label} AS segment, {figure} AS total\n"
            f"FROM {table_identifier}{where}\n"
            f"GROUP BY 1\n"
            f"ORDER BY total {order}, segment{limit}"
        ),
        parameters=tuple(parameters),
        shape="table",
    )


def run_plan(
    plan: nlq.QueryPlan,
    frame: pd.DataFrame,
    connection=None
) -> Any:
    """
    Compile a plan and run it on DuckDB against the frame.

    Args:
        plan:
            A validated plan.

        frame:
            The dataset, registered as the table for this query only.

        connection:
            An open DuckDB connection, or None for a fresh in-memory one.

    Returns:
        A float for an aggregate, or a DataFrame of segment and total for
        a breakdown.
    """

    import duckdb

    compiled = compile_plan(plan, frame)
    owned = connection is None
    connection = connection or duckdb.connect(":memory:")

    try:
        connection.register(TABLE_NAME, frame)
        result = connection.execute(compiled.sql, list(compiled.parameters))

        if compiled.shape == "scalar":
            row = result.fetchone()

            return None if row is None or row[0] is None else float(row[0])

        return result.df()

    finally:
        connection.unregister(TABLE_NAME)

        if owned:
            connection.close()
