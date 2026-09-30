"""
The planner's reply as a schema, for providers that enforce one.

The default planner asks for JSON in the prompt and reads it back with
nlq_model.extract_json_object, which tolerates prose and code fences
around the object. With AI_ANALYST_PLANNER_OUTPUT=structured the schema
below is handed to the provider instead, which constrains generation to
it: the fields and the vocabularies can no longer be misspelt.

Either way the reply goes through the same gate afterwards. The schema is
serialised back to JSON text and read by plan_from_dict and validate_plan,
so a structured reply earns no trust the text reply does not. The schema
checks shape; only the dataset can check that a plan makes sense.

The vocabularies are taken from nlq and aggregation rather than repeated,
so the schema cannot drift from what the executor accepts.
"""

import json
from typing import Literal

from pydantic import BaseModel, Field

import aggregation
import nlq

Intent = Literal[tuple(nlq.INTENTS)]
Aggregation = Literal[tuple(aggregation.AGGREGATIONS)]
Grain = Literal[tuple(aggregation.PERIOD_GRAINS)]
Operator = Literal[tuple(nlq.FILTER_OPERATORS)]
WindowKind = Literal[tuple(nlq.WINDOW_KINDS)]
Direction = Literal[tuple(nlq.DIRECTIONS)]


class PlanFilter(BaseModel):
    """One condition on a column."""

    column: str = Field(description="A column name from the schema.")
    operator: Operator
    values: list[str] = Field(
        default_factory=list,
        description="Comparison values written as text, numbers included.",
    )


class PlanWindow(BaseModel):
    """A span of the timeline."""

    kind: WindowKind
    count: int | None = None
    grain: Grain | None = None
    start: str | None = Field(default=None, description="YYYY-MM-DD")
    end: str | None = Field(default=None, description="YYYY-MM-DD")
    offset: int | None = None


class PlannerReply(BaseModel):
    """Everything the planner may say. Nothing else is accepted."""

    on_topic: bool
    answerable: bool | None = None
    intent: Intent | None = None
    measure: str | None = None
    aggregation: Aggregation | None = None
    date: str | None = None
    grain: Grain | None = None
    segment: str | None = None
    segment_value: str | None = None
    filters: list[PlanFilter] | None = None
    window: PlanWindow | None = None
    comparison: PlanWindow | None = None
    top_n: int | None = None
    direction: Direction | None = None


def reply_to_text(reply: PlannerReply) -> str:
    """
    Write a structured reply as the JSON text the planner reads.

    Unset fields are left out rather than sent as null, so the reply reads
    exactly like a well-behaved text reply.
    """

    return json.dumps(reply.model_dump(exclude_none=True))
