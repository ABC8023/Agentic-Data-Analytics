"""
Evaluation of the question-answering path against known answers.

Each case is a question about a dataset with a known correct outcome:
either a figure, which was computed independently with plain pandas and
checked against sample_data/store_orders_facts.txt, or a refusal. A case is
run through the same answer_question the chat uses, so what is measured is
the behaviour a reader sees, not a component in isolation.

The metric that matters most is the confident-wrong rate: questions that
were answered with a figure when the figure was wrong, or when the right
response was to decline. A refusal costs the reader a rephrase. A wrong
number that looks right costs them a decision.

Two modes, chosen by whether an invoke callable is supplied:

    offline   No model. Questions the rules cannot read are refused, and
              a case that needs a model is reported as skipped rather than
              failed. Runs in CI with no key.

    live      Questions the rules cannot read go to the model planner.
              Every call's latency, token counts and resolved model name
              are recorded, so a run can be compared with the next one.
"""

import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

import file_io
import glossary as glossary_module
import llm
import nlq_answer
import planner_schema

CATEGORIES = ("rules", "model", "off_topic", "unanswerable", "adversarial")

# Plan fields a case may pin. Anything else in "expect" is a figure check
# or a behaviour flag.
PLAN_FIELDS = ("intent", "measure", "aggregation", "segment", "date")

PASS = "pass"
FAIL = "fail"
SKIPPED = "skipped"
ERROR = "error"
UNSCORED = (SKIPPED, ERROR)

DEFAULT_RELATIVE_TOLERANCE = 1e-6
DEFAULT_ABSOLUTE_TOLERANCE = 0.01


@dataclass
class CaseResult:
    """
    What happened to one case.

    Attributes:
        id, category, question:
            Copied from the case.

        status:
            pass, fail, skipped (needs a model, offline) or error (the
            model could not be reached).

        route:
            Who planned the answer: rules, model or nobody.

        expected_route:
            Who should have, when the case says. Empty otherwise.

        answered:
            Whether a figure was shown.

        confident_wrong:
            Answered, and either the figure was wrong or the case called
            for a refusal.

        not_on_topic:
            Whether the model judged the question off topic.

        reasons:
            Why the case failed or was skipped.

        plan:
            The plan in words, when there was one.

        headline:
            The answer as shown, or the refusal reason.

        latency_ms:
            Wall time for the whole question, rules included.

        model_calls, input_tokens, output_tokens:
            What the model was asked and what it cost.
    """

    id: str
    category: str
    question: str
    status: str
    route: str
    expected_route: str
    answered: bool
    confident_wrong: bool
    not_on_topic: bool
    reasons: list[str] = field(default_factory=list)
    plan: str = ""
    headline: str = ""
    latency_ms: float = 0.0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def load_cases(path: str) -> list[dict[str, Any]]:
    """
    Read cases from a JSON Lines file, refusing malformed ones.

    Args:
        path:
            File with one JSON object per line. Blank lines and lines
            starting with // are ignored.

    Returns:
        The cases.

    Raises:
        ValueError: If a case is malformed or an id repeats.
    """

    cases = []
    seen = set()

    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()

            if not text or text.startswith("//"):
                continue

            case = json.loads(text)

            for key in ("id", "category", "question", "expect"):
                if key not in case:
                    raise ValueError(f"Line {number}: missing '{key}'.")

            if case["category"] not in CATEGORIES:
                raise ValueError(
                    f"Line {number}: unknown category "
                    f"'{case['category']}'."
                )

            if case["id"] in seen:
                raise ValueError(f"Line {number}: repeated id '{case['id']}'.")

            seen.add(case["id"])
            cases.append(case)

    return cases


def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset the way the application does."""

    return file_io.load_tabular_file(path)["dataframe"]


def figures_match(expected: Any, actual: Any, check: dict[str, Any]) -> bool:
    """
    Compare one figure with its expected value.

    Numbers match within an absolute or a relative tolerance, whichever is
    looser. Text matches case-insensitively.
    """

    if actual is None:
        return False

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            number = float(actual)

        except (TypeError, ValueError):
            return False

        return math.isclose(
            number,
            float(expected),
            rel_tol=check.get("relative_tolerance", DEFAULT_RELATIVE_TOLERANCE),
            abs_tol=check.get("tolerance", DEFAULT_ABSOLUTE_TOLERANCE),
        )

    return str(actual).strip().lower() == str(expected).strip().lower()


def shown_text(answer: nlq_answer.Answer) -> str:
    """Everything the reader could see in an answer, as one string."""

    parts = [answer.headline, answer.error, json.dumps(answer.figures, default=str)]

    if answer.table is not None:
        parts.append(answer.table.to_csv(index=False))

    return "\n".join(parts)


def score_case(
    case: dict[str, Any],
    result: dict[str, Any],
    offline: bool
) -> tuple[str, list[str], bool]:
    """
    Decide whether a case passed.

    Args:
        case:
            The case.

        result:
            Output of nlq_answer.answer_question.

        offline:
            Whether no model was available.

    Returns:
        (status, reasons, confident_wrong).
    """

    expect = case["expect"]
    answer: nlq_answer.Answer = result["answer"]
    plan = result["plan"]
    answered = bool(answer.success)
    reasons: list[str] = []

    # A quota refusal or a timeout says nothing about the model's
    # judgement, so it is kept out of the pass rate rather than scored as
    # a wrong answer. The run reports how many there were.
    if result.get("model_unreachable") and not answered:
        return ERROR, [answer.error[-200:]], False

    text = shown_text(answer)

    for forbidden in expect.get("forbid", []):
        if forbidden.lower() in text.lower():
            reasons.append(f"shown text contains forbidden '{forbidden}'")

    if expect.get("refuse"):
        if answered:
            reasons.append(
                "answered with a figure when the right response was to "
                "decline"
            )

        if (
            not offline
            and expect.get("off_topic")
            and not result.get("not_on_topic")
        ):
            reasons.append("the model did not flag the question off topic")

        status = FAIL if reasons else PASS

        return status, reasons, answered

    if not answered:
        if expect.get("allow_refusal"):
            return (FAIL if reasons else PASS), reasons, False

        if offline and case["category"] == "model":
            return SKIPPED, ["needs a model"], False

        reasons.append(f"not answered: {answer.error[:160]}")

        return FAIL, reasons, False

    wrong = False

    for name in PLAN_FIELDS:
        if name not in expect or plan is None:
            continue

        wanted = expect[name]
        actual = getattr(plan, name, None)
        allowed = wanted if isinstance(wanted, list) else [wanted]

        if actual not in allowed:
            reasons.append(f"{name} was {actual!r}, expected {wanted!r}")
            wrong = True

    for check in expect.get("figures", []):
        actual = answer.figures.get(check["key"])

        if not figures_match(check["equals"], actual, check):
            reasons.append(
                f"figure '{check['key']}' was {actual!r}, expected "
                f"{check['equals']!r}"
            )
            wrong = True

    # The route is measured separately, not scored here: a right figure
    # reached by the model instead of the rules is slower and costlier,
    # but it is still right.
    status = FAIL if reasons else PASS

    return status, reasons, wrong


def run_case(
    case: dict[str, Any],
    frame: pd.DataFrame,
    invoke: Callable[[str], str] | None,
    completions: list[llm.Completion],
    glossary: "glossary_module.Glossary | None" = None
) -> CaseResult:
    """
    Ask one question and score the answer.

    Args:
        case:
            The case.

        frame:
            The dataset.

        invoke:
            Model callable, or None for an offline run.

        completions:
            The list the invoke callable appends to, so the calls made for
            this case can be attributed to it.

    Returns:
        The result.
    """

    before = len(completions)
    started = time.perf_counter()

    result = nlq_answer.answer_question(
        frame, case["question"], invoke, glossary=glossary
    )

    latency_ms = (time.perf_counter() - started) * 1000
    calls = completions[before:]

    status, reasons, wrong = score_case(case, result, invoke is None)
    answer = result["answer"]
    expected_route = case["expect"].get("route", "")

    return CaseResult(
        id=case["id"],
        category=case["category"],
        question=case["question"],
        status=status,
        route=result["planned_by"],
        expected_route=expected_route,
        answered=bool(answer.success),
        confident_wrong=bool(wrong and answer.success),
        not_on_topic=bool(result.get("not_on_topic")),
        reasons=reasons,
        plan=result["plan"].describe() if result["plan"] else "",
        headline=(answer.headline or answer.error)[:300],
        latency_ms=round(latency_ms, 1),
        model_calls=len(calls),
        input_tokens=sum(call.input_tokens or 0 for call in calls),
        output_tokens=sum(call.output_tokens or 0 for call in calls),
    )


def percentile(values: list[float], share: float) -> float | None:
    """Nearest-rank percentile, None for no values."""

    if not values:
        return None

    ordered = sorted(values)
    rank = max(1, math.ceil(share * len(ordered)))

    return ordered[rank - 1]


def summarise(
    results: list[CaseResult],
    offline: bool,
    input_price_per_million: float | None = None,
    output_price_per_million: float | None = None
) -> dict[str, Any]:
    """
    Reduce results to the numbers worth tracking between runs.

    Args:
        results:
            Every case result.

        offline:
            Whether the run had no model.

        input_price_per_million, output_price_per_million:
            Token prices, in any currency, when a cost estimate is wanted.
            Prices change, so none is assumed.

    Returns:
        The summary.
    """

    scored = [result for result in results if result.status not in UNSCORED]
    passed = [result for result in scored if result.status == PASS]
    answered = [result for result in results if result.answered]

    by_category = {}

    for category in CATEGORIES:
        members = [result for result in results if result.category == category]

        if not members:
            continue

        members_scored = [m for m in members if m.status not in UNSCORED]

        by_category[category] = {
            "cases": len(members),
            "scored": len(members_scored),
            "passed": sum(m.status == PASS for m in members_scored),
            "skipped": sum(m.status == SKIPPED for m in members),
            "errors": sum(m.status == ERROR for m in members),
        }

    # Offline, nothing can be routed to the model, so only cases expected
    # on the rules route are measurable there.
    routable = [
        result
        for result in scored
        if result.expected_route
        and not (offline and result.expected_route == "model")
    ]

    routes: dict[str, int] = {}

    for result in results:
        routes[result.route] = routes.get(result.route, 0) + 1

    model_latencies = [
        result.latency_ms for result in results if result.model_calls
    ]
    input_tokens = sum(result.input_tokens for result in results)
    output_tokens = sum(result.output_tokens for result in results)

    cost = None

    if input_price_per_million is not None and output_price_per_million is not None:
        cost = round(
            input_tokens / 1e6 * input_price_per_million
            + output_tokens / 1e6 * output_price_per_million,
            6,
        )

    return {
        "cases": len(results),
        "scored": len(scored),
        "passed": len(passed),
        "skipped": sum(result.status == SKIPPED for result in results),
        "errors": sum(result.status == ERROR for result in results),
        "pass_rate": round(len(passed) / len(scored), 4) if scored else None,
        "answered": len(answered),
        "confident_wrong": sum(result.confident_wrong for result in results),
        "confident_wrong_rate": (
            round(
                sum(result.confident_wrong for result in answered)
                / len(answered),
                4,
            )
            if answered
            else None
        ),
        "by_category": by_category,
        "routes": routes,
        "route_accuracy": (
            round(
                sum(r.route == r.expected_route for r in routable)
                / len(routable),
                4,
            )
            if routable
            else None
        ),
        "model_calls": sum(result.model_calls for result in results),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": cost,
        "latency_ms_p50_all": percentile(
            [result.latency_ms for result in results], 0.5
        ),
        "latency_ms_p50_model": percentile(model_latencies, 0.5),
        "latency_ms_p95_model": percentile(model_latencies, 0.95),
    }


def recording_invoker(
    chat_model,
    config: llm.LLMConfig,
    completions: list[llm.Completion],
    cache: dict[str, str] | None = None,
    min_interval_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    structured: bool = False
) -> Callable[[str], str]:
    """
    Build an invoke callable that records each call, with optional replay.

    Args:
        chat_model:
            The chat model.

        config:
            Its configuration, for the requested model name.

        completions:
            List every live call is appended to.

        cache:
            Optional mapping of prompt hash to reply. A hit is replayed
            without a call and without being recorded, so re-scoring a run
            costs nothing. Misses are added.

        min_interval_seconds:
            Least time between the starts of two live calls. A free-tier
            key allows a handful of requests a minute, and a run that
            outpaces it measures the quota rather than the model.

        sleep:
            Replaced in tests.

        structured:
            Ask for the reply through the provider's enforced schema
            instead of JSON requested in the prompt.

    Returns:
        The callable.
    """

    last_started = [None]
    mode = llm.PLANNER_OUTPUT_STRUCTURED if structured else llm.PLANNER_OUTPUT_TEXT

    def invoke(prompt: str) -> str:
        key = hashlib.sha256(
            f"{config.model}\n{mode}\n{prompt}".encode()
        ).hexdigest()

        if cache is not None and key in cache:
            return cache[key]

        if min_interval_seconds > 0 and last_started[0] is not None:
            waited = time.monotonic() - last_started[0]

            if waited < min_interval_seconds:
                sleep(min_interval_seconds - waited)

        last_started[0] = time.monotonic()

        if structured:
            completion = llm.complete_structured(
                chat_model,
                prompt,
                planner_schema.PlannerReply,
                planner_schema.reply_to_text,
                config.model,
            )

        else:
            completion = llm.complete(chat_model, prompt, config.model)
        completions.append(completion)

        if cache is not None:
            cache[key] = completion.text

        return completion.text

    return invoke


def run(
    cases: list[dict[str, Any]],
    dataset_root: str,
    invoke: Callable[[str], str] | None = None,
    completions: list[llm.Completion] | None = None
) -> list[CaseResult]:
    """
    Run every case.

    Args:
        cases:
            From load_cases.

        dataset_root:
            Directory each case's dataset path is resolved against.

        invoke:
            Model callable, or None for offline.

        completions:
            The list invoke records into.

    Returns:
        One result per case, in order.
    """

    completions = completions if completions is not None else []
    frames: dict[str, pd.DataFrame] = {}
    glossaries: dict[str, glossary_module.Glossary] = {}
    results = []

    for case in cases:
        dataset = case.get("dataset", "sample_data/store_orders.csv")

        if dataset not in frames:
            frames[dataset] = load_dataset(os.path.join(dataset_root, dataset))

        glossary_path = case.get("glossary")

        if glossary_path and glossary_path not in glossaries:
            with open(
                os.path.join(dataset_root, glossary_path), encoding="utf-8"
            ) as handle:
                glossaries[glossary_path] = glossary_module.Glossary.from_text(
                    handle.read()
                )

        results.append(run_case(
            case,
            frames[dataset],
            invoke,
            completions,
            glossaries.get(glossary_path) if glossary_path else None,
        ))

    return results


def report(
    results: list[CaseResult],
    summary: dict[str, Any],
    metadata: dict[str, Any]
) -> dict[str, Any]:
    """Assemble the JSON report written after a run."""

    return {
        "metadata": metadata,
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
