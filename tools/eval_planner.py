"""
Run the golden question set and write a report.

Offline, no key needed, the default:

    python tools/eval_planner.py

Live, against the configured model. Pin a dated model so the result can be
reproduced; the report records the model that actually answered either way:

    python tools/eval_planner.py --live --model gemini-2.5-flash

Replaying a previous live run from its cache costs nothing and re-scores
the same replies, which is how a change to scoring or execution is checked
without paying for the model again:

    python tools/eval_planner.py --live --cache tools/results/eval-cache.json

Exit status is 1 when the run falls below --min-pass-rate or exceeds
--max-confident-wrong, so a workflow can gate on it.
"""

import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import evaluation  # noqa: E402
import llm  # noqa: E402

DEFAULT_CASES = os.path.join(ROOT, "evals", "planner_cases.jsonl")
DEFAULT_OUT = os.path.join(ROOT, "tools", "results")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send questions the rules cannot read to the model."
    )
    parser.add_argument(
        "--model",
        help=f"Model to use. Sets {llm.MODEL_ENV} for this run."
    )
    parser.add_argument(
        "--cache",
        help="JSON file of replies keyed by prompt hash, read and updated."
    )
    parser.add_argument("--input-price", type=float, help="Per 1M input tokens.")
    parser.add_argument("--output-price", type=float, help="Per 1M output tokens.")
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Enforce the reply schema through the provider (planner_schema.py)."
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=0.0,
        help="Seconds between live calls. About 13 suits a 5-per-minute free tier."
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--only", help="Comma-separated categories to run.")
    parser.add_argument("--min-pass-rate", type=float, default=0.0)
    parser.add_argument("--max-confident-wrong", type=int, default=None)

    return parser.parse_args(argv)


def print_summary(results, summary, metadata):
    print(
        f"\nmode={metadata['mode']} model={metadata.get('model_requested')} "
        f"cases={summary['cases']}"
    )

    for category, counts in summary["by_category"].items():
        print(
            f"  {category:<13} {counts['passed']:>3}/{counts['scored']:<3} "
            f"passed   {counts['skipped']} skipped   {counts['errors']} errors"
        )

    if summary["errors"]:
        print(
            f"  {summary['errors']} case(s) could not reach the model and are "
            "not scored. Rerun with --cache and --min-interval to fill them in."
        )

    print(
        f"\n  pass rate            {summary['pass_rate']}\n"
        f"  confident wrong      {summary['confident_wrong']} "
        f"(rate {summary['confident_wrong_rate']} of answered)\n"
        f"  route accuracy       {summary['route_accuracy']}\n"
        f"  routes               {summary['routes']}\n"
        f"  model calls          {summary['model_calls']} "
        f"({summary['input_tokens']} in / {summary['output_tokens']} out tokens)\n"
        f"  latency p50 all      {summary['latency_ms_p50_all']} ms\n"
        f"  latency p50/p95 model {summary['latency_ms_p50_model']} / "
        f"{summary['latency_ms_p95_model']} ms"
    )

    if summary["estimated_cost"] is not None:
        print(f"  estimated cost       {summary['estimated_cost']}")

    failures = [r for r in results if r.status == evaluation.FAIL]

    if failures:
        print("\nFailures:")

        for result in failures:
            flag = " CONFIDENT-WRONG" if result.confident_wrong else ""
            print(f"  {result.id} [{result.route}]{flag} {result.question!r}")

            for reason in result.reasons:
                print(f"      - {reason}")


def main(argv=None) -> int:
    arguments = parse_arguments(argv)

    if arguments.model:
        os.environ[llm.MODEL_ENV] = arguments.model

    cases = evaluation.load_cases(arguments.cases)

    if arguments.only:
        wanted = {name.strip() for name in arguments.only.split(",")}
        cases = [case for case in cases if case["category"] in wanted]

    completions: list[llm.Completion] = []
    invoke = None
    cache = None
    config = None

    if arguments.live:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(ROOT, ".env"))
        config = llm.load_config()

        if config is None:
            print("--live needs GOOGLE_API_KEY or GEMINI_API_KEY.", file=sys.stderr)

            return 2

        if arguments.cache and os.path.exists(arguments.cache):
            with open(arguments.cache, encoding="utf-8") as handle:
                cache = json.load(handle)

        elif arguments.cache:
            cache = {}

        invoke = evaluation.recording_invoker(
            llm.build_chat_model(config),
            config,
            completions,
            cache,
            min_interval_seconds=arguments.min_interval,
            structured=arguments.structured,
        )

    results = evaluation.run(cases, ROOT, invoke, completions)
    summary = evaluation.summarise(
        results,
        offline=invoke is None,
        input_price_per_million=arguments.input_price,
        output_price_per_million=arguments.output_price,
    )

    metadata = {
        "mode": "live" if invoke else "offline",
        "run_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "cases_file": os.path.relpath(arguments.cases, ROOT),
        "model_requested": config.model if config else None,
        "model_is_rolling_alias": config.is_rolling_alias if config else None,
        "models_answered": sorted({c.model for c in completions}),
        "planner_output": (
            ("structured" if arguments.structured else "text") if invoke else None
        ),
        "thinking_budget": config.thinking_budget if config else None,
        "timeout_seconds": config.timeout if config else None,
        "max_retries": config.max_retries if config else None,
        "replayed_from_cache": arguments.cache if cache else None,
    }

    if config and config.is_rolling_alias:
        print(
            f"Note: '{config.model}' is a rolling alias, so this result may "
            "not reproduce. Pin a dated model with --model.",
            file=sys.stderr,
        )

    print_summary(results, summary, metadata)

    os.makedirs(arguments.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(arguments.out, f"eval-{metadata['mode']}-{stamp}.json")

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(evaluation.report(results, summary, metadata), handle, indent=2)

    print(f"\nReport: {os.path.relpath(path, ROOT)}")

    if cache is not None and arguments.cache:
        with open(arguments.cache, "w", encoding="utf-8") as handle:
            json.dump(cache, handle)

    failed_gate = False

    if summary["pass_rate"] is not None and summary["pass_rate"] < arguments.min_pass_rate:
        print(f"Pass rate below {arguments.min_pass_rate}.", file=sys.stderr)
        failed_gate = True

    if (
        arguments.max_confident_wrong is not None
        and summary["confident_wrong"] > arguments.max_confident_wrong
    ):
        print(
            f"More than {arguments.max_confident_wrong} confident-wrong answers.",
            file=sys.stderr,
        )
        failed_gate = True

    return 1 if failed_gate else 0


if __name__ == "__main__":
    sys.exit(main())
