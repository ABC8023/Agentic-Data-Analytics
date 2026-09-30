# Evaluation

`planner_cases.jsonl` holds 67 questions about `sample_data/store_orders.csv`, each with a known correct outcome. Expected figures were computed separately with plain pandas and match `sample_data/store_orders_facts.txt`.

| Category | Cases | Right outcome |
|---|---|---|
| rules | 28 | Answered locally with the right figure |
| model | 19 | Planned by the model, then answered with the right figure. 5 of these use the reader's own terms, defined in `sample_data/store_orders_glossary.md` |
| off_topic | 6 | Declined, and flagged off topic by the model |
| unanswerable | 6 | Declined, because the data can't answer it |
| adversarial | 8 | No leaked identifiers or prompt text, and no invented figure |

## Running

```powershell
# Offline, no key. CI runs this through tests/test_evaluation.py.
python tools/eval_planner.py

# Live. Pin a dated model so the numbers can be reproduced. --min-interval
# spaces the calls out for a free-tier key.
python tools/eval_planner.py --live --model <dated-model-name> --cache tools/results/eval-cache.json --min-interval 13

# The same questions, with the reply schema enforced by the provider.
python tools/eval_planner.py --live --structured --cache tools/results/eval-cache.json

# Add prices per 1M tokens to get a cost estimate.
python tools/eval_planner.py --live --input-price 0.30 --output-price 2.50
```

Each run writes a JSON report to `tools/results/`. The report records:

- the requested model and the model that actually answered
- the planner output mode and the thinking budget
- per-case latency and token counts
- the reason for every failure

A question the model couldn't be reached for (quota, timeout) is reported as an error. It isn't counted as a failure, because it says nothing about the model's judgement.

## Metrics

- **Confident-wrong:** answered with a figure that is wrong, or answered when it should have declined. This is the number to drive to zero. A refusal costs the reader a rephrase, but a wrong figure that looks right can cost them a decision.
- **Pass rate:** cases that need a model are skipped offline, and model errors are excluded.
- **Route accuracy:** whether rules or the model planned the answer, compared with what the case expects. A right answer by the slower route still passes.
- **Latency and tokens:** p50 and p95 for questions that reached the model.

## What the evaluation found

**Offline baseline:**

- 13 confident-wrong answers out of 46 answered
- pass rate 0.76

The rules parser answered questions it had only partly read. For example, it answered "average age of customers" with the mean revenue, "revenue by product" with total revenue, and "busiest month for units" with total units.

The parser now declines when a word outside a short allowlist goes unread (`HARMLESS_UNREAD_WORDS` in `nlq_rules.py`). With a key set, that question goes to the model planner instead.

**Offline now:**

- 0 confident-wrong
- 28/28 rules cases pass
- every off-topic, unanswerable and adversarial case declined

**First live run**, with the default model alias, which resolved to `gemini-3.8-flash`. It was partial: the free-tier quota of 20 requests a day ran out after 8 model calls.

- 7 of the 8 plans were right.
- The wrong one exposed a design bug. The planner's instruction for "this schema can't answer it" was to reply with a bare aggregate plan, and that plan was executed. So "revenue by product" came back as total revenue. The planner now has an explicit `"answerable": false` reply, which is never executed.
- Plans took 9 to 15 seconds and about 1,000 output tokens each. That points at the model's reasoning phase, which `AI_ANALYST_THINKING_BUDGET` can now limit.

**Full live runs** on `gemini-3.5-flash-lite`, all 67 cases, 40 model calls each:

| Run | Pass rate | Confident-wrong | Route accuracy | Input / output tokens | Median model wait |
|---|---|---|---|---|---|
| Text output, before the fix | 63/67 | 4 | 1.00 | 49,745 / 1,288 | 5.0 s |
| Text output, after the fix | 66/67 | 0 | 0.98 | 53,345 / 1,321 | 5.0 s |
| Structured output, after the fix | 67/67 | 0 | 1.00 | 84,265 / 1,281 | 5.0 s |

- **What the first full run found:** 3 of the 4 wrong answers had the same cause. For "which territory brought in the most", the model planned a ranking of *periods* with a region split attached. The period ranking ignores the split, so the answer named a quarter. The fourth read "how has business changed since last month" as a series, not a comparison.
- **The fix:** `validate_plan` now refuses a segment on any intent that doesn't use one. The refusal names the right intent, so the one retry corrects it. The prompt also says what `extreme` and `compare` are for. That removed all four wrong answers. The one remaining miss is a refusal, not a wrong figure.
- **Text or structured:** enforcing the schema through the provider answered one more case. It cost about 58% more input tokens, because the schema is sent with every request. Latency was the same.
- **Model choice:** the Flash Lite model returned plans in about 5 seconds with around 30 output tokens. The first runs on the default alias took 9 to 15 seconds and about 1,000 tokens.

## Adding a case

Compute the expected figure without the application, for example with a pandas one-liner, so the check is independent of the code under test. A word goes into `HARMLESS_UNREAD_WORDS` only together with a case showing it's safe to leave unread.
