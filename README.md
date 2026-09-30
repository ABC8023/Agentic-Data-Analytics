# AI Data Analyst

Ask questions about a CSV or Excel file in plain English, and get answers you can check.

This is a data-analysis app built around one rule: **the language model plans, pandas computes.**

- Most questions are read by a local rules parser and never reach a model.
- A question the parser can't read goes to a model. The model returns a query plan in a fixed vocabulary, not an answer.
- The plan is validated against the dataset and executed locally. Every answer shows the plan, the working, and the equivalent SQL.

**Live application:** [agentic-data-analytics-jckbbdq2bppa2m6sgdycaf.streamlit.app](https://agentic-data-analytics-jckbbdq2bppa2m6sgdycaf.streamlit.app/)  
**GitHub:** [github.com/ABC8023/Agentic-Data-Analytics](https://github.com/ABC8023/Agentic-Data-Analytics)  

---

## Screenshots

| Landing | Dashboard |
|---|---|
| ![Landing page with the upload panel and sample datasets](docs/screenshots/landing.png) | ![Dashboard with the headline finding and four key figures](docs/screenshots/dashboard.png) |
| **Time series** | **AI analysis** |
| ![Quarterly units with trend, expected band and a tested projection](docs/screenshots/timeseries.png) | ![Chat with example questions built from the table's columns](docs/screenshots/ai.png) |

Captured from `sample_data/store_orders.csv` with `tools/screenshot.py`.

---

## How a question is answered

```mermaid
flowchart TD
    Q[Question] --> R{Rules parser<br/>reads every word?}
    R -->|yes| P[Query plan]
    R -->|no, and a key is set| G[Glossary retrieval<br/>BM25, optional embeddings]
    G --> M[Model planner<br/>JSON or enforced schema]
    M --> V{Plan valid<br/>for this dataset?}
    V -->|no| F[One retry with the errors]
    F --> V
    V -->|yes| P
    M -->|off topic or unanswerable| D[Declined, with the reason]
    R -->|no, and no key| D
    P --> X[Executed in pandas]
    X --> A[Answer + plan + working + equivalent SQL]
    V -->|still no| AG[AI analyst agent<br/>tool calls, streamed]
    AG --> T[Answer + calculation trail]
```

| Route | Who planned it | Where the figures come from | Label in the UI |
|---|---|---|---|
| Rules | Local parser | pandas | Read locally |
| Model planner | Gemini, as a JSON plan | pandas | Interpreted by AI |
| AI analyst | Gemini agent with 10 tools | Tool results, composed by the model | Written by AI |

---

## Design decisions

**The model writes plans, not answers or SQL.**
- A plan is a small typed object: intent, measure, aggregation, filters, window and segment.
- `nlq.plan_from_dict` refuses unknown fields.
- `nlq.validate_plan` checks every column and value against the real data.
- A figure in a model reply has no route to the screen, because the reply is only ever read as a plan.

**A half-read question is declined, not guessed.**
- The first evaluation run found the rules parser answering "average age of customers" with the mean revenue, because it read "average" and ignored the rest.
- Any unread word outside a short allowlist now stops the rules. With a key set, the question goes to the model instead. See [Evaluation](#evaluation).

**Untrusted text is quoted.**
- The question, column names, category labels and glossary entries all come from the user or the uploaded file.
- In the prompt they're written as JSON strings under a note that they're data, not instructions.
- Anything that reaches raw HTML is escaped.

**Rows never reach the model.**
- The planner sees column names, kinds and the labels of filterable columns. It sees no figures.
- Every agent tool is wrapped by `privacy.model_safe`, which strips row-level content.
- `tests/test_privacy.py` fails if a new tool skips it.

**Everything a model does is measured.**
- Each question emits one JSON event: route, outcome, latency, tokens, and the model that actually answered.
- The chat shows running totals for the session.
- `AI_ANALYST_MODEL` pins a model, because the default is a rolling alias.

---

## Evaluation

`evals/planner_cases.jsonl` holds 67 questions with known answers, grouped as rules, model, off topic, unanswerable and adversarial. Expected figures were computed separately with plain pandas.

```powershell
python tools/eval_planner.py                     # offline, no key
python tools/eval_planner.py --live --model <m>  # against a model
```

| Offline run | Before | After |
|---|---|---|
| Confident-wrong answers | 13 | 0 |
| Rules cases passed | 28/28 | 28/28 |
| Unanswerable and adversarial cases passed | 7/14 | 14/14 |
| Pass rate | 0.76 | 1.00 |

Live runs on `gemini-3.5-flash-lite`, all 67 cases:

| Live run | Pass rate | Confident-wrong | Input tokens |
|---|---|---|---|
| Text output, before the fix | 63/67 | 4 | 49.7K |
| Text output, after the fix | 66/67 | 0 | 53.3K |
| Structured output | 67/67 | 0 | 84.3K |

- The live runs found two planner bugs, both now fixed and covered by tests:
  - A "can't answer this" reply was executed as a total.
  - "Which territory brought in the most" was planned as a ranking of periods, so the answer named a quarter.
- Enforcing the plan schema through the provider answered one more case, for about 58% more input tokens. Median wait was about 5 seconds per plan either way.

The offline gate runs in CI on every push (`tests/test_evaluation.py`). The live set runs nightly when a key is configured (`.github/workflows/eval.yml`).

More detail is in [evals/README.md](evals/README.md).

---

## Features

**Dataset overview and quality**
- Row and column counts, dtypes and missing values
- Duplicates, constant columns and identifier detection

**Cleaning**
- Duplicate removal, column removal and missing-value strategies
- Preview, apply, restore and download

**Dashboard**
- Headline figures, each shown with its calculation
- Evidence cards with severity, and recommendations
- Change waterfall, segment heatmap and a downloadable brief

**Time series**
- Theil-Sen trend
- Anomaly band calibrated for a 5% family-wise false-alarm rate
- Seasonal projection, backtested against a naive baseline

**Charts**
- Six chart types, with automatic recommendations

**Machine learning**
- Classification or regression recommended from the target
- Three models per task, compared with pipeline preprocessing
- Leaderboards, confusion matrix, permutation importance and downloads

**AI analysis**
- The routing described above
- Streamed agent answers with a calculation trail
- An optional business glossary for the reader's own terms, such as "takings" or "territory"

**SQL**
- Aggregate and breakdown plans compile to parameterised DuckDB SQL (`sql_backend.py`)
- A parity test holds the SQL and pandas results to the same figures

**HTTP API**
- `POST /v1/ask`, with API keys and a per-key rate limit

---

## Project structure

```text
app.py                  Streamlit entry point: sign-in, top bar, loading, tab wiring
app_tabs/               One module per tab, each with a render(...) function
auth.py                 Optional sign-in: OIDC via st.login, or a shared password
ui_theme.py             Theme CSS, chart template, table and chart helpers
ui_cache.py             Memoised profiling for Streamlit reruns
ui_figures.py           Timeline, waterfall and heatmap figures
ui_components.py        Answer, evidence, glossary and usage rendering

nlq.py                  Query plan vocabulary and validation
nlq_rules.py            Rules parser, with the unread-words gate
nlq_model.py            Model planner: prompt, JSON extraction, retry
nlq_answer.py           Router and pandas executor
planner_schema.py       Pydantic schema for enforced structured output
glossary.py             Glossary chunking, BM25, optional embeddings (RRF)
sql_backend.py          Plan to parameterised SQL, run on DuckDB

agent_data_analysis.py  Agent tools, system prompt, model clients, streaming
llm.py                  Model config: name, timeout, retries, usage, embeddings
privacy.py              The row-level boundary around every agent tool
audit.py                Calculation trail from agent tool calls
telemetry.py            One JSON event per question, session totals

dataset_store.py        Session-scoped dataset cache and safe loading
profiling.py            Quality, schema, statistics and date detection
cleaning.py             Cleaning operations
ml.py                   Training pipelines, metrics and feature importance
charts.py               Chart recommendations and Plotly figures
business_insights.py    Dashboard evidence, recommendations and brief
aggregation.py, timeseries.py, anomalies.py, forecasting.py, segments.py,
schema.py, file_io.py, formatting.py

api.py                  FastAPI service, with an in-memory or Redis rate limit
evaluation.py           Evaluation scoring and reporting
evals/                  Golden question set and its notes
tools/eval_planner.py   Evaluation command line
tools/screenshot.py     Headless-Chrome captures for design review
tools/a11y_check.py     axe-core and keyboard checks in headless Chrome
tests/                  1,000+ unit, contract and app tests
sample_data/            Sample datasets, known facts and a glossary
Dockerfile              One image for the app or the API
```

---

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\activate          # macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env           # then set GOOGLE_API_KEY
streamlit run app.py
```

Without a key, everything except open-ended questions still works. Figures, periods, segments and shares are all answered locally.

To try it, load `sample_data/store_orders.csv` and upload `sample_data/store_orders_glossary.md` in the AI analysis tab. Then ask:

```text
total revenue
revenue last month compared with the month before
which region has the highest revenue
what were takings from big orders        (needs a key: glossary + planner)
```

### HTTP API

```powershell
$env:AI_ANALYST_API_KEYS = "generate-a-long-random-key"
uvicorn api:app --port 8000
```

```powershell
curl -X POST http://127.0.0.1:8000/v1/ask `
  -H "Authorization: Bearer generate-a-long-random-key" `
  -H "Content-Type: application/json" `
  -d '{"dataset": "store_orders.csv", "question": "revenue by region"}'
```

The response carries the figures, the plan, the working and the SQL.

- With no keys configured, every `/v1` route returns 503.
- The rate limit is kept in memory by default. Set `AI_ANALYST_REDIS_URL` and every instance shares one budget per key.
- Only computed answers are served. The agent isn't exposed.

### Docker

```powershell
docker build -t ai-data-analyst .
docker run -p 8501:8501 --env-file .env ai-data-analyst
docker run -p 8000:8000 --env-file .env -e AI_ANALYST_API_KEYS=change-me ai-data-analyst uvicorn api:app --host 0.0.0.0 --port 8000
```

### Deploying

On Streamlit Community Cloud:

1. Push the repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), create an app from the repository with `app.py` as the entry point.
3. Under the app's **Secrets**, add `GOOGLE_API_KEY`, and either an `[auth]` block for sign-in (see `.streamlit/secrets.toml.example`) or `AI_ANALYST_APP_PASSWORD`.
4. Put the app's URL at the top of this README.

The Dockerfile works on any container host, such as Cloud Run, Fly.io or Azure Container Apps. Pass secrets as environment variables, never baked into the image.

---

## Configuration

Every setting is optional except the key. All of them are listed in `.env.example`.

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Enables the model planner and the agent |
| `AI_ANALYST_MODEL` | Pin a model. The default, `gemini-flash-latest`, is a rolling alias |
| `AI_ANALYST_LLM_TIMEOUT`, `AI_ANALYST_LLM_MAX_RETRIES` | Per-request timeout (default 60 s) and retries (default 2) |
| `AI_ANALYST_THINKING_BUDGET` | Cap on the model's reasoning tokens |
| `AI_ANALYST_PLANNER_OUTPUT=structured` | Enforce the plan schema through the provider |
| `AI_ANALYST_EMBEDDING_MODEL` | Add embedding retrieval to the glossary. This sends the glossary text to the provider |
| `AI_ANALYST_EVENT_LOG` | Also append question events to a JSON Lines file |
| `AI_ANALYST_LOG_QUESTION_TEXT=1` | Include the question text in events. Off by default |
| `AI_ANALYST_API_KEYS`, `AI_ANALYST_API_RATE_LIMIT`, `AI_ANALYST_API_DATA_DIR`, `AI_ANALYST_API_USE_MODEL` | HTTP API settings |
| `AI_ANALYST_REDIS_URL` | Share the API rate limit across instances |
| `AI_ANALYST_APP_PASSWORD`, `AI_ANALYST_ALLOWED_EMAILS` | Sign-in for the Streamlit app: a shared password, or an allowlist for single sign-on |

---

## Development

```powershell
pip install -r requirements-dev.txt
ruff check .
python -m unittest discover -s tests -t .
python tools/eval_planner.py
```

For design review, `tools/screenshot.py` captures the running app in headless Chrome, for example `python tools/screenshot.py dashboard --query sample=store_orders --full`. Long pages are split into tiles no larger than 1,800 px. The theme itself is set in `.streamlit/config.toml`.

CI runs lint, a byte-compile and the full suite on Python 3.12 and 3.13, with no API key. The suite includes the offline evaluation gate, the privacy contract, and the SQL/pandas parity test.

---

## Security and privacy

- **Secrets:** API keys live in `.env`, which is git-ignored and Docker-ignored.
- **Model boundary:** individual rows are never sent to a model. Identifier-like columns are only ever counted.
- **Untrusted text:** user and file text is quoted in prompts. Anything interpolated into raw HTML is escaped. Questions are capped at 500 characters.
- **Uploads:** uploaded files are size-checked, and Excel macros are never run.
- **API:** dataset names can't leave the data folder.
- **SQL:** SQL is parameterised, and identifiers are allow-listed against the table's columns.

- **Sign-in:** off by default. For a deployment, add an `[auth]` section to `.streamlit/secrets.toml` for single sign-on through `st.login` (Google, Microsoft Entra, Auth0 or any OIDC provider), and optionally restrict it with `AI_ANALYST_ALLOWED_EMAILS`. For a demo link, set `AI_ANALYST_APP_PASSWORD`. See `auth.py`.
- **Accessibility:** `tools/a11y_check.py` reports 0 axe-core WCAG 2.1 AA violations on the landing, dashboard, overview, AI and ML pages, and a visible focus ring at every keyboard stop. Automated checks catch only part of the problem, and the app hasn't been tested with a screen reader.

---

## Limitations

- Model evaluation uses a single train/test split, and hyperparameter search is limited.
- The SQL backend covers aggregate and breakdown plans. Time-series, compare, trend and share plans run in pandas.
- The streamed agent answer was checked by hand against the real model: the first words arrived in 2 to 4 seconds, with two tool calls behind the answer. Only the fake-model tests run automatically.
- Free-tier Gemini quotas are per model and small, often 20 requests a day. Pin a model with spare quota using `AI_ANALYST_MODEL`.
- The Docker image is built and smoke-tested in CI, not locally.

---

## License

MIT. See `LICENCE`.
