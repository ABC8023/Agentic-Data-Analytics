# Changelog

## 2.1.0

### Added

- Cross-validation in model training. Each model is scored over 5 folds of the training rows by default (3 to 10 folds, stratified for classification), with the mean and spread shown. Models are ranked on this score, and the test rows are only used for the final check (`ml_tuning.py`).
- Optional hyperparameter search. It tries up to 8 settings per model with randomised search inside the same folds, and shows the settings it chose. The page notes that a tuned cross-validated score is slightly optimistic, and the test score isn't.
- A model download: a zip with the fitted pipeline in skops format, a model card (features, date conversions, scores, settings, a training-data fingerprint, library versions and caveats), and a loading guide. `ml_export.load_model` refuses a file with types this app never writes (`ml_export.py`).
- `tools/screenshot.py --click` presses buttons and ticks checkboxes before the capture.
- A captioned demo GIF at the top of the README, recorded from the running app by `tools/record_demo.py`. The script also checks, in a real browser, that the model download arrives.

### Fixed

- The "rows have no usable values" note appeared twice under one answer, because the parser and the executor both reported it.

### Changed

- The leaderboard is ranked on the cross-validated score instead of the test score. Test metrics are labelled "Test" to make the difference clear.
- The agent's training tools report how the models were compared.

## 2.0.0

### Added

- Evaluation harness. It has 67 golden questions with independently computed answers, offline and live modes, a reply cache, request throttling and JSON reports (`evaluation.py`, `tools/eval_planner.py`, `evals/`). The offline gate runs in CI, and a nightly workflow runs the live set.
- Model configuration in one place (`llm.py`): model name, timeout, retries, thinking budget, token usage, and the model that actually answered.
- A JSON event for each question, with route, outcome, latency and tokens, plus session totals in the chat (`telemetry.py`).
- Optional structured output for the planner, with the schema enforced by the provider (`planner_schema.py`).
- Business glossary retrieval for the planner, using BM25 and optionally embeddings fused by reciprocal rank fusion (`glossary.py`).
- Parameterised SQL for aggregate and breakdown plans on DuckDB, with a pandas parity test (`sql_backend.py`). Answers now show the equivalent SQL.
- Streamed agent answers.
- HTTP API with API keys, a per-key rate limit and confined dataset paths (`api.py`).
- Dockerfile and .dockerignore.
- Prompt-injection and adversarial evaluation cases.

### Fixed

- A model plan could attach a segment to an intent that ignores it, so "which territory brought in the most" named a quarter. Such plans are now refused with a message that names the right intent, and the retry fixes them. Live confident-wrong answers went from 4 to 0.
- Empty content chunks streamed around tool calls showed as `[]` in the AI analyst's answer.

- Column names from an uploaded file could inject HTML into the dashboard. Every raw-HTML interpolation is now escaped.
- The rules parser answered questions it had only partly read. For example, "average age of customers" came back as the mean revenue. An unread word outside a small allowlist now declines the question, or passes it to the model planner.
- The planner's "can't answer" reply was executed as a total. It's now an explicit `"answerable": false` and is never executed.
- The model clients had no timeout or retry limit.
- The question, column names and labels went into the prompt as-is. They're now quoted as JSON strings, and questions are capped at 500 characters.
- A pandas 3 deprecation warning in dtype selection.

### Redesigned

- Light analyst theme set natively in `.streamlit/config.toml`: warm paper, one teal accent, Geist and Geist Mono, WCAG AA contrast. The old dark CSS fought Streamlit's light defaults, which left white buttons with invisible labels.
- New landing page: a statement of how answers are made, an upload panel, and one-click sample datasets. `?sample=store_orders` opens a sample from a link.
- A site header: full window width, white with a hairline, and pinned to the top while the page scrolls. It holds the name on the left, and on the right the AI status on the start page, or the open table and "Change dataset" inside one. It stays one line on a phone.
- The name and its line sit together as one lockup. Away from the start page, the name is a button back to it, which closes the table and clears any `?sample=` link.
- The AI tab puts the chat first, offers example questions built from the table's own columns, and moves settings to a side column.
- Dashboard evidence is a two-column grid, sorted by severity.
- Charts use the app's own palette on white, with every chart rendered through one helper.

### Production readiness

- Optional sign-in for the Streamlit app (`auth.py`): single sign-on through `st.login`, with an email or domain allowlist, or a shared password with a session lockout after five wrong attempts. Off unless configured.
- The API rate limit can be shared across instances through Redis (`AI_ANALYST_REDIS_URL`). It uses a sliding-window counter keyed on a hash of the API key.
- A CI job builds the Docker image, checks that the API answers with a key and refuses without one, checks the Streamlit health endpoint, and checks that the container runs as a non-root user.
- Accessibility: 0 axe-core WCAG 2.1 AA violations on five pages, and a visible focus ring at every keyboard stop (`tools/a11y_check.py`). Caption contrast went from 4.35:1 to 5.7:1.
- Computed tables show numbers with thousands separators, and shares as percentages.
- Screenshots in `docs/screenshots/`, and a deploying section in the README.

### Changed

- Every tab moved out of `app.py` into `app_tabs/`, one module each. `app.py` went from 3,415 lines to about 470.
- `agent_data_analysis.py` is split into `dataset_store`, `cleaning`, `ml`, `profiling` and `charts`. The agent module keeps the tools, the prompt and the model clients.
- Theme, memoised profiling, figures and rendering helpers moved out of `app.py` into `ui_*` modules.
- The agent tool `recomending_visualisations` is renamed `recommend_visualisations`.
- CI sets Gemini key variables instead of the stale OpenAI one.

## 1.0.0 — Portfolio release

### Added

- CSV upload and validation, and DataFrame caching
- Dataset overview, data-quality analysis, and numerical and categorical statistics
- Interactive data cleaning
- Custom Plotly visualisations and automatic recommendations
- Classification and regression recommendation, model comparison, confusion matrix, and actual-versus-predicted charts
- Permutation feature importance, and leaderboard and prediction downloads
- LangChain tool-calling agent and the AI Analyst chat
