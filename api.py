"""
An HTTP API over the question-answering path.

    POST /v1/ask        {"dataset": "store_orders.csv", "question": "total revenue"}
    GET  /v1/datasets   the datasets that can be asked about
    GET  /healthz       liveness, no authentication

Only computed answers are served. A question goes through the same rules,
model planner and pandas executor as the chat tab, and the response
carries the plan, the working and the equivalent SQL. The conversational
agent is not exposed: its answers are composed prose, and an API client
is usually a program that will act on the figure.

Security, in the order a request meets it:

    Authentication. Every /v1 route needs a key from AI_ANALYST_API_KEYS
    (comma-separated), sent as "Authorization: Bearer <key>" or
    "X-API-Key: <key>". With no keys configured every /v1 request is
    refused: the API fails closed rather than serving the data openly.

    Rate limiting. AI_ANALYST_API_RATE_LIMIT requests per key per minute,
    30 by default. The counter lives in this process unless
    AI_ANALYST_REDIS_URL is set, in which case every instance shares one
    budget per key through Redis.

    Input. The question is capped at the chat's length limit. The dataset
    is a file name, resolved inside AI_ANALYST_API_DATA_DIR (sample_data
    by default) and refused if it resolves anywhere else.

Run it with:

    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import contextlib
import hashlib
import hmac
import os
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import file_io
import llm
import nlq_answer
import nlq_model
import sql_backend
import telemetry

ROOT = os.path.dirname(os.path.abspath(__file__))

API_KEYS_ENV = "AI_ANALYST_API_KEYS"
RATE_LIMIT_ENV = "AI_ANALYST_API_RATE_LIMIT"
DATA_DIR_ENV = "AI_ANALYST_API_DATA_DIR"
USE_MODEL_ENV = "AI_ANALYST_API_USE_MODEL"
REDIS_URL_ENV = "AI_ANALYST_REDIS_URL"

DEFAULT_RATE_LIMIT = 30
RATE_WINDOW_SECONDS = 60.0
MAX_TABLE_ROWS = 50
DATASET_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,120}$"


def configured_keys() -> list[str]:
    raw = os.environ.get(API_KEYS_ENV, "")

    return [key.strip() for key in raw.split(",") if key.strip()]


def rate_limit() -> int:
    try:
        value = int(os.environ.get(RATE_LIMIT_ENV, "").strip())

    except ValueError:
        return DEFAULT_RATE_LIMIT

    return value if value > 0 else DEFAULT_RATE_LIMIT


def data_dir() -> str:
    return os.path.realpath(
        os.environ.get(DATA_DIR_ENV, "").strip()
        or os.path.join(ROOT, "sample_data")
    )


class RateLimiter:
    """Sliding-window counter per key, for one process."""

    def __init__(self):
        self.calls: dict[str, deque] = {}
        self.lock = threading.Lock()

    def allow(self, key: str, limit: int, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now

        with self.lock:
            window = self.calls.setdefault(key, deque())

            while window and now - window[0] >= RATE_WINDOW_SECONDS:
                window.popleft()

            if len(window) >= limit:
                return False

            window.append(now)

            return True

    def reset(self) -> None:
        with self.lock:
            self.calls.clear()


class RedisRateLimiter:
    """
    A rate limit shared by every instance, kept in Redis.

    Sliding-window counter: one counter per key per minute, and the
    estimate for "the last 60 seconds" is this minute's count plus last
    minute's count weighted by how much of it still falls inside the
    window. INCR is atomic, so two instances cannot both take the last
    slot. The estimate can be off by a few requests at a boundary, which
    is the usual trade for a limiter that needs no Lua and no locks.

    If Redis cannot be reached, requests are allowed and the failure is
    logged: an outage of the limiter should not take the API down with it.
    """

    PREFIX = "ai_analyst:rl"

    def __init__(self, client, window: float = RATE_WINDOW_SECONDS):
        self.client = client
        self.window = window

    def allow(self, key: str, limit: int, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        index = int(now // self.window)
        elapsed = (now % self.window) / self.window
        current = f"{self.PREFIX}:{key}:{index}"
        previous = f"{self.PREFIX}:{key}:{index - 1}"

        try:
            pipe = self.client.pipeline()
            pipe.incr(current)
            pipe.expire(current, int(self.window * 2))
            pipe.get(previous)
            count, _, earlier = pipe.execute()

            estimate = int(count) + int(earlier or 0) * (1 - elapsed)

            if estimate > limit:
                # A refused request does not use up the allowance.
                self.client.decr(current)

                return False

            return True

        except Exception as error:  # noqa: BLE001
            telemetry.logger.warning(
                '{"event": "rate_limiter_unavailable", "type": "%s"}',
                type(error).__name__,
            )

            return True

    def reset(self) -> None:
        for name in self.client.scan_iter(f"{self.PREFIX}:*"):
            self.client.delete(name)


def build_limiter():
    """In-memory by default; Redis when AI_ANALYST_REDIS_URL is set."""

    url = os.environ.get(REDIS_URL_ENV, "").strip()

    if not url:
        return RateLimiter()

    # Imported here, so the dependency is only needed when it is used.
    import redis

    return RedisRateLimiter(redis.Redis.from_url(url, socket_timeout=1.0))


limiter = build_limiter()


def limiter_identity(key: str) -> str:
    """
    A stable, non-secret name for an API key, for the shared limit.

    Derived from the key itself rather than its position in the list, so
    every instance counts the same caller against the same budget even if
    their key lists are ordered differently.
    """

    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def authenticate(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> str:
    """
    Check the caller's key and spend one unit of its rate limit.

    Returns:
        A short, non-secret label for the key, for logs.
    """

    keys = configured_keys()

    if not keys:
        raise HTTPException(
            status_code=503,
            detail=f"The API has no keys configured. Set {API_KEYS_ENV}.",
        )

    supplied = x_api_key or ""

    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()

    # Every configured key is compared, in constant time, so the response
    # time does not say how close a guess was or which key matched.
    matched = None
    identity = None

    for index, key in enumerate(keys):
        if hmac.compare_digest(supplied.encode(), key.encode()):
            matched = f"key{index + 1}"
            identity = limiter_identity(key)

    if matched is None:
        raise HTTPException(
            status_code=401,
            detail="A valid API key is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not limiter.allow(identity, rate_limit()):
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached. Try again in a minute.",
            headers={"Retry-After": str(int(RATE_WINDOW_SECONDS))},
        )

    return matched


class AskRequest(BaseModel):
    dataset: str = Field(pattern=DATASET_NAME_PATTERN)
    question: str = Field(min_length=1, max_length=nlq_model.MAX_QUESTION_CHARS)


def resolve_dataset(name: str) -> str:
    """
    Find a dataset file inside the data directory, refusing anything else.

    Raises:
        HTTPException: 404 when the file does not exist there or is not a
        supported format.
    """

    root = data_dir()
    path = os.path.realpath(os.path.join(root, name))

    if os.path.commonpath([root, path]) != root or path == root:
        raise HTTPException(status_code=404, detail="No such dataset.")

    if not os.path.isfile(path) or not file_io.is_supported(path):
        raise HTTPException(status_code=404, detail="No such dataset.")

    return path


_FRAMES: dict[tuple[str, float], pd.DataFrame] = {}
_FRAMES_LOCK = threading.Lock()
MAX_CACHED_FRAMES = 8


def load_frame(path: str) -> pd.DataFrame:
    """Load a dataset, reusing it until the file changes."""

    key = (path, os.path.getmtime(path))

    with _FRAMES_LOCK:
        if key in _FRAMES:
            return _FRAMES[key]

    frame = file_io.load_tabular_file(path)["dataframe"]

    with _FRAMES_LOCK:
        if len(_FRAMES) >= MAX_CACHED_FRAMES:
            _FRAMES.pop(next(iter(_FRAMES)))

        _FRAMES[key] = frame

    return frame


def json_safe(value: Any) -> Any:
    """Convert numpy and pandas values to plain JSON types."""

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        number = float(value)

        return None if np.isnan(number) or np.isinf(number) else number

    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()

    if value is pd.NA or value is pd.NaT:
        return None

    return value


def model_invoker(sink: list):
    """The planner callable, when a key is set and the API may use it."""

    if os.environ.get(USE_MODEL_ENV, "1").strip() == "0":
        return None

    config = llm.load_config()

    if config is None:
        return None

    # Imported here: the agent module loads the ML stack, which the API
    # only needs when a model is actually configured.
    from agent_data_analysis import plan_invoker

    return plan_invoker(sink=sink)


@contextlib.asynccontextmanager
async def lifespan(_app):
    # Events go to the console when the server runs. Tests that call the
    # app without starting it stay quiet.
    telemetry.configure_console_logging()

    yield


app = FastAPI(
    title="AI Data Analyst API",
    version="1.0.0",
    description="Questions about tabular data, answered by arithmetic.",
    lifespan=lifespan,
)


@app.middleware("http")
async def no_store(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


@app.exception_handler(Exception)
async def unexpected_error(request: Request, error: Exception):
    # Internals stay in the server log; the caller gets a generic message.
    telemetry.logger.error(
        '{"event": "api_error", "type": "%s"}', type(error).__name__
    )

    return JSONResponse(status_code=500, content={"detail": "Internal error."})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/datasets")
def datasets(caller: str = Depends(authenticate)) -> dict[str, list[str]]:
    root = data_dir()
    names = sorted(
        name
        for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name)) and file_io.is_supported(name)
    )

    return {"datasets": names}


@app.post("/v1/ask")
def ask(body: AskRequest, caller: str = Depends(authenticate)) -> dict[str, Any]:
    path = resolve_dataset(body.dataset)
    frame = load_frame(path)

    request_id = telemetry.new_request_id()
    calls: list = []
    started = time.perf_counter()

    result = nlq_answer.answer_question(frame, body.question, model_invoker(calls))

    latency_ms = (time.perf_counter() - started) * 1000
    event = telemetry.question_event(result, latency_ms, calls, request_id)
    event["entry_point"] = "api"
    event["caller"] = caller
    telemetry.emit(event)

    answer = result["answer"]
    plan = result["plan"]

    sql = None

    if plan is not None and answer.success:
        try:
            sql = sql_backend.compile_plan(plan, frame).display()

        except sql_backend.UnsupportedPlan:
            sql = None

    table = None

    if answer.table is not None:
        table = json_safe(answer.table.head(MAX_TABLE_ROWS).to_dict(orient="records"))

    return {
        "request_id": request_id,
        "answered": bool(answer.success),
        "outcome": event["outcome"],
        "planned_by": result["planned_by"],
        "headline": answer.headline if answer.success else answer.error,
        "figures": json_safe(answer.figures),
        "plan": plan.describe() if plan else None,
        "table": table,
        "notes": list(result.get("notes") or []),
        "trail": json_safe(answer.trail),
        "sql": sql,
    }
