"""
The HTTP API answers with computed figures, and refuses callers without a
key, callers over their limit, and dataset names that leave its folder.
"""

import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import api
import nlq_model

KEY = "test-key-not-secret"
ENVIRONMENT = {
    api.API_KEYS_ENV: f"{KEY},second-test-key",
    api.RATE_LIMIT_ENV: "",
    api.DATA_DIR_ENV: "",
    api.USE_MODEL_ENV: "0",
    "GOOGLE_API_KEY": "",
    "GEMINI_API_KEY": "",
}


class ApiTestCase(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch.dict(os.environ, ENVIRONMENT)
        patcher.start()
        self.addCleanup(patcher.stop)
        api.limiter.reset()
        self.client = TestClient(api.app, raise_server_exceptions=False)

    def ask(self, question, dataset="store_orders.csv", headers=None):
        return self.client.post(
            "/v1/ask",
            json={"dataset": dataset, "question": question},
            headers={"Authorization": f"Bearer {KEY}"} if headers is None else headers,
        )


class HealthTest(ApiTestCase):

    def test_health_needs_no_key(self):
        response = self.client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")


class AuthenticationTest(ApiTestCase):

    def test_no_key_is_refused(self):
        self.assertEqual(self.ask("total revenue", headers={}).status_code, 401)

    def test_a_wrong_key_is_refused(self):
        response = self.ask(
            "total revenue", headers={"Authorization": "Bearer wrong"}
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["WWW-Authenticate"], "Bearer")

    def test_the_x_api_key_header_is_accepted(self):
        response = self.ask("total revenue", headers={"X-API-Key": "second-test-key"})

        self.assertEqual(response.status_code, 200)

    def test_with_no_keys_configured_the_api_fails_closed(self):
        with mock.patch.dict(os.environ, {api.API_KEYS_ENV: ""}):
            response = self.ask("total revenue")

        self.assertEqual(response.status_code, 503)

    def test_listing_datasets_needs_a_key(self):
        self.assertEqual(self.client.get("/v1/datasets").status_code, 401)


class RateLimitTest(ApiTestCase):

    def test_requests_over_the_limit_are_refused(self):
        with mock.patch.dict(os.environ, {api.RATE_LIMIT_ENV: "2"}):
            statuses = [self.ask("total revenue").status_code for _ in range(3)]

        self.assertEqual(statuses, [200, 200, 429])

    def test_each_key_has_its_own_limit(self):
        with mock.patch.dict(os.environ, {api.RATE_LIMIT_ENV: "1"}):
            first = self.ask("total revenue").status_code
            other = self.ask(
                "total revenue", headers={"X-API-Key": "second-test-key"}
            ).status_code

        self.assertEqual((first, other), (200, 200))

    def test_the_window_slides(self):
        limiter = api.RateLimiter()

        self.assertTrue(limiter.allow("k", 1, now=0.0))
        self.assertFalse(limiter.allow("k", 1, now=30.0))
        self.assertTrue(limiter.allow("k", 1, now=61.0))


class FakeRedis:
    """
    The handful of Redis commands the limiter uses, in memory.

    Shared by two limiters below to stand in for two API instances
    talking to one Redis.
    """

    def __init__(self, fail=False):
        self.values = {}
        self.expiry = {}
        self.fail = fail

    def _check(self):
        if self.fail:
            raise ConnectionError("redis is down")

    def incr(self, name):
        self._check()
        self.values[name] = int(self.values.get(name, 0)) + 1

        return self.values[name]

    def decr(self, name):
        self._check()
        self.values[name] = int(self.values.get(name, 0)) - 1

        return self.values[name]

    def expire(self, name, seconds):
        self._check()
        self.expiry[name] = seconds

        return True

    def get(self, name):
        self._check()
        value = self.values.get(name)

        return None if value is None else str(value).encode()

    def scan_iter(self, pattern):
        prefix = pattern.rstrip("*")

        return [name for name in list(self.values) if name.startswith(prefix)]

    def delete(self, name):
        self.values.pop(name, None)

    def pipeline(self):
        client = self

        class Pipeline:

            def __init__(self):
                self.calls = []

            def __getattr__(self, method):
                def queue(*args):
                    self.calls.append((method, args))

                    return self

                return queue

            def execute(self):
                return [getattr(client, m)(*a) for m, a in self.calls]

        return Pipeline()


class RedisRateLimiterTest(unittest.TestCase):

    def test_two_instances_share_one_budget(self):
        shared = FakeRedis()
        first = api.RedisRateLimiter(shared)
        second = api.RedisRateLimiter(shared)
        now = 1_000_000.0

        results = [
            first.allow("caller", 3, now=now),
            second.allow("caller", 3, now=now + 1),
            first.allow("caller", 3, now=now + 2),
            second.allow("caller", 3, now=now + 3),
        ]

        self.assertEqual(results, [True, True, True, False])

    def test_a_refused_request_does_not_use_the_allowance(self):
        shared = FakeRedis()
        limiter = api.RedisRateLimiter(shared)
        now = 1_000_000.0

        for _ in range(5):
            limiter.allow("caller", 2, now=now)

        current = f"{api.RedisRateLimiter.PREFIX}:caller:{int(now // 60)}"

        self.assertEqual(shared.values[current], 2)

    def test_last_minute_counts_less_as_it_slides_out(self):
        shared = FakeRedis()
        limiter = api.RedisRateLimiter(shared)
        start = 1_000_020.0 - (1_000_020.0 % 60)

        for _ in range(4):
            self.assertTrue(limiter.allow("caller", 4, now=start + 1))

        # Early in the next minute nearly all of the last one still counts.
        self.assertFalse(limiter.allow("caller", 4, now=start + 61))
        # Near its end, little of it does.
        self.assertTrue(limiter.allow("caller", 4, now=start + 115))

    def test_each_caller_has_its_own_budget(self):
        limiter = api.RedisRateLimiter(FakeRedis())

        self.assertTrue(limiter.allow("a", 1, now=1_000_000.0))
        self.assertTrue(limiter.allow("b", 1, now=1_000_000.0))
        self.assertFalse(limiter.allow("a", 1, now=1_000_000.0))

    def test_counters_expire(self):
        shared = FakeRedis()
        api.RedisRateLimiter(shared).allow("caller", 5, now=1_000_000.0)

        self.assertTrue(all(seconds == 120 for seconds in shared.expiry.values()))

    def test_an_unreachable_redis_lets_requests_through(self):
        limiter = api.RedisRateLimiter(FakeRedis(fail=True))

        self.assertTrue(limiter.allow("caller", 1))

    def test_the_backend_is_chosen_from_the_environment(self):
        with mock.patch.dict(os.environ, {api.REDIS_URL_ENV: ""}):
            self.assertIsInstance(api.build_limiter(), api.RateLimiter)

        with mock.patch.dict(os.environ, {api.REDIS_URL_ENV: "redis://localhost:6399/0"}):
            self.assertIsInstance(api.build_limiter(), api.RedisRateLimiter)

    def test_the_identity_depends_on_the_key_not_its_position(self):
        self.assertEqual(api.limiter_identity("k"), api.limiter_identity("k"))
        self.assertNotEqual(api.limiter_identity("k"), api.limiter_identity("j"))
        self.assertEqual(len(api.limiter_identity("secret-key")), 16)


class AskTest(ApiTestCase):

    def test_a_total_is_computed(self):
        response = self.ask("total revenue")
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["answered"])
        self.assertEqual(body["planned_by"], "rules")
        self.assertAlmostEqual(body["figures"]["value"], 1513129.01, places=2)
        self.assertIn('SUM(TRY_CAST("revenue" AS DOUBLE))', body["sql"])
        self.assertTrue(body["trail"])

    def test_a_breakdown_returns_a_table(self):
        body = self.ask("revenue by region").json()

        self.assertEqual(body["figures"]["leader"], "north")
        self.assertEqual(body["table"][0]["region"], "north")

    def test_a_half_read_question_is_declined_not_guessed(self):
        body = self.ask("average age of customers").json()

        self.assertFalse(body["answered"])
        self.assertEqual(body["outcome"], "not_understood")
        self.assertIn("'age'", body["headline"])

    def test_row_identifiers_never_appear(self):
        body = self.ask("list all order references")

        self.assertNotIn("ORD-0", body.text)

    def test_an_overlong_question_is_refused(self):
        response = self.ask("x" * (nlq_model.MAX_QUESTION_CHARS + 1))

        self.assertEqual(response.status_code, 422)

    def test_the_datasets_are_listed(self):
        response = self.client.get(
            "/v1/datasets", headers={"Authorization": f"Bearer {KEY}"}
        )

        self.assertIn("store_orders.csv", response.json()["datasets"])
        self.assertNotIn("store_orders_facts.txt", response.json()["datasets"])


class DatasetPathTest(ApiTestCase):

    def test_an_unknown_dataset_is_not_found(self):
        self.assertEqual(self.ask("total revenue", dataset="missing.csv").status_code, 404)

    def test_a_path_outside_the_folder_is_refused(self):
        for name in ("../.env", "..\\.env", "../app.py", "/etc/passwd", "C:\\Windows\\win.ini"):
            with self.subTest(name=name):
                self.assertIn(
                    self.ask("total revenue", dataset=name).status_code, (404, 422)
                )

    def test_a_file_that_is_not_a_table_is_refused(self):
        response = self.ask("total revenue", dataset="store_orders_facts.txt")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
