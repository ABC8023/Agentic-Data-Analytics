"""
The app stays open when no gate is configured, and closed until a visitor
proves they may use it when one is.
"""

import os
import unittest
from unittest import mock

os.environ["GOOGLE_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

from streamlit.testing.v1 import AppTest  # noqa: E402

import auth  # noqa: E402

APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "app.py"
)
PASSWORD = "correct horse battery staple"


def start(**environment):
    values = {auth.PASSWORD_ENV: "", auth.ALLOWED_EMAILS_ENV: ""}
    values.update(environment)
    patcher = mock.patch.dict(os.environ, values)
    patcher.start()

    app = AppTest.from_file(APP_PATH, default_timeout=600)
    app.run()

    return app, patcher


class NoGateTest(unittest.TestCase):

    def test_the_app_opens_without_configuration(self):
        app, patcher = start()
        self.addCleanup(patcher.stop)

        self.assertEqual(auth.login_mode(), auth.MODE_NONE)
        self.assertEqual([t.label for t in app.text_input], [])
        self.assertTrue(any(b.key == "sample_store_orders" for b in app.button))


class PasswordGateTest(unittest.TestCase):

    def test_nothing_renders_before_sign_in(self):
        app, patcher = start(**{auth.PASSWORD_ENV: PASSWORD})
        self.addCleanup(patcher.stop)

        keys = [button.key for button in app.button]

        self.assertEqual(app.text_input[0].label, "Password")
        self.assertNotIn("sample_store_orders", keys)
        self.assertEqual(len(app.tabs), 0)

    def test_a_wrong_password_is_refused(self):
        app, patcher = start(**{auth.PASSWORD_ENV: PASSWORD})
        self.addCleanup(patcher.stop)

        app.text_input[0].set_value("guess")
        app.button(key="auth_submit").click().run()

        self.assertIn("not right", app.error[0].value)
        self.assertTrue(
            "auth_ok" not in app.session_state
            or not app.session_state["auth_ok"]
        )

    def test_the_right_password_opens_the_app(self):
        app, patcher = start(**{auth.PASSWORD_ENV: PASSWORD})
        self.addCleanup(patcher.stop)

        app.text_input[0].set_value(PASSWORD)
        app.button(key="auth_submit").click().run()

        keys = [button.key for button in app.button]

        self.assertTrue(app.session_state["auth_ok"])
        self.assertIn("sample_store_orders", keys)
        self.assertIn("auth_logout_bar", keys)

    def test_repeated_failures_lock_the_session(self):
        app, patcher = start(**{auth.PASSWORD_ENV: PASSWORD})
        self.addCleanup(patcher.stop)

        for _ in range(auth.MAX_ATTEMPTS):
            app.text_input[0].set_value("guess")
            app.button(key="auth_submit").click().run()

        self.assertTrue(app.button(key="auth_submit").disabled)
        self.assertIn("Too many attempts", app.warning[0].value)


class EmailAllowlistTest(unittest.TestCase):

    def test_an_empty_list_allows_any_signed_in_account(self):
        self.assertTrue(auth.email_allowed("someone@example.com", ""))

    def test_an_exact_address_matches_case_insensitively(self):
        self.assertTrue(auth.email_allowed("Ana@Example.com", "ana@example.com"))

    def test_a_domain_rule_matches_the_whole_domain(self):
        self.assertTrue(auth.email_allowed("ana@example.com", "@example.com"))
        self.assertFalse(auth.email_allowed("ana@example.com.evil.io", "@example.com"))

    def test_an_unlisted_or_missing_address_is_refused(self):
        self.assertFalse(auth.email_allowed("ana@other.org", "@example.com"))
        self.assertFalse(auth.email_allowed(None, "@example.com"))


class PasswordCompareTest(unittest.TestCase):

    def test_only_the_exact_password_matches(self):
        self.assertTrue(auth.password_matches("abc", "abc"))
        self.assertFalse(auth.password_matches("abc ", "abc"))
        self.assertFalse(auth.password_matches("", "abc"))


if __name__ == "__main__":
    unittest.main()
