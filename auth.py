"""
Who may open the app.

Off by default, so a local run and the test suite need nothing. Two ways to
turn it on, checked in this order:

    OIDC single sign-on, through Streamlit's own st.login. Configure an
    [auth] section in .streamlit/secrets.toml (Google, Microsoft Entra,
    Auth0, Okta or any OIDC provider). Optionally restrict who gets in with
    AI_ANALYST_ALLOWED_EMAILS, a comma-separated list of addresses or
    @domains. This is the mode for a real deployment.

    A shared password, from AI_ANALYST_APP_PASSWORD. Enough for a demo link
    shared with a few people. It is compared in constant time, and a
    session is locked out for a minute after five wrong attempts.

Streamlit's session state lives on the server, so a visitor cannot mark
themselves as signed in from the browser.
"""

import hmac
import os
import time

import streamlit as st

PASSWORD_ENV = "AI_ANALYST_APP_PASSWORD"
ALLOWED_EMAILS_ENV = "AI_ANALYST_ALLOWED_EMAILS"

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 60

MODE_NONE = "none"
MODE_OIDC = "oidc"
MODE_PASSWORD = "password"


def oidc_configured() -> bool:
    """Whether secrets.toml has an [auth] section for st.login."""

    try:
        return "auth" in st.secrets

    except Exception:  # noqa: BLE001
        # No secrets file at all is the ordinary local case.
        return False


def login_mode() -> str:
    """Which gate applies: none, oidc or password."""

    if oidc_configured():
        return MODE_OIDC

    if os.environ.get(PASSWORD_ENV, "").strip():
        return MODE_PASSWORD

    return MODE_NONE


def email_allowed(email: str | None, allowed: str | None = None) -> bool:
    """
    Check a signed-in address against the allowlist.

    Args:
        email:
            The address the identity provider reported.

        allowed:
            Comma-separated addresses or @domains. Read from
            AI_ANALYST_ALLOWED_EMAILS when not given. Empty allows anyone
            the provider signed in.

    Returns:
        Whether the address may use the app.
    """

    raw = os.environ.get(ALLOWED_EMAILS_ENV, "") if allowed is None else allowed
    rules = [rule.strip().lower() for rule in raw.split(",") if rule.strip()]

    if not rules:
        return True

    address = (email or "").strip().lower()

    if not address:
        return False

    for rule in rules:
        if rule.startswith("@") and address.endswith(rule):
            return True

        if address == rule:
            return True

    return False


def password_matches(supplied: str, expected: str) -> bool:
    """Compare in constant time, so timing says nothing about the guess."""

    return hmac.compare_digest(
        (supplied or "").encode("utf-8"),
        (expected or "").encode("utf-8"),
    )


def _locked_for(now: float) -> float:
    until = st.session_state.get("auth_locked_until", 0.0)

    return max(0.0, until - now)


def _password_gate() -> bool:
    if st.session_state.get("auth_ok"):
        return True

    st.markdown(
        '<div class="panel-title">Sign in</div>'
        '<p class="panel-note">This workspace is protected with a password.</p>',
        unsafe_allow_html=True,
    )

    now = time.monotonic()
    locked = _locked_for(now)

    supplied = st.text_input("Password", type="password", key="auth_password")

    if st.button("Sign in", key="auth_submit", type="primary", disabled=locked > 0):
        if password_matches(supplied, os.environ.get(PASSWORD_ENV, "").strip()):
            st.session_state.auth_ok = True
            st.session_state.auth_attempts = 0
            st.session_state.pop("auth_password", None)
            st.rerun()

        attempts = st.session_state.get("auth_attempts", 0) + 1
        st.session_state.auth_attempts = attempts

        if attempts >= MAX_ATTEMPTS:
            st.session_state.auth_locked_until = now + LOCKOUT_SECONDS
            st.session_state.auth_attempts = 0
            # Redraw, so the button this run already drew shows as locked.
            st.rerun()

        else:
            st.error("That password is not right.")

    if locked > 0:
        st.warning(
            f"Too many attempts. Try again in {int(locked) + 1} seconds."
        )

    return False


def _oidc_gate() -> bool:
    user = st.user

    if not user.is_logged_in:
        st.markdown(
            '<div class="panel-title">Sign in</div>'
            '<p class="panel-note">Use your organisation account to open this '
            "workspace.</p>",
            unsafe_allow_html=True,
        )
        st.button("Sign in", key="auth_login", type="primary", on_click=st.login)

        return False

    if not email_allowed(user.get("email")):
        st.error(
            "This account is signed in but does not have access to this "
            "workspace."
        )
        st.button("Sign out", key="auth_logout", on_click=st.logout)

        return False

    return True


def require_login() -> bool:
    """
    Show the sign-in screen unless the visitor may use the app.

    Returns:
        True when the app should render. The caller stops otherwise.
    """

    mode = login_mode()

    if mode == MODE_OIDC:
        return _oidc_gate()

    if mode == MODE_PASSWORD:
        return _password_gate()

    return True


def render_sign_out() -> None:
    """A sign-out control, shown only when a gate is active."""

    mode = login_mode()

    if mode == MODE_OIDC and st.user.is_logged_in:
        st.button("Sign out", key="auth_logout_bar", type="tertiary", on_click=st.logout)

    elif (
        mode == MODE_PASSWORD
        and st.session_state.get("auth_ok")
        and st.button("Sign out", key="auth_logout_bar", type="tertiary")
    ):
        st.session_state.auth_ok = False
        st.rerun()
