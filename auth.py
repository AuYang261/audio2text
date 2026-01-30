import hmac
import os
from typing import Optional

from fastapi import Request


def _env(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v not in (None, "") else default


APP_USERNAME = _env("APP_USERNAME", "")
APP_PASSWORD = _env("APP_PASSWORD", "")
SESSION_COOKIE_NAME = _env("APP_SESSION_COOKIE", "whisper_session")
# NOTE: This is a simple demo secret; change it for any real deployment.
SESSION_SECRET = _env("APP_SESSION_SECRET", "")
if not APP_USERNAME or not APP_PASSWORD or not SESSION_SECRET:
    raise ValueError(
        f"APP_USERNAME({APP_USERNAME}), APP_PASSWORD({APP_PASSWORD}), and APP_SESSION_SECRET({SESSION_SECRET}) environment variables must be set"
    )


def _sign(username: str) -> str:
    msg = username.encode("utf-8")
    key = SESSION_SECRET.encode("utf-8")
    return hmac.new(key, msg, digestmod="sha256").hexdigest()


def make_session_value(username: str) -> str:
    return f"{username}.{_sign(username)}"


def verify_session_value(value: str) -> Optional[str]:
    try:
        username, sig = value.split(".", 1)
    except ValueError:
        return None

    expected = _sign(username)
    if hmac.compare_digest(sig, expected):
        return username

    return None


def verify_credentials(username: str, password: str) -> bool:
    return hmac.compare_digest(username, APP_USERNAME) and hmac.compare_digest(
        password, APP_PASSWORD
    )


def get_logged_in_user(request: Request) -> Optional[str]:
    v = request.cookies.get(SESSION_COOKIE_NAME)
    if not v:
        return None
    return verify_session_value(v)
