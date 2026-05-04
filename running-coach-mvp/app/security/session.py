"""Stateless JWT session and CSRF helpers for cookie-based auth."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from app.config import settings

SESSION_COOKIE_NAME = "rc_session"
CSRF_COOKIE_NAME = "rc_csrf"
CSRF_HEADER_NAME = "X-CSRF-Token"


def create_session_jwt(user_id: int) -> tuple[str, int]:
    """Create short-lived signed JWT."""

    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=settings.session_ttl_hours)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }
    token = jwt.encode(payload, settings.secret_key, algorithm="HS256")
    return token, int(expires.timestamp())


def decode_session_jwt(token: str) -> int:
    """Validate and decode session JWT."""

    payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    user_id = int(payload["sub"])
    return user_id


def generate_csrf_token() -> str:
    """Generate random CSRF token for double-submit cookie strategy."""

    return secrets.token_urlsafe(32)

