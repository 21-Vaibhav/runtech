"""Strava API client with OAuth, token refresh, and rate-limit backoff."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from app.config import settings
from app.data.database import User

LOGGER = logging.getLogger(__name__)

STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_API_BASE = "https://www.strava.com/api/v3"
SHORT_LIMIT = 200
DAILY_LIMIT = 2000


class StravaClient:
    """Minimal Strava API wrapper.

    Args:
        user: User object containing OAuth tokens.
    """

    def __init__(self, user: User | None = None) -> None:
        self.user = user

    def get_auth_url(self, state: str = "running-coach") -> str:
        """Build OAuth URL.

        Args:
            state: CSRF state value.

        Returns:
            str: Authorization URL.
        """

        return (
            f"{STRAVA_AUTH_URL}?client_id={settings.strava_client_id}&response_type=code"
            f"&redirect_uri={settings.strava_redirect_uri}&approval_prompt=auto"
            f"&scope=read,activity:read_all&state={state}"
        )

    def exchange_code(self, code: str) -> dict[str, Any]:
        """Exchange OAuth code for tokens.

        Args:
            code: OAuth authorization code.

        Returns:
            dict[str, Any]: Token payload.
        """

        normalized_code = self._normalize_code(code)
        if not settings.strava_client_id or not settings.strava_client_secret:
            raise ValueError("Missing STRAVA_CLIENT_ID or STRAVA_CLIENT_SECRET in environment")
        if not normalized_code:
            raise ValueError("OAuth code is empty or invalid")

        payload = {
            "client_id": settings.strava_client_id,
            "client_secret": settings.strava_client_secret,
            "code": normalized_code,
            "grant_type": "authorization_code",
            "redirect_uri": settings.strava_redirect_uri,
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(STRAVA_TOKEN_URL, data=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text
                raise ValueError(f"Strava token exchange failed ({response.status_code}): {detail}") from exc
            return response.json()

    @staticmethod
    def _normalize_code(raw_code: str) -> str:
        """Extract OAuth code from plain token or callback URL."""

        candidate = (raw_code or "").strip()
        if "code=" not in candidate:
            return candidate
        if candidate.startswith("http://") or candidate.startswith("https://"):
            parsed = urlparse(candidate)
            return parse_qs(parsed.query).get("code", [""])[0]
        parsed = parse_qs(candidate.lstrip("?"))
        return parsed.get("code", [""])[0]

    def refresh_access_token(self) -> dict[str, Any]:
        """Refresh expired access token.

        Returns:
            dict[str, Any]: Refreshed token payload.
        """

        if self.user is None:
            raise ValueError("User is required for token refresh")

        payload = {
            "client_id": settings.strava_client_id,
            "client_secret": settings.strava_client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self.user.refresh_token,
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(STRAVA_TOKEN_URL, data=payload)
            response.raise_for_status()
            return response.json()

    def _needs_refresh(self) -> bool:
        if self.user is None:
            return False
        now = datetime.now(timezone.utc).timestamp()
        return self.user.token_expires_at <= int(now + 60)

    def _rate_limit_backoff(self, response: httpx.Response) -> None:
        usage = response.headers.get("X-RateLimit-Usage", "0,0")
        limits = response.headers.get("X-RateLimit-Limit", f"{SHORT_LIMIT},{DAILY_LIMIT}")
        try:
            short_usage, daily_usage = [int(v) for v in usage.split(",")]
            short_limit, daily_limit = [int(v) for v in limits.split(",")]
        except ValueError:
            return

        if short_usage >= short_limit - 2:
            LOGGER.warning("Approaching 15-min rate limit. Backing off 90 seconds.")
            time.sleep(90)
        if daily_usage >= daily_limit - 5:
            LOGGER.warning("Approaching daily rate limit. Backing off 10 minutes.")
            time.sleep(600)

    def api_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Authenticated GET with auto refresh and rate-limit handling.

        Args:
            path: API path after `/api/v3`.
            params: Optional query parameters.

        Returns:
            Any: Decoded JSON response.
        """

        if self.user is None:
            raise ValueError("User is required for API requests")

        if self._needs_refresh():
            refreshed = self.refresh_access_token()
            self.user.access_token = refreshed["access_token"]
            self.user.refresh_token = refreshed["refresh_token"]
            self.user.token_expires_at = refreshed["expires_at"]

        headers = {"Authorization": f"Bearer {self.user.access_token}"}
        url = f"{STRAVA_API_BASE}{path}"
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=headers, params=params)
            if response.status_code == 429:
                LOGGER.warning("Strava returned 429; backing off 60 seconds")
                time.sleep(60)
                response = client.get(url, headers=headers, params=params)
            response.raise_for_status()
            self._rate_limit_backoff(response)
            return response.json()

    def list_activities(self, after_epoch: int, page: int = 1, per_page: int = 100) -> list[dict[str, Any]]:
        """List activities.

        Args:
            after_epoch: Start timestamp.
            page: Pagination page.
            per_page: Page size.

        Returns:
            list[dict[str, Any]]: Activity list.
        """

        return self.api_get(
            "/athlete/activities",
            params={"after": after_epoch, "page": page, "per_page": per_page},
        )

    def get_activity_streams(self, activity_id: int) -> dict[str, Any]:
        """Fetch streams for one activity.

        Args:
            activity_id: Strava activity id.

        Returns:
            dict[str, Any]: Stream payload.
        """

        keys = "time,heartrate,velocity_smooth,altitude,cadence,latlng"
        return self.api_get(
            f"/activities/{activity_id}/streams",
            params={"keys": keys, "key_by_type": "true"},
        )


