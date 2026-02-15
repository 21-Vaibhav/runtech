"""FastAPI server exposing running coach MVP endpoints."""

from __future__ import annotations

import logging
import json
import base64
import hashlib
import hmac
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from typing import Literal

from app.config import settings
from app.data.database import (
    Activity,
    ComputedMetric,
    Recommendation,
    SessionLocal,
    StateEstimate,
    User,
    UserStateCache,
    init_db,
)
from app.data.pipeline import sync_activities
from app.data.strava_client import StravaClient
from app.decision.messages import readiness_color, recommendation_message, weekly_summary_text
from app.decision.optimizer import RecommendationResult, recommend_action
from app.feedback.tracker import compute_stats, log_feedback
from app.models.state_estimator import update_state_estimates

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Running Coach MVP", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


class AuthCallbackRequest(BaseModel):
    code: str = Field(min_length=1, max_length=4096)
    state: str | None = Field(default=None, min_length=8, max_length=256)


class SyncRequest(BaseModel):
    user_id: int = Field(gt=0)
    days_back: int = Field(default=90, ge=1, le=365)


class FeedbackRequest(BaseModel):
    user_id: int = Field(gt=0)
    feedback_date: date
    actual_action: Literal["rest", "easy", "moderate", "hard"]
    observed_outcome: dict[str, Any] = Field(default_factory=dict)


class RecommendRequest(BaseModel):
    model: Literal["kalman", "impulse_response"] = "kalman"


@dataclass
class CachedRecommendation:
    payload: dict[str, Any]
    created_at: datetime


class RecommendationCache:
    """Simple in-memory TTL cache for recommendation responses."""

    def __init__(self, ttl_seconds: int = 21600, max_items: int = 1024) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._store: "OrderedDict[tuple[int, str], CachedRecommendation]" = OrderedDict()

    def get(self, user_id: int, day: date) -> dict[str, Any] | None:
        key = (user_id, day.isoformat())
        item = self._store.get(key)
        if item is None:
            return None
        if (datetime.utcnow() - item.created_at).total_seconds() > self.ttl_seconds:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return item.payload

    def set(self, user_id: int, day: date, payload: dict[str, Any]) -> None:
        key = (user_id, day.isoformat())
        self._store[key] = CachedRecommendation(payload=payload, created_at=datetime.utcnow())
        self._store.move_to_end(key)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)


_REC_CACHE = RecommendationCache(ttl_seconds=6 * 60 * 60)


def _extract_code(raw_code: str) -> str:
    text = (raw_code or "").strip()
    if "code=" not in text:
        return text
    if text.startswith("http://") or text.startswith("https://"):
        parsed = urlparse(text)
        return parse_qs(parsed.query).get("code", [""])[0]
    return parse_qs(text.lstrip("?")).get("code", [""])[0]


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _token_secret() -> str:
    return settings.session_secret or settings.strava_client_secret or "running-coach-session-secret"


def _create_session_token(user_id: int, ttl_seconds: int = 60 * 60 * 24 * 14) -> tuple[str, int]:
    expires_at = int(time.time()) + ttl_seconds
    payload = f"{user_id}.{expires_at}"
    sig = hmac.new(_token_secret().encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    token = base64.urlsafe_b64encode(f"{payload}.{sig}".encode("utf-8")).decode("utf-8").rstrip("=")
    return token, expires_at


def _parse_session_token(token: str) -> int:
    try:
        padded = token + "=" * ((4 - len(token) % 4) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8")
        user_s, exp_s, sig = raw.split(".", 2)
        payload = f"{user_s}.{exp_s}"
        expected = hmac.new(_token_secret().encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError("invalid signature")
        if int(exp_s) < int(time.time()):
            raise ValueError("expired token")
        return int(user_s)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Unauthorized") from exc


def require_session_user(authorization: str | None = Header(default=None, alias="Authorization")) -> int:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1].strip()
    return _parse_session_token(token)


def _ensure_same_user(auth_user_id: int, target_user_id: int) -> None:
    if auth_user_id != target_user_id:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Attach baseline security headers and normalized error response."""

    try:
        response = await call_next(request)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Unhandled server error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.on_event("startup")
def startup() -> None:
    try:
        init_db()
    except Exception as exc:
        LOGGER.exception("Database initialization failed during startup: %s", exc)
    LOGGER.info("startup complete")


def _upsert_user_from_token_payload(token_payload: dict[str, Any], db: Session) -> dict[str, Any]:
    athlete = token_payload.get("athlete", {})
    athlete_id = int(athlete.get("id"))
    user = db.scalar(select(User).where(User.strava_athlete_id == athlete_id))
    if user is None:
        user = User(
            strava_athlete_id=athlete_id,
            username=athlete.get("username"),
            access_token=token_payload["access_token"],
            refresh_token=token_payload["refresh_token"],
            token_expires_at=token_payload["expires_at"],
        )
        db.add(user)
        db.flush()
    else:
        user.username = athlete.get("username")
        user.access_token = token_payload["access_token"]
        user.refresh_token = token_payload["refresh_token"]
        user.token_expires_at = token_payload["expires_at"]

    session_token, session_expires_at = _create_session_token(user.id)
    return {
        "user_id": user.id,
        "strava_athlete_id": athlete_id,
        "session_token": session_token,
        "session_expires_at": session_expires_at,
    }


def _readiness_label(form: float, acwr_value: float, confidence: float) -> str:
    if acwr_value > 1.5 or form < -8:
        return "caution"
    if confidence > 0.75 and form > 2 and acwr_value < 1.2:
        return "ready"
    return "steady"


@app.get("/auth/url")
def auth_url() -> dict[str, str]:
    if not settings.strava_client_id or not settings.strava_client_secret:
        raise HTTPException(
            status_code=500,
            detail="Strava credentials missing. Configure STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET.",
        )
    state = StravaClient.create_oauth_state()
    return {"url": StravaClient().get_auth_url(state=state), "state": state}


@app.post("/auth/callback")
def auth_callback(payload: AuthCallbackRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    client = StravaClient()
    if payload.state and not client.validate_oauth_state(payload.state):
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state. Retry authorization.")
    try:
        token_payload = client.exchange_code(_extract_code(payload.code))
    except Exception as exc:
        message = str(exc)
        if "invalid_grant" in message.lower():
            raise HTTPException(
                status_code=400,
                detail=(
                    "OAuth code is invalid/expired/already used, or redirect URI mismatched. "
                    "If you authorized in browser already, callback may have already completed and consumed the code. "
                    "Retry from /auth/url."
                ),
            )
        raise HTTPException(status_code=400, detail=f"OAuth exchange failed: {exc}") from exc
    return _upsert_user_from_token_payload(token_payload, db)


def _oauth_callback_html(payload: dict[str, Any], error: str | None = None) -> HTMLResponse:
    message = {
        "type": "running_coach_oauth_success" if not error else "running_coach_oauth_error",
        "payload": payload,
        "error": error,
    }
    script_data = json.dumps(message)
    html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>RunTech OAuth</title>
    <style>
      body {{ font-family: Segoe UI, Arial, sans-serif; background:#0b1220; color:#e6f0ff; margin:0; display:flex; min-height:100vh; align-items:center; justify-content:center; }}
      .card {{ max-width:560px; border:1px solid #2b4768; border-radius:12px; padding:18px; background:#111d30; }}
      h2 {{ margin:0 0 8px; }}
      p {{ color:#b8cee8; line-height:1.45; }}
      .ok {{ color:#8ff3c4; }}
      .err {{ color:#ffb1bd; }}
      button {{ margin-top:10px; border:none; border-radius:8px; padding:8px 12px; font-weight:700; cursor:pointer; }}
    </style>
  </head>
  <body>
    <div class="card">
      <h2 class="{'ok' if not error else 'err'}">{'Strava connected' if not error else 'OAuth failed'}</h2>
      <p>{'You can close this tab and return to the app.' if not error else error}</p>
      <button onclick="window.location.href='/'">Return to app</button>
    </div>
    <script>
      (function () {{
        const msg = {script_data};
        try {{
          if (window.opener && window.opener !== window) {{
            window.opener.postMessage(msg, window.location.origin);
            setTimeout(() => window.close(), 500);
          }}
        }} catch (_) {{}}
      }})();
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html, status_code=200 if not error else 400)


@app.get("/auth/callback")
def auth_callback_get(
    request: Request,
    code: str = Query(..., min_length=1),
    state: str | None = Query(default=None, min_length=8, max_length=256),
    json_mode: bool = Query(default=False, alias="json"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    try:
        result = auth_callback(AuthCallbackRequest(code=code, state=state), db)
    except HTTPException as exc:
        accepts_html = "text/html" in request.headers.get("accept", "")
        if accepts_html and not json_mode:
            return _oauth_callback_html({}, error=str(exc.detail))
        raise
    accepts_html = "text/html" in request.headers.get("accept", "")
    if accepts_html and not json_mode:
        return _oauth_callback_html(result)
    return result


@app.get("/callback")
def callback_alias(
    request: Request,
    code: str = Query(..., min_length=1),
    state: str | None = Query(default=None, min_length=8, max_length=256),
    json_mode: bool = Query(default=False, alias="json"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return auth_callback_get(request=request, code=code, state=state, json_mode=json_mode, db=db)


@app.get("/users")
def list_users(auth_user_id: int = Depends(require_session_user), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    row = db.get(User, auth_user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    return [
        {
            "id": row.id,
            "strava_athlete_id": row.strava_athlete_id,
            "username": row.username,
            "token_expires_at": row.token_expires_at,
        }
    ]


@app.get("/me")
def me(auth_user_id: int = Depends(require_session_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    row = db.get(User, auth_user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": row.id,
        "strava_athlete_id": row.strava_athlete_id,
        "username": row.username,
        "token_expires_at": row.token_expires_at,
    }


@app.post("/sync")
def sync(payload: SyncRequest, auth_user_id: int = Depends(require_session_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    _ensure_same_user(auth_user_id, payload.user_id)
    t0 = time.perf_counter()
    try:
        result = sync_activities(db=db, user_id=payload.user_id, days_back=payload.days_back)
    except ValueError as exc:
        message = str(exc)
        if "not found" in message.lower():
            message = f"{message}. Authenticate first via /auth/url and /auth/callback."
        raise HTTPException(status_code=404, detail=message) from exc
    except Exception as exc:
        LOGGER.exception("sync failed")
        raise HTTPException(status_code=500, detail=f"Sync failed: {exc}") from exc

    LOGGER.info("/sync user=%s elapsed=%.2fs", payload.user_id, time.perf_counter() - t0)
    return {"synced_activities": result.synced_activities, "computed_metrics": result.computed_metrics}


@app.get("/state/{user_id}")
def state(user_id: int, auth_user_id: int = Depends(require_session_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    _ensure_same_user(auth_user_id, user_id)
    cache = db.scalar(select(UserStateCache).where(UserStateCache.user_id == user_id))
    if cache is None:
        raise HTTPException(status_code=404, detail="State cache missing. Run update-state first.")
    return {
        "as_of": cache.updated_at.isoformat(),
        "model": cache.model_name,
        "fitness": cache.fitness,
        "fatigue": cache.fatigue,
        "form": cache.form,
        "fitness_ci": [cache.fitness_ci_low, cache.fitness_ci_high],
        "fatigue_ci": [cache.fatigue_ci_low, cache.fatigue_ci_high],
        "acwr": cache.acwr,
    }


@app.post("/state/update/{user_id}")
def state_update(user_id: int, auth_user_id: int = Depends(require_session_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    """Explicitly recompute state cache for the user."""
    _ensure_same_user(auth_user_id, user_id)

    t0 = time.perf_counter()
    try:
        result = update_state_estimates(db=db, user_id=user_id)
    except Exception as exc:
        LOGGER.exception("state update failed")
        raise HTTPException(status_code=500, detail=f"State update failed: {exc}") from exc
    LOGGER.info("/state/update user=%s elapsed=%.3fs", user_id, time.perf_counter() - t0)
    return {
        "updated": True,
        "kalman": {
            "fitness": result["kalman"].fitness,
            "fatigue": result["kalman"].fatigue,
            "form": result["kalman"].form,
        },
        "impulse_response": {
            "fitness": result["impulse_response"].fitness,
            "fatigue": result["impulse_response"].fatigue,
            "form": result["impulse_response"].form,
        },
    }


def _build_recommendation_payload(
    result: RecommendationResult,
    state_cache: UserStateCache,
) -> dict[str, Any]:
    form = float(result.reasoning_dict.get("form", 0.0))
    acwr_value = float(result.reasoning_dict.get("acwr", 1.0))
    explanation = recommendation_message(form, acwr_value, result.recommended_action)
    readiness = _readiness_label(
        form=form,
        acwr_value=acwr_value,
        confidence=result.confidence_score,
    )

    return {
        "recommended_action": result.recommended_action,
        "confidence_score": result.confidence_score,
        "reasoning_dict": result.reasoning_dict,
        "narrative": explanation,
        "action_message": explanation,
        "readiness": readiness,
        "readiness_color": readiness_color(readiness),
    }


@app.post("/recommend/{user_id}")
async def recommend(
    user_id: int,
    payload: RecommendRequest,
    auth_user_id: int = Depends(require_session_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Generate recommendation from cached state and cache result for 6h."""
    _ensure_same_user(auth_user_id, user_id)

    t0 = time.perf_counter()
    cache_hit = _REC_CACHE.get(user_id=user_id, day=date.today())
    if cache_hit is not None:
        LOGGER.info("/recommend user=%s cache_hit=true elapsed=%.3fs", user_id, time.perf_counter() - t0)
        return cache_hit

    state_cache = db.scalar(select(UserStateCache).where(UserStateCache.user_id == user_id))
    if state_cache is None:
        raise HTTPException(status_code=404, detail="State cache missing. Run update-state first.")

    # recommend_action is now quick because it reads only cached state.
    result = await run_in_threadpool(recommend_action, db, user_id)
    response = await run_in_threadpool(_build_recommendation_payload, result, state_cache)

    _REC_CACHE.set(user_id=user_id, day=date.today(), payload=response)
    LOGGER.info("/recommend user=%s cache_hit=false elapsed=%.3fs", user_id, time.perf_counter() - t0)
    return response


@app.post("/feedback")
def feedback(payload: FeedbackRequest, auth_user_id: int = Depends(require_session_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    _ensure_same_user(auth_user_id, payload.user_id)
    try:
        log = log_feedback(
            db=db,
            user_id=payload.user_id,
            feedback_date=payload.feedback_date,
            actual_action=payload.actual_action,
            observed_outcome=payload.observed_outcome,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"id": log.id, "recommended_action": log.recommended_action, "actual_action": log.actual_action}


@app.get("/stats/{user_id}")
def stats(user_id: int, auth_user_id: int = Depends(require_session_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    _ensure_same_user(auth_user_id, user_id)
    summary = compute_stats(db, user_id)
    return {
        "agreement_rate": summary.agreement_rate,
        "adherence_rate": summary.adherence_rate,
        "prediction_rmse": summary.prediction_rmse,
        "overtraining_rate": summary.overtraining_rate,
        "threshold_adjustment": summary.threshold_adjustment,
    }


@app.get("/history/{user_id}")
def history(
    user_id: int,
    days: int = 30,
    auth_user_id: int = Depends(require_session_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _ensure_same_user(auth_user_id, user_id)
    since = datetime.utcnow() - timedelta(days=days)
    rows = db.scalars(
        select(Activity)
        .where(Activity.user_id == user_id, Activity.start_date >= since)
        .order_by(Activity.start_date.desc())
    ).all()

    items = [
        {
            "start_date": row.start_date.isoformat(),
            "name": row.name,
            "type": row.type,
            "distance_m": row.distance_m,
            "moving_time_s": row.moving_time_s,
            "avg_hr": row.average_heartrate,
            "quality": row.data_quality_score,
        }
        for row in rows
    ]

    week_rows = [r for r in rows if r.start_date >= datetime.utcnow() - timedelta(days=7)]
    activity_mix: dict[str, int] = {}
    for row in week_rows:
        activity_mix[row.type] = activity_mix.get(row.type, 0) + 1
    total_time_min = sum(r.moving_time_s for r in week_rows) / 60.0
    weekly = weekly_summary_text(len(week_rows), total_time_min, activity_mix)

    return {"activities": items, "weekly_summary": weekly}


@app.get("/dashboard/{user_id}")
def dashboard(user_id: int, auth_user_id: int = Depends(require_session_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    _ensure_same_user(auth_user_id, user_id)
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)

    activities = db.scalars(
        select(Activity)
        .where(Activity.user_id == user_id, Activity.start_date >= week_ago)
        .order_by(Activity.start_date.desc())
    ).all()

    by_type: dict[str, int] = {}
    total_time_min = 0.0
    total_distance_km = 0.0
    for row in activities:
        by_type[row.type] = by_type.get(row.type, 0) + 1
        total_time_min += row.moving_time_s / 60.0
        total_distance_km += row.distance_m / 1000.0

    cache = db.scalar(select(UserStateCache).where(UserStateCache.user_id == user_id))
    acwr_value = cache.acwr if cache else 1.0

    rec_row = db.scalar(
        select(Recommendation)
        .where(Recommendation.user_id == user_id)
        .order_by(Recommendation.recommendation_date.desc(), Recommendation.id.desc())
    )

    activity_days = {row.start_date.date() for row in db.scalars(select(Activity).where(Activity.user_id == user_id)).all()}
    streak = 0
    cursor = date.today()
    while cursor in activity_days:
        streak += 1
        cursor -= timedelta(days=1)

    confidence = rec_row.confidence_score if rec_row else 0.6
    form = cache.form if cache else 0.0

    trend_rows = db.scalars(
        select(ComputedMetric)
        .where(ComputedMetric.user_id == user_id, ComputedMetric.metric_name == "acwr")
        .order_by(ComputedMetric.metric_date.desc())
        .limit(30)
    ).all()
    acwr_trend = [{"date": r.metric_date.isoformat(), "value": r.metric_value} for r in reversed(trend_rows)]

    state_trend_rows = db.scalars(
        select(StateEstimate)
        .where(StateEstimate.user_id == user_id, StateEstimate.model_name == "kalman")
        .order_by(StateEstimate.estimate_date.desc())
        .limit(30)
    ).all()
    state_trend = [
        {"date": r.estimate_date.isoformat(), "fitness": r.fitness, "fatigue": r.fatigue, "form": r.form}
        for r in reversed(state_trend_rows)
    ]

    load_rows = db.scalars(
        select(ComputedMetric)
        .where(ComputedMetric.user_id == user_id, ComputedMetric.metric_name == "session_load")
        .order_by(ComputedMetric.metric_date.desc())
        .limit(56)
    ).all()
    weekly_load_bucket: dict[str, float] = {}
    for row in reversed(load_rows):
        iso_year, iso_week, _ = row.metric_date.isocalendar()
        key = f"{iso_year}-W{iso_week:02d}"
        weekly_load_bucket[key] = weekly_load_bucket.get(key, 0.0) + row.metric_value
    weekly_load_trend = [{"week": k, "value": round(v, 2)} for k, v in weekly_load_bucket.items()]

    latest_recommendation_payload = None
    if rec_row is not None:
        rec_reason = rec_row.reasoning_dict or {}
        rec_form = float(rec_reason.get("form", form))
        rec_acwr = float(rec_reason.get("acwr", acwr_value))
        readiness = _readiness_label(form=rec_form, acwr_value=rec_acwr, confidence=rec_row.confidence_score)
        action_msg = recommendation_message(rec_form, rec_acwr, rec_row.recommended_action)
        latest_recommendation_payload = {
            "date": rec_row.recommendation_date.isoformat(),
            "action": rec_row.recommended_action,
            "confidence": rec_row.confidence_score,
            "reasoning": rec_reason,
            "action_message": action_msg,
            "readiness": readiness,
            "readiness_color": readiness_color(readiness),
            "narrative": action_msg,
        }

    return {
        "today": date.today().isoformat(),
        "weekly": {
            "sessions": len(activities),
            "total_time_min": round(total_time_min, 1),
            "total_distance_km": round(total_distance_km, 2),
            "activity_mix": by_type,
            "streak_days": streak,
            "summary_text": weekly_summary_text(len(activities), total_time_min, by_type),
        },
        "state": (
            {
                "fitness": cache.fitness,
                "fatigue": cache.fatigue,
                "form": cache.form,
                "fitness_ci": [cache.fitness_ci_low, cache.fitness_ci_high],
                "fatigue_ci": [cache.fatigue_ci_low, cache.fatigue_ci_high],
            }
            if cache
            else None
        ),
        "acwr": round(acwr_value, 2),
        "readiness": _readiness_label(form=form, acwr_value=acwr_value, confidence=confidence),
        "latest_recommendation": latest_recommendation_payload,
        "trends": {
            "acwr": acwr_trend,
            "state": state_trend,
            "weekly_load": weekly_load_trend,
        },
    }
