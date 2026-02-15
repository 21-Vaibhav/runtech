"""FastAPI server exposing running coach MVP endpoints."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from app.decision.optimizer import RecommendationResult, recommend_action
from app.feedback.tracker import compute_stats, log_feedback
from app.llm.narrator import LocalNarrator
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


_NARRATOR: LocalNarrator | None = None
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


def get_narrator() -> LocalNarrator:
    global _NARRATOR
    if _NARRATOR is None:
        _NARRATOR = LocalNarrator()
    return _NARRATOR


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
    narrator = get_narrator()
    narrator.preload()
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

    return {"user_id": user.id, "strava_athlete_id": athlete_id}


def _readiness_label(form: float, acwr_value: float, confidence: float) -> str:
    if acwr_value > 1.5 or form < -8:
        return "caution"
    if confidence > 0.75 and form > 2 and acwr_value < 1.2:
        return "ready"
    return "steady"


@app.get("/auth/url")
def auth_url() -> dict[str, str]:
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
                    "OAuth code is expired/used or redirect URI mismatched. "
                    "Start again from /auth/url and complete authorization with the same redirect URI."
                ),
            )
        raise HTTPException(status_code=400, detail=f"OAuth exchange failed: {exc}") from exc
    return _upsert_user_from_token_payload(token_payload, db)


@app.get("/auth/callback")
def auth_callback_get(
    code: str = Query(..., min_length=1),
    state: str | None = Query(default=None, min_length=8, max_length=256),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return auth_callback(AuthCallbackRequest(code=code, state=state), db)


@app.get("/callback")
def callback_alias(
    code: str = Query(..., min_length=1),
    state: str | None = Query(default=None, min_length=8, max_length=256),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return auth_callback(AuthCallbackRequest(code=code, state=state), db)


@app.get("/users")
def list_users(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.scalars(select(User).order_by(User.id.asc())).all()
    return [
        {
            "id": row.id,
            "strava_athlete_id": row.strava_athlete_id,
            "username": row.username,
            "token_expires_at": row.token_expires_at,
        }
        for row in rows
    ]


@app.post("/sync")
def sync(payload: SyncRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
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
def state(user_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
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
def state_update(user_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Explicitly recompute state cache for the user."""

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
    narrator: LocalNarrator,
    state_cache: UserStateCache,
) -> dict[str, Any]:
    signals = {
        "fitness_estimate": (state_cache.fitness, state_cache.fitness_ci_low, state_cache.fitness_ci_high),
        "fatigue_estimate": (state_cache.fatigue, state_cache.fatigue_ci_low, state_cache.fatigue_ci_high),
        "form": result.reasoning_dict.get("form", 0.0),
        "acwr": result.reasoning_dict.get("acwr", 1.0),
        "recommended_action": result.recommended_action,
    }
    explanation = narrator.explain_fast(signals)

    return {
        "recommended_action": result.recommended_action,
        "confidence_score": result.confidence_score,
        "reasoning_dict": result.reasoning_dict,
        "narrative": explanation,
    }


@app.post("/recommend/{user_id}")
async def recommend(
    user_id: int,
    payload: RecommendRequest,
    db: Session = Depends(get_db),
    narrator: LocalNarrator = Depends(get_narrator),
) -> dict[str, Any]:
    """Generate recommendation from cached state and cache result for 6h."""

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
    response = await run_in_threadpool(_build_recommendation_payload, result, narrator, state_cache)

    _REC_CACHE.set(user_id=user_id, day=date.today(), payload=response)
    LOGGER.info("/recommend user=%s cache_hit=false elapsed=%.3fs", user_id, time.perf_counter() - t0)
    return response


@app.post("/feedback")
def feedback(payload: FeedbackRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
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
def stats(user_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
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
    db: Session = Depends(get_db),
    narrator: LocalNarrator = Depends(get_narrator),
) -> dict[str, Any]:
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

    weekly_dict = {
        "sessions": len([r for r in rows if r.start_date >= datetime.utcnow() - timedelta(days=7)]),
        "avg_quality": sum(r.data_quality_score for r in rows) / len(rows) if rows else 0.0,
        "acwr": 1.0,
    }
    weekly = narrator.explain_fast(
        {
            "recommended_action": "easy",
            "form": 0.0,
            "acwr": weekly_dict["acwr"],
        }
    )

    return {"activities": items, "weekly_summary": weekly}


@app.get("/dashboard/{user_id}")
def dashboard(user_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
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
        .where(Recommendation.user_id == user_id, Recommendation.recommendation_date == date.today())
        .order_by(Recommendation.id.desc())
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

    return {
        "today": date.today().isoformat(),
        "weekly": {
            "sessions": len(activities),
            "total_time_min": round(total_time_min, 1),
            "total_distance_km": round(total_distance_km, 2),
            "activity_mix": by_type,
            "streak_days": streak,
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
        "latest_recommendation": (
            {
                "action": rec_row.recommended_action,
                "confidence": rec_row.confidence_score,
                "reasoning": rec_row.reasoning_dict,
            }
            if rec_row
            else None
        ),
        "trends": {
            "acwr": acwr_trend,
            "state": state_trend,
            "weekly_load": weekly_load_trend,
        },
    }
