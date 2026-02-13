"""FastAPI server exposing running coach MVP endpoints."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.database import Activity, SessionLocal, User, init_db
from app.data.pipeline import sync_activities
from app.data.strava_client import StravaClient
from app.decision.optimizer import recommend_action
from app.feedback.tracker import compute_stats, log_feedback
from app.llm.narrator import LocalNarrator
from app.models.state_estimator import update_state_estimates

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Running Coach MVP", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuthCallbackRequest(BaseModel):
    code: str


class SyncRequest(BaseModel):
    user_id: int
    days_back: int = 90


class FeedbackRequest(BaseModel):
    user_id: int
    feedback_date: date
    actual_action: str
    observed_outcome: dict[str, Any] = Field(default_factory=dict)


class RecommendRequest(BaseModel):
    model: str = "kalman"


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


@app.on_event("startup")
def startup() -> None:
    init_db()


def _upsert_user_from_token_payload(token_payload: dict[str, Any], db: Session) -> dict[str, Any]:
    """Create or update local user from Strava token exchange payload."""

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


def _existing_user_fallback(db: Session, note: str) -> dict[str, Any] | None:
    """Return an already linked user when OAuth code is stale/reused."""

    row = db.scalar(select(User).order_by(User.id.asc()))
    if row is None:
        return None
    return {
        "user_id": row.id,
        "strava_athlete_id": row.strava_athlete_id,
        "username": row.username,
        "note": note,
    }


@app.get("/auth/url")
def auth_url() -> dict[str, str]:
    """Get Strava OAuth URL."""

    client = StravaClient()
    return {"url": client.get_auth_url()}


@app.post("/auth/callback")
def auth_callback(payload: AuthCallbackRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Exchange OAuth code and persist user tokens."""

    client = StravaClient()
    try:
        token_payload = client.exchange_code(_extract_code(payload.code))
    except Exception as exc:
        message = str(exc)
        if "invalid_grant" in message.lower():
            fallback = _existing_user_fallback(
                db,
                note=(
                    "The OAuth code is one-time-use and was already consumed. "
                    "Using existing linked user instead."
                ),
            )
            if fallback is not None:
                return fallback
        raise HTTPException(status_code=400, detail=f"OAuth exchange failed: {exc}") from exc

    return _upsert_user_from_token_payload(token_payload, db)


@app.get("/auth/callback")
def auth_callback_get(code: str = Query(...), db: Session = Depends(get_db)) -> dict[str, Any]:
    """Handle OAuth callback from browser query-string flow."""

    client = StravaClient()
    try:
        token_payload = client.exchange_code(_extract_code(code))
    except Exception as exc:
        message = str(exc)
        if "invalid_grant" in message.lower():
            fallback = _existing_user_fallback(
                db,
                note=(
                    "The OAuth code is one-time-use and was already consumed. "
                    "Using existing linked user instead."
                ),
            )
            if fallback is not None:
                return fallback
        raise HTTPException(status_code=400, detail=f"OAuth exchange failed: {exc}") from exc
    return _upsert_user_from_token_payload(token_payload, db)


@app.get("/callback")
def callback_alias(code: str = Query(...), db: Session = Depends(get_db)) -> dict[str, Any]:
    """Compatibility callback route for default redirect URI `/callback`."""

    client = StravaClient()
    try:
        token_payload = client.exchange_code(_extract_code(code))
    except Exception as exc:
        message = str(exc)
        if "invalid_grant" in message.lower():
            fallback = _existing_user_fallback(
                db,
                note=(
                    "The OAuth code is one-time-use and was already consumed. "
                    "Using existing linked user instead."
                ),
            )
            if fallback is not None:
                return fallback
        raise HTTPException(status_code=400, detail=f"OAuth exchange failed: {exc}") from exc
    return _upsert_user_from_token_payload(token_payload, db)


@app.get("/users")
def list_users(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """List known users to simplify selecting user_id in UI."""

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
    """Sync activities and streams for user."""

    try:
        result = sync_activities(db=db, user_id=payload.user_id, days_back=payload.days_back)
    except ValueError as exc:
        message = str(exc)
        if "not found" in message.lower():
            message = (
                f"{message}. Authenticate first via /auth/url and /auth/callback, then retry sync with returned user_id."
            )
        raise HTTPException(status_code=404, detail=message) from exc
    except Exception as exc:
        LOGGER.exception("Sync failed")
        raise HTTPException(status_code=500, detail=f"Sync failed: {exc}") from exc

    return {"synced_activities": result.synced_activities, "computed_metrics": result.computed_metrics}


@app.get("/state/{user_id}")
def state(user_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get current fitness/fatigue/form from both models."""

    try:
        results = update_state_estimates(db, user_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"State update failed: {exc}") from exc

    return {
        model: {
            "fitness": value.fitness,
            "fatigue": value.fatigue,
            "form": value.form,
            "fitness_ci": value.fitness_ci,
            "fatigue_ci": value.fatigue_ci,
        }
        for model, value in results.items()
    }


@app.post("/recommend/{user_id}")
def recommend(user_id: int, payload: RecommendRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Generate today's recommendation."""

    try:
        result = recommend_action(db=db, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    narrator = LocalNarrator()
    signals = {
        "fitness_estimate": (50.0, 45.0, 55.0),
        "fatigue_estimate": (40.0, 34.0, 46.0),
        "form": result.reasoning_dict.get("form", 0.0),
        "acwr": result.reasoning_dict.get("acwr", 1.0),
        "recommended_action": result.recommended_action,
    }
    explanation = narrator.explain(signals)

    return {
        "recommended_action": result.recommended_action,
        "confidence_score": result.confidence_score,
        "reasoning_dict": result.reasoning_dict,
        "narrative": explanation,
    }


@app.post("/feedback")
def feedback(payload: FeedbackRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Log actual action and observed outcome."""

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
    """Get recommendation and prediction quality stats."""

    summary = compute_stats(db, user_id)
    return {
        "agreement_rate": summary.agreement_rate,
        "adherence_rate": summary.adherence_rate,
        "prediction_rmse": summary.prediction_rmse,
        "overtraining_rate": summary.overtraining_rate,
        "threshold_adjustment": summary.threshold_adjustment,
    }


@app.get("/history/{user_id}")
def history(user_id: int, days: int = 30, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get recent activities and weekly LLM summary."""

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
            "distance_m": row.distance_m,
            "moving_time_s": row.moving_time_s,
            "avg_hr": row.average_heartrate,
            "quality": row.data_quality_score,
        }
        for row in rows
    ]

    narrator = LocalNarrator()
    weekly = narrator.weekly_summary(
        {
            "sessions": len([r for r in rows if r.start_date >= datetime.utcnow() - timedelta(days=7)]),
            "avg_quality": sum(r.data_quality_score for r in rows) / len(rows) if rows else 0.0,
            "acwr": 1.0,
        }
    )

    return {"activities": items, "weekly_summary": weekly}
