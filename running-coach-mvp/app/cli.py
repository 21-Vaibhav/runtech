"""CLI entrypoint for the Running Coach MVP."""

from __future__ import annotations

import json
import logging
import webbrowser
from datetime import datetime, timedelta

import click
from sqlalchemy import select

from app.data.database import SessionLocal, User, UserStateCache, init_db
from app.data.pipeline import sync_activities
from app.data.strava_client import StravaClient
from app.decision.optimizer import recommend_action
from app.feedback.tracker import compute_stats
from app.models.state_estimator import update_state_estimates

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Running coach CLI."""


@cli.command("init")
def init_cmd() -> None:
    """Initialize database schema."""

    init_db()
    click.echo("Database initialized")


@cli.command("auth")
@click.option("--code", default="", help="OAuth callback code")
def auth_cmd(code: str) -> None:
    """Run OAuth flow and store tokens."""

    client = StravaClient()
    url = client.get_auth_url()
    click.echo(f"Open this URL and authorize: {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    if not code:
        code = click.prompt("Paste OAuth code", type=str)

    payload = client.exchange_code(code)
    athlete = payload.get("athlete", {})

    db = SessionLocal()
    try:
        athlete_id = int(athlete["id"])
        user = db.scalar(select(User).where(User.strava_athlete_id == athlete_id))
        if user is None:
            user = User(
                strava_athlete_id=athlete_id,
                username=athlete.get("username"),
                access_token=payload["access_token"],
                refresh_token=payload["refresh_token"],
                token_expires_at=payload["expires_at"],
            )
            db.add(user)
            db.flush()
        else:
            user.access_token = payload["access_token"]
            user.refresh_token = payload["refresh_token"]
            user.token_expires_at = payload["expires_at"]
        db.commit()
        click.echo(f"Authenticated user_id={user.id}")
    finally:
        db.close()


@cli.command("sync")
@click.option("--user-id", required=True, type=int)
@click.option("--days", default=90, type=int)
def sync_cmd(user_id: int, days: int) -> None:
    """Sync activities for a user."""

    db = SessionLocal()
    try:
        result = sync_activities(db=db, user_id=user_id, days_back=days)
        db.commit()
        click.echo(json.dumps({"synced": result.synced_activities, "computed": result.computed_metrics}, indent=2))
    finally:
        db.close()


@cli.command("update-state")
@click.option("--user-id", required=True, type=int)
@click.option("--force", is_flag=True, default=False, help="Force recomputation even if cache is fresh")
def update_state_cmd(user_id: int, force: bool) -> None:
    """Run state estimators for user."""

    db = SessionLocal()
    try:
        cache = db.scalar(select(UserStateCache).where(UserStateCache.user_id == user_id))
        if cache and not force and cache.updated_at >= datetime.utcnow() - timedelta(hours=6):
            click.echo(
                json.dumps(
                    {
                        "status": "skipped",
                        "reason": "state cache is fresh",
                        "updated_at": cache.updated_at.isoformat(),
                    },
                    indent=2,
                )
            )
            return
        result = update_state_estimates(db=db, user_id=user_id)
        db.commit()
        payload = {
            key: {"fitness": v.fitness, "fatigue": v.fatigue, "form": v.form} for key, v in result.items()
        }
        click.echo(json.dumps(payload, indent=2))
    finally:
        db.close()


@cli.command("recommend")
@click.option("--user-id", required=True, type=int)
def recommend_cmd(user_id: int) -> None:
    """Get daily recommendation."""

    db = SessionLocal()
    try:
        rec = recommend_action(db=db, user_id=user_id)
        db.commit()
        click.echo(
            json.dumps(
                {
                    "recommended_action": rec.recommended_action,
                    "confidence_score": rec.confidence_score,
                    "reasoning": rec.reasoning_dict,
                },
                indent=2,
                default=str,
            )
        )
    finally:
        db.close()


@cli.command("stats")
@click.option("--user-id", required=True, type=int)
def stats_cmd(user_id: int) -> None:
    """Show calibration metrics."""

    db = SessionLocal()
    try:
        summary = compute_stats(db=db, user_id=user_id)
        click.echo(
            json.dumps(
                {
                    "agreement_rate": summary.agreement_rate,
                    "adherence_rate": summary.adherence_rate,
                    "prediction_rmse": summary.prediction_rmse,
                    "overtraining_rate": summary.overtraining_rate,
                    "threshold_adjustment": summary.threshold_adjustment,
                },
                indent=2,
            )
        )
    finally:
        db.close()


if __name__ == "__main__":
    cli()
