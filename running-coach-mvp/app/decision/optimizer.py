"""Constraint-based recommendation optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.data.database import Race, Recommendation, StateEstimate, UserStateCache
from app.models.metrics import detect_periodization_phase, polarized_distribution

ACTIONS = ["rest", "easy", "moderate", "hard"]
ACTION_EFFECTS = {
    "rest": {"fitness_gain": -0.2, "fatigue_penalty": -2.0, "injury_risk": 0.0},
    "easy": {"fitness_gain": 0.4, "fatigue_penalty": -0.5, "injury_risk": 0.1},
    "moderate": {"fitness_gain": 0.8, "fatigue_penalty": 1.0, "injury_risk": 0.4},
    "hard": {"fitness_gain": 1.2, "fatigue_penalty": 2.0, "injury_risk": 0.8},
}
FATIGUE_UPPER_THRESHOLD = 75.0


@dataclass
class RecommendationResult:
    """Recommendation output object."""

    recommended_action: str
    confidence_score: float
    reasoning_dict: dict


def _latest_state_cache(db: Session, user_id: int) -> UserStateCache | None:
    cache = db.scalar(select(UserStateCache).where(UserStateCache.user_id == user_id))
    if cache is not None:
        return cache

    # Backward-compatible fallback for older records/tests that only have state_estimates.
    state = db.scalar(
        select(StateEstimate)
        .where(StateEstimate.user_id == user_id, StateEstimate.model_name == "kalman")
        .order_by(StateEstimate.estimate_date.desc())
    )
    if state is None:
        return None

    return UserStateCache(
        user_id=user_id,
        model_name="kalman",
        fitness=state.fitness,
        fatigue=state.fatigue,
        form=state.form,
        fitness_ci_low=state.fitness_ci_low,
        fitness_ci_high=state.fitness_ci_high,
        fatigue_ci_low=state.fatigue_ci_low,
        fatigue_ci_high=state.fatigue_ci_high,
        acwr=1.0,
    )


def _days_to_next_race(db: Session, user_id: int, today: date) -> int | None:
    race = db.scalar(
        select(Race)
        .where(and_(Race.user_id == user_id, Race.race_date >= today))
        .order_by(Race.race_date.asc())
    )
    return (race.race_date - today).days if race else None


def recommend_action(db: Session, user_id: int, today: date | None = None) -> RecommendationResult:
    """Generate daily action under hard constraints and soft utility.

    Args:
        db: Database session.
        user_id: User id.
        today: Optional recommendation date.

    Hard constraints implemented:
        1) Block hard if fatigue CI upper exceeds threshold.
        2) Block hard/moderate after a hard/moderate previous day.
        3) Max hard sessions per week.
        4) 14-day taper and no hard/moderate in final 7 days pre-race.
        5) Block hard/moderate when ACWR > injury threshold.

    Returns:
        RecommendationResult: Action and confidence with reasoning.
    """

    today = today or date.today()
    state = _latest_state_cache(db, user_id)
    if state is None:
        raise ValueError("State cache not available. Run update-state first.")

    acwr_value = state.acwr
    days_to_race = _days_to_next_race(db, user_id, today)
    phase = detect_periodization_phase(days_to_race, acwr_value)

    recent = db.scalars(
        select(Recommendation)
        .where(Recommendation.user_id == user_id, Recommendation.recommendation_date >= today - timedelta(days=13))
        .order_by(Recommendation.recommendation_date.asc())
    ).all()
    action_history = [row.recommended_action for row in recent]

    yesterday = db.scalar(
        select(Recommendation).where(
            Recommendation.user_id == user_id,
            Recommendation.recommendation_date == today - timedelta(days=1),
        )
    )

    last_week = db.scalars(
        select(Recommendation).where(
            Recommendation.user_id == user_id,
            Recommendation.recommendation_date >= today - timedelta(days=6),
            Recommendation.recommendation_date <= today,
        )
    ).all()
    hard_runs_week = sum(1 for item in last_week if item.recommended_action == "hard")

    blocked: set[str] = set()
    hard_reasons: list[str] = []

    if state.fatigue_ci_high > FATIGUE_UPPER_THRESHOLD:
        blocked.add("hard")
        hard_reasons.append("fatigue_ci_upper_above_threshold")
    if yesterday and yesterday.recommended_action in {"hard", "moderate"}:
        blocked.update({"hard", "moderate"})
        hard_reasons.append("no_back_to_back_intensity")
    if hard_runs_week >= settings.max_hard_runs_per_week:
        blocked.add("hard")
        hard_reasons.append("max_hard_runs_per_week")
    if days_to_race is not None and days_to_race <= 14:
        blocked.add("hard")
        hard_reasons.append("taper_14_day_rule")
        if days_to_race <= 7:
            blocked.add("moderate")
            hard_reasons.append("no_hard_last_7_days")
    if acwr_value > settings.injury_risk_threshold:
        blocked.update({"hard", "moderate"})
        hard_reasons.append("acwr_above_1_5")

    easy_pct, hard_pct = polarized_distribution(action_history[-20:])
    if hard_pct > 0.2:
        blocked.update({"hard", "moderate"})
        hard_reasons.append("polarized_80_20_enforced")

    scores: dict[str, float] = {}
    form = state.form
    efficiency_declining = form < -5

    for action in ACTIONS:
        if action in blocked:
            continue
        obj = ACTION_EFFECTS[action]["fitness_gain"] - ACTION_EFFECTS[action]["fatigue_penalty"] - ACTION_EFFECTS[action]["injury_risk"]

        if form > 10 and action in {"moderate", "hard"}:
            obj += 0.4
        if form < -10 and action in {"rest", "easy"}:
            obj += 0.5
        if efficiency_declining and action == "rest":
            obj += 0.3
        if phase == "taper" and action in {"moderate", "hard"}:
            obj -= 0.8
        if phase == "peak" and action == "hard":
            obj += 0.2

        scores[action] = obj

    if not scores:
        best = "rest"
    else:
        best = max(scores, key=scores.get)

    confidence = 0.85
    confidence -= min(0.35, 0.08 * len(hard_reasons))
    confidence = max(0.2, min(0.95, confidence))

    reasoning = {
        "objective": "fitness_gain - fatigue_penalty - injury_risk",
        "acwr": acwr_value,
        "form": form,
        "phase": phase,
        "days_to_race": days_to_race,
        "blocked_actions": sorted(blocked),
        "hard_constraint_reasons": hard_reasons,
        "scores": scores,
        "polarized_easy_pct": easy_pct,
        "polarized_hard_pct": hard_pct,
    }

    existing_today = db.scalar(
        select(Recommendation).where(
            Recommendation.user_id == user_id,
            Recommendation.recommendation_date == today,
        )
    )
    if existing_today is not None:
        db.delete(existing_today)

    db.add(
        Recommendation(
            user_id=user_id,
            recommendation_date=today,
            recommended_action=best,
            confidence_score=confidence,
            reasoning_dict=reasoning,
        )
    )

    return RecommendationResult(best, confidence, reasoning)
