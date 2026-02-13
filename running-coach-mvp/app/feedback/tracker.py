"""Feedback logging and calibration metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.database import CalibrationLog, Recommendation, StateEstimate


@dataclass
class StatsSummary:
    """Feedback quality summary."""

    agreement_rate: float
    adherence_rate: float
    prediction_rmse: float
    overtraining_rate: float
    threshold_adjustment: str | None


def log_feedback(
    db: Session,
    user_id: int,
    feedback_date: date,
    actual_action: str,
    observed_outcome: dict,
) -> CalibrationLog:
    """Persist feedback triple.

    Args:
        db: Database session.
        user_id: User id.
        feedback_date: Date of completed workout.
        actual_action: Executed action.
        observed_outcome: Observed outcomes dict.

    Returns:
        CalibrationLog: Stored feedback row.
    """

    rec = db.scalar(
        select(Recommendation).where(
            Recommendation.user_id == user_id,
            Recommendation.recommendation_date == feedback_date,
        )
    )
    if rec is None:
        raise ValueError("No recommendation found for feedback date")

    row = CalibrationLog(
        user_id=user_id,
        log_date=feedback_date,
        recommended_action=rec.recommended_action,
        actual_action=actual_action,
        observed_outcome=observed_outcome,
    )
    db.add(row)
    return row


def _rmse(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def compute_stats(db: Session, user_id: int, lookback_days: int = 90) -> StatsSummary:
    """Compute calibration statistics.

    Args:
        db: Database session.
        user_id: User id.
        lookback_days: Rolling window length.

    Returns:
        StatsSummary: Aggregate quality metrics.
    """

    since = date.today() - timedelta(days=lookback_days)
    logs = db.scalars(
        select(CalibrationLog)
        .where(CalibrationLog.user_id == user_id, CalibrationLog.log_date >= since)
        .order_by(CalibrationLog.log_date.asc())
    ).all()

    if not logs:
        return StatsSummary(0.0, 0.0, 0.0, 0.0, None)

    agreement = sum(1 for row in logs if row.recommended_action == row.actual_action) / len(logs)
    adherence = agreement

    errors: list[float] = []
    overtraining_count = 0
    for row in logs:
        pred = db.scalar(
            select(StateEstimate)
            .where(StateEstimate.user_id == user_id, StateEstimate.estimate_date == row.log_date, StateEstimate.model_name == "kalman")
        )
        if pred:
            observed_delta = float(row.observed_outcome.get("fatigue_delta", 0.0))
            predicted_delta = float(row.observed_outcome.get("predicted_fatigue_delta", pred.fatigue * 0.01))
            errors.append(observed_delta - predicted_delta)

        if row.observed_outcome.get("overtrained") is True:
            overtraining_count += 1

    overtraining_rate = overtraining_count / len(logs)
    adjustment = "lower_injury_risk_threshold_to_1.4" if overtraining_rate > 0.10 else None

    return StatsSummary(
        agreement_rate=agreement,
        adherence_rate=adherence,
        prediction_rmse=_rmse(errors),
        overtraining_rate=overtraining_rate,
        threshold_adjustment=adjustment,
    )
