"""State estimation models: Kalman + Banister impulse-response with cache writes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.data.database import ComputedMetric, StateEstimate, UserStateCache
from app.models.confidence import confidence_interval


@dataclass
class StateResult:
    """Unified state estimate output."""

    model_name: str
    fitness: float
    fatigue: float
    form: float
    fitness_ci: tuple[float, float]
    fatigue_ci: tuple[float, float]


def _daily_load(db: Session, user_id: int) -> dict[date, float]:
    rows = db.scalars(
        select(ComputedMetric)
        .where(ComputedMetric.user_id == user_id, ComputedMetric.metric_name == "session_load")
        .order_by(ComputedMetric.metric_date.asc())
    ).all()
    if not rows:
        rows = db.scalars(
            select(ComputedMetric)
            .where(ComputedMetric.user_id == user_id, ComputedMetric.metric_name == "trimp")
            .order_by(ComputedMetric.metric_date.asc())
        ).all()

    out: dict[date, float] = {}
    for row in rows:
        out[row.metric_date] = out.get(row.metric_date, 0.0) + row.metric_value
    return out


def _date_range(loads: dict[date, float]) -> list[date]:
    if not loads:
        return [datetime.now(UTC).date()]
    min_day = min(loads.keys())
    max_day = max(max(loads.keys()), datetime.now(UTC).date())
    days: list[date] = []
    cur = min_day
    while cur <= max_day:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def run_kalman_state(db: Session, user_id: int) -> StateResult:
    """Estimate state using 2D Kalman update.

    Equations:
        x_t^- = A x_{t-1} + B u_t
        P_t^- = A P_{t-1} A^T + Q
        K_t = P_t^- (P_t^- + R)^{-1}
        x_t = x_t^- + K_t (z_t - x_t^-)

    State dimensions represent fitness and fatigue driven by daily load observations.
    """

    loads = _daily_load(db, user_id)
    days = _date_range(loads)

    tau_fit = float(settings.fitness_decay_days)
    tau_fat = float(settings.fatigue_decay_days)
    a_fit = math.exp(-1.0 / tau_fit)
    a_fat = math.exp(-1.0 / tau_fat)

    x_fit, x_fat = 50.0, 35.0
    p_fit, p_fat = 10.0, 10.0
    q_fit, q_fat = 1.2, 1.8
    r_fit, r_fat = 5.0, 5.0

    latest_day = days[-1]
    for day in days:
        load = loads.get(day, 0.0)

        x_fit = a_fit * x_fit + (1.0 - a_fit) * load
        x_fat = a_fat * x_fat + (1.0 - a_fat) * load
        p_fit = a_fit * p_fit * a_fit + q_fit
        p_fat = a_fat * p_fat * a_fat + q_fat

        z_fit, z_fat = load, load
        k_fit = p_fit / (p_fit + r_fit)
        k_fat = p_fat / (p_fat + r_fat)
        x_fit = x_fit + k_fit * (z_fit - x_fit)
        x_fat = x_fat + k_fat * (z_fat - x_fat)
        p_fit = (1.0 - k_fit) * p_fit
        p_fat = (1.0 - k_fat) * p_fat

    fit_ci = confidence_interval(x_fit, p_fit)
    fat_ci = confidence_interval(x_fat, p_fat)
    form = x_fit - x_fat

    db.execute(delete(StateEstimate).where(StateEstimate.user_id == user_id, StateEstimate.model_name == "kalman"))
    db.add(
        StateEstimate(
            user_id=user_id,
            estimate_date=latest_day,
            model_name="kalman",
            fitness=x_fit,
            fatigue=x_fat,
            form=form,
            fitness_ci_low=fit_ci[0],
            fitness_ci_high=fit_ci[1],
            fatigue_ci_low=fat_ci[0],
            fatigue_ci_high=fat_ci[1],
        )
    )
    return StateResult("kalman", x_fit, x_fat, form, fit_ci, fat_ci)


def run_impulse_response_state(db: Session, user_id: int) -> StateResult:
    """Estimate state with Banister impulse-response model.

    Equations:
        fitness(t) = sum_i(load_i * exp(-delta_i / tau_fit))
        fatigue(t) = sum_i(load_i * exp(-delta_i / tau_fat))
        form(t) = fitness(t) - fatigue(t)
    """

    loads = _daily_load(db, user_id)
    days = _date_range(loads)

    tau_fit = float(settings.fitness_decay_days)
    tau_fat = float(settings.fatigue_decay_days)

    latest_day = days[-1]
    fit = 0.0
    fat = 0.0
    for load_day, load in loads.items():
        delta = (latest_day - load_day).days
        if delta < 0:
            continue
        fit += load * math.exp(-delta / tau_fit)
        fat += load * math.exp(-delta / tau_fat)

    fit = max(0.0, fit)
    fat = max(0.0, fat)
    form = fit - fat

    fit_ci = confidence_interval(fit, max(4.0, fit * 0.1))
    fat_ci = confidence_interval(fat, max(4.0, fat * 0.12))

    db.execute(delete(StateEstimate).where(StateEstimate.user_id == user_id, StateEstimate.model_name == "impulse_response"))
    db.add(
        StateEstimate(
            user_id=user_id,
            estimate_date=latest_day,
            model_name="impulse_response",
            fitness=fit,
            fatigue=fat,
            form=form,
            fitness_ci_low=fit_ci[0],
            fitness_ci_high=fit_ci[1],
            fatigue_ci_low=fat_ci[0],
            fatigue_ci_high=fat_ci[1],
        )
    )

    return StateResult("impulse_response", fit, fat, form, fit_ci, fat_ci)


def _latest_acwr(db: Session, user_id: int) -> float:
    row = db.scalar(
        select(ComputedMetric)
        .where(ComputedMetric.user_id == user_id, ComputedMetric.metric_name == "acwr")
        .order_by(ComputedMetric.metric_date.desc())
    )
    return row.metric_value if row else 1.0


def update_state_estimates(db: Session, user_id: int) -> dict[str, StateResult]:
    """Run full state pipeline and update cache table.

    This function should be called by the explicit update-state workflow.
    /recommend reads only from cache to minimize latency.
    """

    kalman = run_kalman_state(db, user_id)
    impulse = run_impulse_response_state(db, user_id)
    acwr_value = _latest_acwr(db, user_id)

    cache = db.scalar(select(UserStateCache).where(UserStateCache.user_id == user_id))
    if cache is None:
        cache = UserStateCache(
            user_id=user_id,
            updated_at=datetime.utcnow(),
            model_name="kalman",
            fitness=kalman.fitness,
            fatigue=kalman.fatigue,
            form=kalman.form,
            fitness_ci_low=kalman.fitness_ci[0],
            fitness_ci_high=kalman.fitness_ci[1],
            fatigue_ci_low=kalman.fatigue_ci[0],
            fatigue_ci_high=kalman.fatigue_ci[1],
            acwr=acwr_value,
        )
        db.add(cache)
    else:
        cache.updated_at = datetime.utcnow()
        cache.model_name = "kalman"
        cache.fitness = kalman.fitness
        cache.fatigue = kalman.fatigue
        cache.form = kalman.form
        cache.fitness_ci_low = kalman.fitness_ci[0]
        cache.fitness_ci_high = kalman.fitness_ci[1]
        cache.fatigue_ci_low = kalman.fatigue_ci[0]
        cache.fatigue_ci_high = kalman.fatigue_ci[1]
        cache.acwr = acwr_value

    return {"kalman": kalman, "impulse_response": impulse}
