from datetime import date, timedelta

from app.data.database import ComputedMetric, SessionLocal, init_db
from app.models.state_estimator import run_impulse_response_state


def test_state_estimator_decay_rates() -> None:
    init_db()
    db = SessionLocal()
    try:
        old_day = date.today() - timedelta(days=14)
        db.add(
            ComputedMetric(
                user_id=1,
                activity_id=None,
                metric_date=old_day,
                metric_name="trimp",
                metric_value=100.0,
                metadata_json={},
            )
        )
        db.commit()

        state = run_impulse_response_state(db=db, user_id=1)
        assert state.fitness > state.fatigue
        assert state.fitness_ci[0] < state.fitness < state.fitness_ci[1]
        assert state.fatigue_ci[0] < state.fatigue < state.fatigue_ci[1]
    finally:
        db.close()
