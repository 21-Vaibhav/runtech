from datetime import date

from app.data.database import Recommendation, SessionLocal, StateEstimate, init_db
from app.decision.optimizer import recommend_action


def test_decision_constraints_block_hard_and_moderate() -> None:
    init_db()
    db = SessionLocal()
    try:
        today = date.today()
        db.add(
            StateEstimate(
                user_id=1,
                estimate_date=today,
                model_name="kalman",
                fitness=60.0,
                fatigue=70.0,
                form=-10.0,
                fitness_ci_low=55.0,
                fitness_ci_high=65.0,
                fatigue_ci_low=65.0,
                fatigue_ci_high=80.0,
            )
        )
        db.add(
            Recommendation(
                user_id=1,
                recommendation_date=today,
                recommended_action="hard",
                confidence_score=0.8,
                reasoning_dict={},
            )
        )
        db.commit()

        result = recommend_action(db=db, user_id=1, today=today)
        assert result.recommended_action in {"rest", "easy"}
        assert "hard" in result.reasoning_dict["blocked_actions"]
    finally:
        db.close()
