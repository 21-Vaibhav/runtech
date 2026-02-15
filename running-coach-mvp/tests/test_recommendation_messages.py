from app.decision.messages import recommendation_message, readiness_color, weekly_summary_text


def test_recommendation_message_recovery_case() -> None:
    message = recommendation_message(form=-1.0, acwr_value=1.4, recommended_action="rest")
    assert "Rest today" in message


def test_recommendation_message_easy_zone2_case() -> None:
    message = recommendation_message(form=1.5, acwr_value=1.1, recommended_action="easy")
    assert "Easy Zone 2" in message


def test_recommendation_message_moderate_case() -> None:
    message = recommendation_message(form=0.4, acwr_value=1.2, recommended_action="moderate")
    assert "Moderate session" in message


def test_readiness_color_mapping() -> None:
    assert readiness_color("ready") == "green"
    assert readiness_color("steady") == "yellow"
    assert readiness_color("caution") == "red"


def test_weekly_summary_text() -> None:
    text = weekly_summary_text(3, 112.4, {"Run": 1, "Strength": 2})
    assert text.startswith("This week: 3 sessions")
    assert "112 min total" in text
    assert "1 run" in text.lower()

