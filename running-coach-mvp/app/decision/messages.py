"""Hardcoded recommendation and summary copy helpers."""

from __future__ import annotations

from collections import Counter


def recommendation_message(form: float, acwr_value: float, recommended_action: str) -> str:
    """Map state values to actionable recommendation text.

    Args:
        form: Current form value (fitness - fatigue).
        acwr_value: Current acute:chronic workload ratio.
        recommended_action: Optimizer-selected action.

    Returns:
        str: Actionable human-readable recommendation.
    """

    if form < 0 and acwr_value > 1.3:
        return "Rest today. Optional 15-30 min walking, mobility, and hydration focus."
    if form > 1.0 and acwr_value <= 1.3:
        return "Easy Zone 2 session: 30-45 min at conversational effort."
    if -0.5 <= form <= 1.0 and acwr_value <= 1.5:
        return "Moderate session today. Keep effort controlled and stop if HR drifts unusually."
    if recommended_action == "hard":
        return "Hard session allowed. Warm up well and keep total hard volume limited."
    if recommended_action == "moderate":
        return "Moderate intensity session: stay below all-out effort and monitor fatigue."
    if recommended_action == "easy":
        return "Easy day: 25-40 min relaxed movement plus light mobility work."
    return "Recovery day recommended. Prioritize sleep, nutrition, and gentle movement."


def readiness_color(readiness: str) -> str:
    """Return UI color token for readiness badge."""

    mapping = {
        "ready": "green",
        "steady": "yellow",
        "caution": "red",
    }
    return mapping.get(readiness, "yellow")


def weekly_summary_text(sessions: int, total_time_min: float, activity_mix: dict[str, int]) -> str:
    """Build compact weekly narrative summary.

    Example:
        "This week: 3 sessions, 112 min total, 1 run & 2 strength."
    """

    if not activity_mix:
        return f"This week: {sessions} sessions, {round(total_time_min)} min total, no synced activity types yet."

    parts = []
    for name, count in Counter(activity_mix).most_common():
        parts.append(f"{count} {name.lower()}")
    if len(parts) > 1:
        types_text = " & ".join(parts[:2]) if len(parts) == 2 else ", ".join(parts[:2]) + f" +{len(parts) - 2} more"
    else:
        types_text = parts[0]
    return f"This week: {sessions} sessions, {round(total_time_min)} min total, {types_text}."

