"""Deterministic narrator for explanation text (serverless-safe)."""

from __future__ import annotations

from typing import Any


class LocalNarrator:
    """Rule-based narrator without external model dependencies."""

    def preload(self) -> None:
        """No-op preload for compatibility with existing startup hooks."""

    def explain(self, signals: dict[str, Any]) -> str:
        """Generate a short explanation from structured signals.

        Args:
            signals: Structured signals dictionary.

        Returns:
            str: 2-3 sentence narrative.
        """

        return self._fallback(signals)

    def explain_fast(self, signals: dict[str, Any]) -> str:
        """Return deterministic narrative immediately without model inference."""

        return self._fallback(signals)

    def weekly_summary(self, structured_week: dict[str, Any]) -> str:
        """Generate weekly recap narrative.

        Args:
            structured_week: Weekly aggregates.

        Returns:
            str: Summary text.
        """

        return (
            f"This week included {structured_week.get('sessions', 0)} sessions with ACWR "
            f"at {structured_week.get('acwr', 1.0):.2f}. "
            "Keep easy days easy and prioritize recovery before the next harder effort."
        )

    @staticmethod
    def _fallback(signals: dict[str, Any]) -> str:
        rec = signals.get("recommended_action", "easy")
        form = float(signals.get("form", 0.0))
        acwr_value = float(signals.get("acwr", 1.0))
        return (
            f"Today's recommendation is {rec}. Current form is {form:.1f}, "
            f"and ACWR is {acwr_value:.2f}, so training load is being managed conservatively. "
            "Use easy effort if fatigue or heart-rate drift is higher than usual."
        )
