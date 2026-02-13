"""Local-only LLM narrator for explanation text."""

from __future__ import annotations

import logging
from typing import Any

from transformers import pipeline

from app.config import settings

LOGGER = logging.getLogger(__name__)


class LocalNarrator:
    """Wrapper around local HuggingFace model with rule-based fallback."""

    def __init__(self) -> None:
        self._generator = None
        try:
            # Do not use device_map="auto" to avoid hard dependency on accelerate.
            # The narrator is optional and must never break API behavior.
            self._generator = pipeline(
                "text-generation",
                model=settings.llm_model,
            )
        except Exception as exc:
            LOGGER.warning("Could not initialize local model; fallback will be used: %s", exc)

    def explain(self, signals: dict[str, Any]) -> str:
        """Generate a short explanation from structured signals.

        Args:
            signals: Structured signals dictionary.

        Returns:
            str: 2-3 sentence narrative.
        """

        if self._generator is None:
            return self._fallback(signals)

        prompt = (
            "You are a running coach assistant. Write 2-3 concise sentences. "
            "Use only these structured values and do not invent data: "
            f"{signals}."
        )
        try:
            out = self._generator(prompt, max_new_tokens=90, do_sample=False)
            text = out[0]["generated_text"].replace(prompt, "").strip()
            if not text:
                return self._fallback(signals)
            return " ".join(text.split()[:70])
        except Exception as exc:
            LOGGER.warning("LLM generation failed; fallback used: %s", exc)
            return self._fallback(signals)

    def weekly_summary(self, structured_week: dict[str, Any]) -> str:
        """Generate weekly recap narrative.

        Args:
            structured_week: Weekly aggregates.

        Returns:
            str: Summary text.
        """

        if self._generator is None:
            return (
                f"This week included {structured_week.get('sessions', 0)} sessions with ACWR "
                f"at {structured_week.get('acwr', 1.0):.2f}. "
                f"Keep easy days easy and prioritize recovery before the next hard session."
            )
        try:
            prompt = (
                "Write a 3 sentence weekly running summary from this dictionary without adding facts: "
                f"{structured_week}"
            )
            out = self._generator(prompt, max_new_tokens=100, do_sample=False)
            text = out[0]["generated_text"].replace(prompt, "").strip()
            return text if text else self._fallback(
                {
                    "recommended_action": "easy",
                    "form": 0.0,
                    "acwr": structured_week.get("acwr", 1.0),
                }
            )
        except Exception as exc:
            LOGGER.warning("Weekly summary generation failed; fallback used: %s", exc)
            return (
                f"This week included {structured_week.get('sessions', 0)} sessions with ACWR "
                f"at {structured_week.get('acwr', 1.0):.2f}. "
                "Keep easy days easy and prioritize recovery before the next hard session."
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
