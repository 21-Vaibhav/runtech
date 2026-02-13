"""Confidence and uncertainty helpers."""

from __future__ import annotations

import math


def confidence_interval(value: float, variance: float, z: float = 1.96) -> tuple[float, float]:
    """Build normal-approximation confidence interval.

    Args:
        value: Mean estimate.
        variance: Variance estimate.
        z: Z-score multiplier.

    Returns:
        tuple[float, float]: `(lower, upper)` bounds.
    """

    std = math.sqrt(max(variance, 1e-9))
    margin = z * std
    return value - margin, value + margin


def bounded_score(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp score to closed interval.

    Args:
        value: Raw score.
        low: Lower bound.
        high: Upper bound.

    Returns:
        float: Clamped score.
    """

    return max(low, min(high, value))
