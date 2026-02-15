"""Sports science metrics and data quality checks (NumPy vectorized)."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Iterable

import numpy as np


def trimp_edwards(hr_stream: list[float], max_hr: int = 190) -> float:
    """Compute Edwards TRIMP using heart-rate zone minutes.

    Formula:
        TRIMP = sum(zone_weight_i * minutes_in_zone_i)

    Zones are based on percentage of max HR and weighted [1..5].
    """

    if not hr_stream or max_hr <= 0:
        return 0.0
    hr = np.asarray(hr_stream, dtype=np.float64)
    pct = hr / float(max_hr)

    bins = np.array([0.6, 0.7, 0.8, 0.9, 1.01], dtype=np.float64)
    weights = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    idx = np.digitize(pct, bins=bins, right=True)
    idx = np.clip(idx, 0, len(weights) - 1)

    minutes_per_sample = 1.0 / 60.0
    return float(np.sum(weights[idx]) * minutes_per_sample)


def acwr(series: list[tuple[date, float]], acute_days: int = 7, chronic_days: int = 28) -> dict[date, float]:
    """Compute ACWR with vectorized rolling means.

    Formula:
        ACWR_t = mean(load_{t-acute+1..t}) / mean(load_{t-chronic+1..t})

    Acute defaults to 7 days and chronic to 28 days.
    """

    if not series:
        return {}

    loads_by_date: dict[date, float] = defaultdict(float)
    for d, value in series:
        loads_by_date[d] += float(value)

    min_day = min(loads_by_date)
    max_day = max(loads_by_date)
    n_days = (max_day - min_day).days + 1

    values = np.zeros(n_days, dtype=np.float64)
    for d, value in loads_by_date.items():
        values[(d - min_day).days] = value

    cumsum = np.concatenate([[0.0], np.cumsum(values)])

    def rolling_mean(window: int) -> np.ndarray:
        out = np.zeros_like(values)
        for i in range(n_days):
            left = max(0, i - window + 1)
            total = cumsum[i + 1] - cumsum[left]
            out[i] = total / float(window)
        return out

    acute = rolling_mean(acute_days)
    chronic = rolling_mean(chronic_days)
    ratio = acute / np.maximum(chronic, 1e-6)

    return {min_day + timedelta(days=i): float(ratio[i]) for i in range(n_days)}


def grade_adjusted_pace(distance_m: float, moving_time_s: int, elevation_gain_m: float) -> float:
    """Estimate Minetti-inspired grade-adjusted pace in sec/km."""

    if distance_m <= 0 or moving_time_s <= 0:
        return 0.0
    pace_sec_per_km = moving_time_s / (distance_m / 1000.0)
    grade = elevation_gain_m / max(distance_m, 1.0)
    cost_ratio = 1.0 + 19.5 * grade * grade + 3.6 * grade
    return float(pace_sec_per_km / max(cost_ratio, 0.6))


def efficiency_index(
    distance_m: float,
    moving_time_s: int,
    avg_hr: float | None,
    hr_stream: list[float] | None = None,
) -> tuple[float, float]:
    """Compute pace/HR efficiency and decoupling drift percentage."""

    if distance_m <= 0 or moving_time_s <= 0 or not avg_hr or avg_hr <= 0:
        return 0.0, 0.0

    speed_mps = distance_m / float(moving_time_s)
    eff = speed_mps / float(avg_hr)

    drift = 0.0
    if hr_stream and len(hr_stream) >= 20:
        arr = np.asarray(hr_stream, dtype=np.float64)
        mid = arr.shape[0] // 2
        first = np.mean(arr[:mid])
        second = np.mean(arr[mid:])
        if first > 0:
            drift = float(((second - first) / first) * 100.0)

    return float(eff), drift


def vo2max_daniels(distance_m: float, moving_time_s: int) -> float:
    """Estimate VO2max via Daniels/Gilbert equations."""

    if distance_m <= 0 or moving_time_s <= 0:
        return 0.0

    velocity_m_per_min = distance_m / (moving_time_s / 60.0)
    vo2 = -4.60 + 0.182258 * velocity_m_per_min + 0.000104 * velocity_m_per_min * velocity_m_per_min
    t_min = moving_time_s / 60.0
    pct_vo2max = 0.8 + 0.1894393 * np.exp(-0.012778 * t_min) + 0.2989558 * np.exp(-0.1932605 * t_min)
    return float(vo2 / max(float(pct_vo2max), 1e-6))


def training_monotony_and_strain(daily_loads: Iterable[float]) -> tuple[float, float]:
    """Compute Foster monotony and strain.

    Formula:
        monotony = mean(daily_load) / std(daily_load)
        strain = monotony * sum(daily_load)
    """

    arr = np.asarray(list(daily_loads), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    avg = np.mean(arr)
    stdev = max(float(np.std(arr)), 1e-9)
    monotony = float(avg / stdev)
    strain = float(monotony * np.sum(arr))
    return monotony, strain


def detect_gps_drift(velocity_mps: list[float], latlng: list[list[float]] | list[tuple[float, float]]) -> bool:
    """Detect GPS drift from speed spikes or repeated backtracking points."""

    if velocity_mps:
        v = np.asarray(velocity_mps, dtype=np.float64)
        if np.any(v > (25.0 / 3.6)):
            return True

    if len(latlng) < 3:
        return False

    points = np.asarray(latlng, dtype=np.float64)
    repeated = np.all(points[2:] == points[:-2], axis=1)
    return bool(np.sum(repeated) > max(2, len(latlng) // 20))


def detect_hr_anomalies(hr_stream: list[float]) -> dict[str, bool]:
    """Detect out-of-range and flatline heart-rate anomalies."""

    if not hr_stream:
        return {"out_of_range": True, "flatline": True}
    arr = np.asarray(hr_stream, dtype=np.float64)
    out_of_range = bool(np.any((arr < 40) | (arr > 220)))
    flatline = bool((np.max(arr) - np.min(arr)) < 2.0)
    return {"out_of_range": out_of_range, "flatline": flatline}


def completeness_score(streams: dict[str, dict], expected: list[str]) -> float:
    """Compute expected stream completeness ratio."""

    if not expected:
        return 1.0
    present = sum(1 for key in expected if streams.get(key, {}).get("data"))
    return present / float(len(expected))


def detect_periodization_phase(days_to_race: int | None, acwr_value: float) -> str:
    """Infer periodization phase: base/build/peak/taper."""

    if days_to_race is not None and days_to_race <= 14:
        return "taper"
    if acwr_value < 0.8:
        return "base"
    if acwr_value < 1.2:
        return "build"
    return "peak"


def polarized_distribution(actions: list[str]) -> tuple[float, float]:
    """Calculate easy vs hard action split."""

    if not actions:
        return 0.0, 0.0
    arr = np.asarray(actions, dtype=object)
    easy = np.isin(arr, ["rest", "easy"]).sum()
    hard = np.isin(arr, ["moderate", "hard"]).sum()
    total = max(arr.size, 1)
    return float(easy / total), float(hard / total)
