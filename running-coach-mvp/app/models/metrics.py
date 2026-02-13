"""Sports science metrics and data quality checks."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta
from statistics import mean
from typing import Iterable


def trimp_edwards(hr_stream: list[float], max_hr: int = 190) -> float:
    """Compute Edwards TRIMP from heart-rate zones.

    Args:
        hr_stream: Heart-rate values sampled at 1Hz or near 1Hz.
        max_hr: Athlete max heart rate.

    Returns:
        float: Training impulse.
    """

    if not hr_stream:
        return 0.0

    zone_weights = [1, 2, 3, 4, 5]
    zone_bounds = [0.6, 0.7, 0.8, 0.9, 1.01]
    minutes_per_sample = 1.0 / 60.0
    trimp = 0.0
    for hr in hr_stream:
        pct = hr / max_hr if max_hr > 0 else 0.0
        for idx, bound in enumerate(zone_bounds):
            if pct <= bound:
                trimp += zone_weights[idx] * minutes_per_sample
                break
    return trimp


def acwr(series: list[tuple[date, float]], acute_days: int = 7, chronic_days: int = 28) -> dict[date, float]:
    """Compute ACWR from daily load values.

    Args:
        series: List of `(date, load)` tuples.
        acute_days: Acute rolling window.
        chronic_days: Chronic rolling window.

    Returns:
        dict[date, float]: ACWR by date.
    """

    if not series:
        return {}

    loads_by_date: dict[date, float] = defaultdict(float)
    min_day = min(d for d, _ in series)
    max_day = max(d for d, _ in series)
    for d, value in series:
        loads_by_date[d] += value

    days: list[date] = []
    cursor = min_day
    while cursor <= max_day:
        days.append(cursor)
        cursor += timedelta(days=1)

    output: dict[date, float] = {}
    for day in days:
        acute_values = [loads_by_date.get(day - timedelta(days=i), 0.0) for i in range(acute_days)]
        chronic_values = [loads_by_date.get(day - timedelta(days=i), 0.0) for i in range(chronic_days)]
        acute_avg = sum(acute_values) / float(acute_days)
        chronic_avg = sum(chronic_values) / float(chronic_days)
        output[day] = acute_avg / max(chronic_avg, 1e-6)
    return output


def grade_adjusted_pace(distance_m: float, moving_time_s: int, elevation_gain_m: float) -> float:
    """Estimate Minetti-inspired grade-adjusted pace.

    Args:
        distance_m: Distance in meters.
        moving_time_s: Moving time in seconds.
        elevation_gain_m: Elevation gain in meters.

    Returns:
        float: Grade-adjusted pace seconds per km.
    """

    if distance_m <= 0 or moving_time_s <= 0:
        return 0.0
    pace_sec_per_km = moving_time_s / (distance_m / 1000.0)
    grade = elevation_gain_m / max(distance_m, 1.0)
    cost_ratio = 1.0 + 19.5 * grade**2 + 3.6 * grade
    return pace_sec_per_km / max(cost_ratio, 0.6)


def efficiency_index(
    distance_m: float,
    moving_time_s: int,
    avg_hr: float | None,
    hr_stream: list[float] | None = None,
) -> tuple[float, float]:
    """Compute pace/HR efficiency and simple decoupling drift.

    Args:
        distance_m: Distance in meters.
        moving_time_s: Moving time in seconds.
        avg_hr: Average heart rate.
        hr_stream: Optional full heart-rate stream.

    Returns:
        tuple[float, float]: `(efficiency_index, drift_pct)`.
    """

    if distance_m <= 0 or moving_time_s <= 0 or not avg_hr or avg_hr <= 0:
        return 0.0, 0.0

    speed_mps = distance_m / moving_time_s
    eff = speed_mps / avg_hr

    drift = 0.0
    if hr_stream and len(hr_stream) >= 20:
        midpoint = len(hr_stream) // 2
        first = hr_stream[:midpoint]
        second = hr_stream[midpoint:]
        first_mean = mean(first)
        second_mean = mean(second)
        if first_mean > 0:
            drift = ((second_mean - first_mean) / first_mean) * 100.0
    return eff, drift


def vo2max_daniels(distance_m: float, moving_time_s: int) -> float:
    """Estimate VO2max via Daniels and Gilbert race formula.

    Args:
        distance_m: Distance in meters.
        moving_time_s: Time in seconds.

    Returns:
        float: Estimated VO2max.
    """

    if distance_m <= 0 or moving_time_s <= 0:
        return 0.0

    velocity_m_per_min = distance_m / (moving_time_s / 60.0)
    vo2 = -4.60 + 0.182258 * velocity_m_per_min + 0.000104 * velocity_m_per_min * velocity_m_per_min
    t_min = moving_time_s / 60.0
    pct_vo2max = 0.8 + 0.1894393 * math.exp(-0.012778 * t_min) + 0.2989558 * math.exp(-0.1932605 * t_min)
    return vo2 / max(pct_vo2max, 1e-6)


def training_monotony_and_strain(daily_loads: Iterable[float]) -> tuple[float, float]:
    """Compute Foster monotony and strain.

    Args:
        daily_loads: Weekly daily load values.

    Returns:
        tuple[float, float]: `(monotony, strain)`.
    """

    loads = list(daily_loads)
    if not loads:
        return 0.0, 0.0
    avg = sum(loads) / len(loads)
    variance = sum((x - avg) ** 2 for x in loads) / len(loads)
    stdev = math.sqrt(max(variance, 1e-9))
    monotony = avg / stdev
    strain = monotony * sum(loads)
    return monotony, strain


def detect_gps_drift(velocity_mps: list[float], latlng: list[list[float]] | list[tuple[float, float]]) -> bool:
    """Detect GPS drift from speed spikes or backtracking.

    Args:
        velocity_mps: Velocity stream.
        latlng: GPS coordinate list.

    Returns:
        bool: True if drift-like artifacts exist.
    """

    if any(v > (25.0 / 3.6) for v in velocity_mps):
        return True
    if len(latlng) < 3:
        return False
    backward_steps = 0
    for idx in range(2, len(latlng)):
        if latlng[idx][0] == latlng[idx - 2][0] and latlng[idx][1] == latlng[idx - 2][1]:
            backward_steps += 1
    return backward_steps > max(2, len(latlng) // 20)


def detect_hr_anomalies(hr_stream: list[float]) -> dict[str, bool]:
    """Detect out-of-range and flatline heart-rate anomalies.

    Args:
        hr_stream: Heart-rate series.

    Returns:
        dict[str, bool]: HR anomaly flags.
    """

    if not hr_stream:
        return {"out_of_range": True, "flatline": True}
    out_of_range = any(hr < 40 or hr > 220 for hr in hr_stream)
    flatline = max(hr_stream) - min(hr_stream) < 2.0
    return {"out_of_range": out_of_range, "flatline": flatline}


def completeness_score(streams: dict[str, dict], expected: list[str]) -> float:
    """Compute expected stream completeness ratio.

    Args:
        streams: Stream payload dict.
        expected: Required stream names.

    Returns:
        float: Completeness in `[0, 1]`.
    """

    if not expected:
        return 1.0
    present = sum(1 for key in expected if streams.get(key, {}).get("data"))
    return present / float(len(expected))


def detect_periodization_phase(days_to_race: int | None, acwr_value: float) -> str:
    """Infer periodization phase.

    Args:
        days_to_race: Days until next race.
        acwr_value: Current ACWR.

    Returns:
        str: One of `base`, `build`, `peak`, `taper`.
    """

    if days_to_race is not None and days_to_race <= 14:
        return "taper"
    if acwr_value < 0.8:
        return "base"
    if acwr_value < 1.2:
        return "build"
    return "peak"


def polarized_distribution(actions: list[str]) -> tuple[float, float]:
    """Calculate easy vs hard action split.

    Args:
        actions: Historical actions.

    Returns:
        tuple[float, float]: `(easy_pct, hard_pct)`.
    """

    if not actions:
        return 0.0, 0.0
    easy = sum(1 for a in actions if a in {"rest", "easy"})
    hard = sum(1 for a in actions if a in {"moderate", "hard"})
    total = len(actions)
    return easy / total, hard / total
