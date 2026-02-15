"""Data pipeline for syncing Strava data and computing initial metrics."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.data.database import Activity, ActivityStream, ComputedMetric, User
from app.data.strava_client import StravaClient
from app.models.metrics import (
    acwr,
    completeness_score,
    detect_gps_drift,
    detect_hr_anomalies,
    efficiency_index,
    grade_adjusted_pace,
    trimp_edwards,
    vo2max_daniels,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Sync summary payload."""

    synced_activities: int
    computed_metrics: int


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _stream_data(streams: dict[str, Any], key: str) -> list[float]:
    raw = streams.get(key, {}).get("data", [])
    return [float(v) for v in raw]


def sync_activities(db: Session, user_id: int, days_back: int = 90) -> SyncResult:
    """Sync activities and streams for one user.

    Args:
        db: Database session.
        user_id: Internal user id.
        days_back: History window in days.

    Returns:
        SyncResult: Sync count summary.
    """

    t0 = time.perf_counter()
    user = db.get(User, user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")

    client = StravaClient(user=user)
    after_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
    after_epoch = int(after_dt.timestamp())

    synced = 0
    computed = 0
    page = 1

    stream_rows: list[ActivityStream] = []
    metric_rows: list[ComputedMetric] = []

    while True:
        activities = client.list_activities(after_epoch=after_epoch, page=page, per_page=100)
        if not activities:
            break

        for raw_activity in activities:

            strava_id = int(raw_activity["id"])
            existing = db.scalar(select(Activity).where(Activity.strava_activity_id == strava_id))
            if existing is not None:
                continue

            try:
                streams = client.get_activity_streams(strava_id)
            except Exception:
                # Some activity types may not expose detailed streams.
                streams = {}
            velocity = _stream_data(streams, "velocity_smooth")
            hr_stream = _stream_data(streams, "heartrate")
            alt_stream = _stream_data(streams, "altitude")
            cad_stream = _stream_data(streams, "cadence")
            latlng_stream = streams.get("latlng", {}).get("data", [])

            expected_streams = ["time", "heartrate"]
            if raw_activity.get("type") == "Run":
                expected_streams.extend(["velocity_smooth", "altitude", "cadence"])

            gps_drift = detect_gps_drift(velocity, latlng_stream) if velocity else False
            hr_flags = detect_hr_anomalies(hr_stream) if hr_stream else {"out_of_range": False, "flatline": False}
            completeness = completeness_score(streams, expected=expected_streams)

            quality_flags = {
                "gps_drift": gps_drift,
                "hr_out_of_range": hr_flags["out_of_range"],
                "hr_flatline": hr_flags["flatline"],
            }
            quality_penalty = 0.25 * int(gps_drift) + 0.2 * int(hr_flags["out_of_range"]) + 0.2 * int(hr_flags["flatline"])
            quality = max(0.0, min(1.0, completeness - quality_penalty))

            activity = Activity(
                user_id=user_id,
                strava_activity_id=strava_id,
                name=raw_activity.get("name", "Run"),
                type=raw_activity.get("type", "Run"),
                start_date=_parse_date(raw_activity["start_date"]),
                distance_m=float(raw_activity.get("distance", 0.0)),
                moving_time_s=int(raw_activity.get("moving_time", 0)),
                elapsed_time_s=int(raw_activity.get("elapsed_time", 0)),
                total_elevation_gain_m=float(raw_activity.get("total_elevation_gain", 0.0)),
                average_heartrate=(float(raw_activity["average_heartrate"]) if raw_activity.get("average_heartrate") else None),
                average_cadence=(float(raw_activity["average_cadence"]) if raw_activity.get("average_cadence") else None),
                data_quality_score=quality,
                data_quality_flags=quality_flags,
            )
            db.add(activity)
            db.flush()

            stream_payloads = {
                "heartrate": hr_stream,
                "velocity_smooth": velocity,
                "altitude": alt_stream,
                "cadence": cad_stream,
            }
            for stream_type, values in stream_payloads.items():
                if values:
                    stream_rows.append(ActivityStream(activity_id=activity.id, stream_type=stream_type, values=values))

            duration_min = max(activity.moving_time_s / 60.0, 1.0)
            activity_type = activity.type.lower()
            trimp = trimp_edwards(hr_stream if hr_stream else ([activity.average_heartrate] if activity.average_heartrate else []), max_hr=190)
            gap = grade_adjusted_pace(activity.distance_m, activity.moving_time_s, activity.total_elevation_gain_m) if activity.type == "Run" else 0.0
            eff, drift = (
                efficiency_index(activity.distance_m, activity.moving_time_s, activity.average_heartrate, hr_stream)
                if activity.type == "Run"
                else (0.0, 0.0)
            )
            vo2 = vo2max_daniels(activity.distance_m, activity.moving_time_s) if activity.type == "Run" else 0.0

            # Unified session load so non-running sessions (e.g. weights) still contribute to fatigue.
            if trimp > 0:
                session_load = trimp
            else:
                type_factor = 1.0
                if "weight" in activity_type or "workout" in activity_type:
                    type_factor = 1.35
                elif "ride" in activity_type or "cycle" in activity_type:
                    type_factor = 1.1
                elif "walk" in activity_type or "hike" in activity_type:
                    type_factor = 0.75
                elif "yoga" in activity_type or "pilates" in activity_type:
                    type_factor = 0.55
                session_load = duration_min * type_factor

            metric_pairs = {
                "trimp": trimp,
                "session_load": session_load,
                "grade_adjusted_pace": gap,
                "efficiency_index": eff,
                "efficiency_drift": drift,
                "vo2max": vo2,
            }
            for name, value in metric_pairs.items():
                metric_rows.append(
                    ComputedMetric(
                        user_id=user_id,
                        activity_id=activity.id,
                        metric_date=activity.start_date.date(),
                        metric_name=name,
                        metric_value=float(value),
                        metadata_json={"duration_min": duration_min, "activity_type": activity.type},
                    )
                )
                computed += 1

            synced += 1

            if len(stream_rows) >= 500:
                db.add_all(stream_rows)
                stream_rows.clear()
            if len(metric_rows) >= 1000:
                db.add_all(metric_rows)
                metric_rows.clear()

        page += 1

    if stream_rows:
        db.add_all(stream_rows)
    if metric_rows:
        db.add_all(metric_rows)

    _recompute_daily_acwr(db=db, user_id=user_id)
    LOGGER.info("sync_activities completed user=%s synced=%s metrics=%s elapsed=%.2fs", user_id, synced, computed, time.perf_counter() - t0)
    return SyncResult(synced_activities=synced, computed_metrics=computed)


def _recompute_daily_acwr(db: Session, user_id: int) -> None:
    """Recompute ACWR snapshots from TRIMP metrics.

    Args:
        db: Database session.
        user_id: Internal user id.

    Returns:
        None
    """

    rows = db.scalars(
        select(ComputedMetric)
        .where(ComputedMetric.user_id == user_id, ComputedMetric.metric_name == "session_load")
        .order_by(ComputedMetric.metric_date.asc())
    ).all()
    series = [(row.metric_date, row.metric_value) for row in rows]
    daily_acwr = acwr(series, acute_days=settings.acwr_acute_days, chronic_days=settings.acwr_chronic_days)
    db.execute(delete(ComputedMetric).where(ComputedMetric.user_id == user_id, ComputedMetric.metric_name == "acwr"))
    for dt, value in daily_acwr.items():
        db.add(
            ComputedMetric(
                user_id=user_id,
                activity_id=None,
                metric_date=dt,
                metric_name="acwr",
                metric_value=float(value),
                metadata_json={"acute_days": settings.acwr_acute_days, "chronic_days": settings.acwr_chronic_days},
            )
        )
