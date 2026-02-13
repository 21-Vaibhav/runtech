"""Database models and SQLAlchemy session utilities."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Generator

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Base declarative class for all models."""


engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class User(Base):
    """Athlete record and OAuth token storage."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strava_athlete_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    username: Mapped[str | None] = mapped_column(String(128), nullable=True)
    access_token: Mapped[str] = mapped_column(Text)
    refresh_token: Mapped[str] = mapped_column(Text)
    token_expires_at: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Activity(Base):
    """Normalized activity summary."""

    __tablename__ = "activities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    strava_activity_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    type: Mapped[str] = mapped_column(String(64))
    start_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    distance_m: Mapped[float] = mapped_column(Float)
    moving_time_s: Mapped[int] = mapped_column(Integer)
    elapsed_time_s: Mapped[int] = mapped_column(Integer)
    total_elevation_gain_m: Mapped[float] = mapped_column(Float, default=0.0)
    average_heartrate: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_cadence: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_quality_score: Mapped[float] = mapped_column(Float, default=1.0)
    data_quality_flags: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ActivityStream(Base):
    """Per-activity time-series streams."""

    __tablename__ = "activity_streams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    activity_id: Mapped[int] = mapped_column(ForeignKey("activities.id"), index=True)
    stream_type: Mapped[str] = mapped_column(String(32), index=True)
    values: Mapped[list[float]] = mapped_column(JSON)

    activity: Mapped[Activity] = relationship()


class ComputedMetric(Base):
    """Computed per-day or per-activity metrics."""

    __tablename__ = "computed_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    activity_id: Mapped[int | None] = mapped_column(ForeignKey("activities.id"), nullable=True)
    metric_date: Mapped[date] = mapped_column(Date, index=True)
    metric_name: Mapped[str] = mapped_column(String(64), index=True)
    metric_value: Mapped[float] = mapped_column(Float)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class StateEstimate(Base):
    """Daily model state for both model families."""

    __tablename__ = "state_estimates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    estimate_date: Mapped[date] = mapped_column(Date, index=True)
    model_name: Mapped[str] = mapped_column(String(32), index=True)
    fitness: Mapped[float] = mapped_column(Float)
    fatigue: Mapped[float] = mapped_column(Float)
    form: Mapped[float] = mapped_column(Float)
    fitness_ci_low: Mapped[float] = mapped_column(Float)
    fitness_ci_high: Mapped[float] = mapped_column(Float)
    fatigue_ci_low: Mapped[float] = mapped_column(Float)
    fatigue_ci_high: Mapped[float] = mapped_column(Float)


class Recommendation(Base):
    """Daily recommendation outputs and confidence."""

    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    recommendation_date: Mapped[date] = mapped_column(Date, index=True)
    recommended_action: Mapped[str] = mapped_column(String(16))
    confidence_score: Mapped[float] = mapped_column(Float)
    reasoning_dict: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Race(Base):
    """Race schedule used for taper constraints."""

    __tablename__ = "races"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    race_name: Mapped[str] = mapped_column(String(128))
    race_date: Mapped[date] = mapped_column(Date, index=True)
    goal: Mapped[str | None] = mapped_column(String(255), nullable=True)


class CalibrationLog(Base):
    """Feedback and calibration traces."""

    __tablename__ = "calibration_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    log_date: Mapped[date] = mapped_column(Date, index=True)
    recommended_action: Mapped[str] = mapped_column(String(16))
    actual_action: Mapped[str] = mapped_column(String(16))
    observed_outcome: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


def init_db() -> None:
    """Create all tables.

    Returns:
        None
    """

    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Yield a managed SQLAlchemy session.

    Yields:
        Session: Active database session.
    """

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
