"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)


def _default_database_url() -> str:
    db_path = PROJECT_ROOT / "running_coach.db"
    return f"sqlite:///{db_path.as_posix()}"


@dataclass(frozen=True)
class Settings:
    """Typed runtime settings.

    Args:
        strava_client_id: Strava OAuth client ID.
        strava_client_secret: Strava OAuth client secret.
        strava_redirect_uri: OAuth callback URI.
        database_url: SQLAlchemy database URL.
        llm_model: HuggingFace model id.
        fitness_decay_days: Fitness time constant.
        fatigue_decay_days: Fatigue time constant.
        acwr_acute_days: ACWR acute window in days.
        acwr_chronic_days: ACWR chronic window in days.
        max_hard_runs_per_week: Weekly hard run cap.
        injury_risk_threshold: ACWR threshold for injury risk.
    """

    strava_client_id: str = os.getenv("STRAVA_CLIENT_ID", "")
    strava_client_secret: str = os.getenv("STRAVA_CLIENT_SECRET", "")
    strava_redirect_uri: str = os.getenv("STRAVA_REDIRECT_URI", "http://localhost:8000/auth/callback")
    database_url: str = os.getenv("DATABASE_URL", _default_database_url())
    llm_model: str = os.getenv("LLM_MODEL", "microsoft/phi-3-mini-4k-instruct")
    narrative_mode: str = os.getenv("NARRATIVE_MODE", "fast")
    fitness_decay_days: int = int(os.getenv("FITNESS_DECAY_DAYS", "42"))
    fatigue_decay_days: int = int(os.getenv("FATIGUE_DECAY_DAYS", "7"))
    acwr_acute_days: int = int(os.getenv("ACWR_ACUTE_DAYS", "7"))
    acwr_chronic_days: int = int(os.getenv("ACWR_CHRONIC_DAYS", "28"))
    max_hard_runs_per_week: int = int(os.getenv("MAX_HARD_RUNS_PER_WEEK", "2"))
    injury_risk_threshold: float = float(os.getenv("INJURY_RISK_THRESHOLD", "1.5"))

    def __post_init__(self) -> None:
        """Normalize relative SQLite paths so DB is stable across launch directories."""

        value = self.database_url
        if not value.startswith("sqlite:///"):
            return
        path_part = value.replace("sqlite:///", "", 1)
        # Keep absolute sqlite paths as-is (Unix `/...` or Windows `C:/...`).
        is_windows_abs = len(path_part) >= 3 and path_part[1] == ":" and path_part[2] in ("/", "\\")
        if path_part.startswith("/") or is_windows_abs:
            return
        absolute = (PROJECT_ROOT / path_part).resolve().as_posix()
        object.__setattr__(self, "database_url", f"sqlite:///{absolute}")


settings = Settings()
