# Running Coach MVP

Production-oriented running performance analysis MVP with Strava sync, state estimation, constrained recommendations, feedback calibration, local-only LLM narration, FastAPI, and CLI workflows.

## Structure

```text
running-coach-mvp/
+-- app/
¦   +-- data/
¦   ¦   +-- strava_client.py
¦   ¦   +-- database.py
¦   ¦   +-- pipeline.py
¦   +-- models/
¦   ¦   +-- state_estimator.py
¦   ¦   +-- metrics.py
¦   ¦   +-- confidence.py
¦   +-- decision/
¦   ¦   +-- optimizer.py
¦   +-- feedback/
¦   ¦   +-- tracker.py
¦   +-- llm/
¦   ¦   +-- narrator.py
¦   +-- api/
¦   ¦   +-- server.py
¦   +-- cli.py
+-- tests/
¦   +-- test_state_estimator.py
¦   +-- test_decision_engine.py
¦   +-- test_metrics.py
+-- frontend/
¦   +-- index.html
+-- requirements.txt
+-- setup.sh
+-- .env.example
+-- README.md
```

## Features

- Strava OAuth 2.0 flow, token refresh, and rate-limit backoff.
- SQLite via SQLAlchemy with required tables.
- Metrics: TRIMP, ACWR, grade-adjusted pace, efficiency/drift, VO2max, monotony/strain.
- Dual state models: 2D Kalman and Banister impulse-response with 95% CI.
- Constraint decision engine with taper, ACWR safeguards, and polarized distribution checks.
- Feedback calibration with agreement/adherence/RMSE/overtraining monitoring.
- Local-only LLM narration using `microsoft/phi-3-mini-4k-instruct` with deterministic fallback.
- FastAPI endpoints and Click CLI commands.

## API

- `GET /auth/url`
- `POST /auth/callback`
- `POST /sync`
- `GET /state/{user_id}`
- `POST /recommend/{user_id}`
- `POST /feedback`
- `GET /stats/{user_id}`
- `GET /history/{user_id}`

Swagger: `http://localhost:8000/docs`

## CLI

- `python -m app.cli init`
- `python -m app.cli auth`
- `python -m app.cli sync --user-id 1 --days 90`
- `python -m app.cli update-state --user-id 1`
- `python -m app.cli recommend --user-id 1`
- `python -m app.cli stats --user-id 1`

## Run

```bash
pip install -r requirements.txt
uvicorn app.api.server:app --reload
```

## Test

```bash
pytest -q
```
