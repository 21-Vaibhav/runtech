# Running Coach MVP

Production-oriented running performance analysis app with Strava sync, state estimation, safety-constrained recommendations, local narration, FastAPI API, CLI tooling, and a responsive dashboard.

## Architecture

Data flow is intentionally split for performance:

1. `POST /sync`: fetch and store Strava activities/streams, compute raw metrics.
2. `POST /state/update/{user_id}`: compute state models (Kalman + Banister), write `user_state_cache`.
3. `POST /recommend/{user_id}`: read cached state only, apply constraints, return recommendation.

This keeps recommendation latency low and avoids expensive recomputation on each request.

## Project Structure

```text
running-coach-mvp/
├── app/
│   ├── data/
│   │   ├── strava_client.py
│   │   ├── database.py
│   │   └── pipeline.py
│   ├── models/
│   │   ├── metrics.py
│   │   ├── state_estimator.py
│   │   └── confidence.py
│   ├── decision/
│   │   └── optimizer.py
│   ├── feedback/
│   │   └── tracker.py
│   ├── llm/
│   │   └── narrator.py
│   ├── api/
│   │   └── server.py
│   └── cli.py
├── frontend/
│   └── index.html
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Core Features

- Strava OAuth 2.0 exchange + token refresh.
- SQLite + SQLAlchemy with WAL mode and indexes.
- Vectorized metrics (NumPy): TRIMP, ACWR, efficiency/drift, VO2, monotony/strain.
- Dual state models:
  - Kalman filter state update.
  - Banister impulse-response model.
- Constraint decision engine:
  - fatigue CI guardrails
  - ACWR injury threshold blocks
  - no back-to-back hard/moderate
  - max hard sessions/week
  - race taper constraints
  - polarized 80/20 check
- Recommendation API cache (6h TTL).
- Deterministic rule-based narrator (serverless-safe, no model downloads).
- Responsive dashboard with charts (fitness/fatigue, weekly load, ACWR).

## Metrics and Equations

Implemented and documented in code:

- TRIMP (Edwards): weighted time spent in HR zones.
- ACWR: `acute_7d_mean / chronic_28d_mean`.
- Banister:
  - `fitness(t) = Σ load_i * exp(-Δt / tau_fit)`
  - `fatigue(t) = Σ load_i * exp(-Δt / tau_fat)`
  - `form = fitness - fatigue`
- Kalman:
  - predict: `x^- = A x + B u`
  - update: `x = x^- + K(z - x^-)`
- Monotony: `mean(load) / std(load)`
- Strain: `monotony * sum(load)`

## API Endpoints

- `GET /auth/url`
- `POST /auth/callback`
- `POST /sync`
- `GET /state/{user_id}`
- `POST /state/update/{user_id}`
- `POST /recommend/{user_id}`
- `POST /feedback`
- `GET /stats/{user_id}`
- `GET /history/{user_id}`
- `GET /dashboard/{user_id}`

Swagger: `http://localhost:8000/docs`

## CLI

- `python -m app.cli init`
- `python -m app.cli auth`
- `python -m app.cli sync --user-id 1 --days 90`
- `python -m app.cli update-state --user-id 1`
- `python -m app.cli recommend --user-id 1`
- `python -m app.cli stats --user-id 1`

## Setup

```bash
pip install -r requirements.txt
uvicorn app.api.server:app --reload
```

Open UI: `frontend/index.html` in browser.

## Recommended Workflow

1. `GET /auth/url` and complete Strava authorization.
2. `POST /auth/callback` with code.
3. `POST /sync`
4. `POST /state/update/{user_id}`
5. `POST /recommend/{user_id}`
6. Submit feedback with `POST /feedback`.

## Testing

```bash
pytest -q
```

Current local status: all tests pass.

## Security Notes

- Configure `CORS_ORIGINS` to your frontend origins in production.
- OAuth uses signed `state` tokens; always complete auth from `/auth/url`.
- Session tokens are required for user data endpoints; set `SESSION_SECRET` in production.
