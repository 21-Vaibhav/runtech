"""Vercel entrypoint for FastAPI."""

from fastapi import FastAPI

from app.api.server import app as running_coach_app

app: FastAPI = running_coach_app

