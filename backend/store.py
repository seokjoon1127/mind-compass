"""In-memory session store (MVP — no DB, no persistence across restarts)."""
from __future__ import annotations

from .models import DecisionSession

_sessions: dict[str, DecisionSession] = {}


def save(session: DecisionSession) -> DecisionSession:
    _sessions[session.id] = session
    return session


def get(session_id: str) -> DecisionSession | None:
    return _sessions.get(session_id)


def exists(session_id: str) -> bool:
    return session_id in _sessions
