"""Configuration & secret loading.

The OpenAI API key lives ONLY in the (git-ignored) .env file and is read into the
process here. It is never written to logs, responses, or the frontend.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level above this file's package).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    project_root: Path
    frontend_dir: Path

    @property
    def has_key(self) -> bool:
        return bool(self.openai_api_key) and self.openai_api_key.startswith("sk-")


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o",
        project_root=_PROJECT_ROOT,
        frontend_dir=_PROJECT_ROOT / "frontend",
    )


settings = load_settings()
