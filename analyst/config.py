"""Central configuration. All secrets/limits come from environment variables (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    model: str = "claude-sonnet-5-5"
    llm_enabled: bool = True
    max_upload_mb: int = 200
    max_rows: int = 2_000_000
    max_tool_calls: int = 8
    tool_timeout_s: int = 90
    max_tokens: int = 2048
    send_sample_rows: bool = False  # if True, 5 sample rows are sent to the AI provider
    max_history_states: int = 15  # undo depth (each state is a DataFrame copy)

    @property
    def ai_available(self) -> bool:
        return bool(self.api_key) and self.llm_enabled


def load_settings() -> Settings:
    return Settings(
        api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        model=os.getenv("AI_MODEL", "claude-sonnet-5-5").strip() or "claude-sonnet-5-5",
        llm_enabled=_bool("AI_ENABLED", True),
        max_upload_mb=_int("MAX_UPLOAD_MB", 200),
        max_rows=_int("MAX_ROWS", 2_000_000),
        max_tool_calls=_int("MAX_TOOL_CALLS", 8),
        tool_timeout_s=_int("TOOL_TIMEOUT_S", 90),
        max_tokens=_int("AI_MAX_TOKENS", 2048),
        send_sample_rows=_bool("SEND_SAMPLE_ROWS", False),
        max_history_states=_int("MAX_UNDO_STATES", 15),
    )
