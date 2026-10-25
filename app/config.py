"""Settings come from environment variables. On Render, set them in the service's Environment tab."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

DEFAULT_PREFIX = "/a2a"
DEFAULT_MODEL = "claude-opus-5-5"


@dataclass(frozen=True)
class Settings:
    # Exact base URL you submit to the grader, for example https://host/a2a/ (None: derive from the request).
    public_base_url: str | None
    # Path part of the base URL. Routes are mounted here, so the grader's paths resolve correctly.
    path_prefix: str
    database_path: str
    # None accepts any non-empty Bearer token as its own user. A set restricts access to those tokens.
    allowed_tokens: frozenset[str] | None
    anthropic_api_key: str | None
    model: str
    effort: str
    # Total model time allowed inside one request (the grader allows 45 seconds per request).
    decision_budget_seconds: float
    daily_model_call_limit: int


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env

    base_url = (env.get("PUBLIC_BASE_URL") or "").strip() or None
    prefix = DEFAULT_PREFIX
    if base_url is not None:
        parts = urlsplit(base_url)
        if parts.scheme != "https" or not parts.hostname:
            raise ValueError("PUBLIC_BASE_URL must be an absolute https URL.")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("PUBLIC_BASE_URL must not contain credentials, a query, or a fragment.")
        prefix = parts.path.rstrip("/")

    tokens = (env.get("A2A_ALLOWED_TOKENS") or "").strip()
    allowed = frozenset(t.strip() for t in tokens.split(",") if t.strip()) or None

    return Settings(
        public_base_url=base_url,
        path_prefix=prefix,
        database_path=os.path.join(env.get("DATA_DIR", "data"), "agent.sqlite3"),
        allowed_tokens=allowed,
        anthropic_api_key=(env.get("ANTHROPIC_API_KEY") or "").strip() or None,
        model=(env.get("ANTHROPIC_MODEL") or DEFAULT_MODEL).strip(),
        effort=(env.get("ANTHROPIC_EFFORT") or "medium").strip(),
        decision_budget_seconds=float(env.get("DECISION_BUDGET_SECONDS") or "38"),
        daily_model_call_limit=int(env.get("DAILY_MODEL_CALL_LIMIT") or "400"),
    )
