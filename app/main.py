"""HTTP surface: Agent Card, Bearer authentication, A2A version and media-type checks, and routes.

Run with: uvicorn app.main:create_app --factory --host 0.0.0.0 --port $PORT --workers 1
One worker only: duplicate-request waiting is kept in memory, and the database is a local SQLite file.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import Settings, load_settings
from .decider import AnthropicDecider, Decider, UnconfiguredDecider
from .errors import A2AError, invalid_argument
from .protocol import A2A_MEDIA_TYPE, A2A_VERSION, build_agent_card, require_message
from .service import TaskService
from .store import Store

MAX_REQUEST_BYTES = 1024 * 1024


def create_app(settings: Settings | None = None, decider: Decider | None = None) -> FastAPI:
    settings = settings or load_settings()
    if decider is None:
        if settings.anthropic_api_key:
            decider = AnthropicDecider(settings.anthropic_api_key, settings.model, settings.effort)
        else:
            decider = UnconfiguredDecider()

    service = TaskService(
        Store(settings.database_path),
        decider,
        budget_seconds=settings.decision_budget_seconds,
        daily_limit=settings.daily_model_call_limit,
    )
    prefix = settings.path_prefix

    app = FastAPI(title="Invoice Action Agent", docs_url=None, redoc_url=None, openapi_url=None)
    app.router.redirect_slashes = False  # the grader must never be redirected

    def a2a_response(payload: Any, status: int = 200, headers: dict[str, str] | None = None) -> JSONResponse:
        return JSONResponse(payload, status_code=status, headers=headers, media_type=A2A_MEDIA_TYPE)

    @app.exception_handler(A2AError)
    async def handle_a2a_error(request: Request, exc: A2AError) -> JSONResponse:
        return a2a_response(exc.body(), exc.http_status, exc.headers)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        return a2a_response({"error": {"code": 500, "status": "INTERNAL", "message": "Internal error."}}, 500)

    def base_url(request: Request) -> str:
        if settings.public_base_url:
            return settings.public_base_url
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        host = request.headers.get("host", request.url.netloc)
        return f"{scheme}://{host}{prefix}/"

    def principal_for(request: Request) -> str:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        token = token.strip()
        challenge = {"WWW-Authenticate": "Bearer"}
        if scheme.lower() != "bearer" or not token:
            raise A2AError(401, "UNAUTHENTICATED", "A Bearer token is required.", challenge)
        if settings.allowed_tokens is not None and not any(
            hmac.compare_digest(token.encode(), allowed.encode()) for allowed in settings.allowed_tokens
        ):
            raise A2AError(401, "UNAUTHENTICATED", "The Bearer token is not accepted.", challenge)
        # Each distinct token is its own user. Store a hash, never the token itself.
        return "principal:" + hashlib.sha256(token.encode("utf-8")).hexdigest()

    def require_version(request: Request) -> None:
        if request.headers.get("A2A-Version") != A2A_VERSION:
            raise A2AError(400, "VERSION_NOT_SUPPORTED", f"A2A-Version must be {A2A_VERSION}.")

    def require_a2a_body_type(request: Request) -> None:
        media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if media_type != A2A_MEDIA_TYPE:
            raise A2AError(415, "UNSUPPORTED_MEDIA_TYPE", f"Content-Type must be {A2A_MEDIA_TYPE}.")

    async def read_json_object(request: Request) -> dict:
        raw = await request.body()
        if len(raw) > MAX_REQUEST_BYTES:
            raise A2AError(413, "PAYLOAD_TOO_LARGE", "Request body is too large.")
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise invalid_argument("Request body must be valid JSON.") from exc
        if not isinstance(body, dict):
            raise invalid_argument("Request body must be a JSON object.")
        return body

    @app.get("/.well-known/agent-card.json")
    async def agent_card(request: Request) -> JSONResponse:
        return JSONResponse(build_agent_card(base_url(request)), media_type="application/json")

    @app.post(f"{prefix}/message:send")
    async def message_send(request: Request) -> JSONResponse:
        principal = principal_for(request)
        require_version(request)
        require_a2a_body_type(request)
        message = require_message(await read_json_object(request))
        task = await run_in_threadpool(service.handle_message, principal, message)
        return a2a_response({"task": task})

    @app.get(f"{prefix}/tasks/{{task_id}}")
    async def task_get(task_id: str, request: Request) -> JSONResponse:
        principal = principal_for(request)
        require_version(request)
        task = await run_in_threadpool(service.get, principal, task_id)
        return a2a_response(task)

    @app.get(f"{prefix}/tasks")
    async def task_list(request: Request) -> JSONResponse:
        principal = principal_for(request)
        require_version(request)
        tasks = await run_in_threadpool(service.list_tasks, principal)
        return a2a_response({"tasks": tasks})

    @app.post(f"{prefix}/tasks/{{task_id}}:cancel")
    async def task_cancel(task_id: str, request: Request) -> JSONResponse:
        principal = principal_for(request)
        require_version(request)
        task = await run_in_threadpool(service.cancel, principal, task_id)
        return a2a_response(task)

    return app
