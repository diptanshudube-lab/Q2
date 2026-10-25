"""Errors that become A2A responses. Messages are generic so they never reveal another user's task."""

from __future__ import annotations


class A2AError(Exception):
    def __init__(
        self,
        http_status: int,
        status: str,
        message: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.status = status
        self.message = message
        self.headers = headers

    def body(self) -> dict:
        return {"error": {"code": self.http_status, "status": self.status, "message": self.message}}


def invalid_argument(message: str) -> A2AError:
    return A2AError(400, "INVALID_ARGUMENT", message)


def not_found() -> A2AError:
    # Same answer for "missing" and "belongs to someone else", so existence never leaks.
    return A2AError(404, "NOT_FOUND", "Task not found.")
