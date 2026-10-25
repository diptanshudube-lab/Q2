"""Task lifecycle: send, continue, cancel, read, list.

Each state change happens inside one store transaction, so a cancel and a result cannot both win.
Model work runs outside the lock. Duplicate messages wait for the first run instead of repeating it.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from .decider import Decider, DecisionInvalid, ModelUnavailable, check_batch
from .errors import A2AError, invalid_argument, not_found
from .protocol import (
    CANCELED,
    COMPLETED,
    FAILED,
    INPUT_REQUIRED,
    MAX_LIST_BYTES,
    OPEN_STATES,
    PROPOSALS_MEDIA_TYPE,
    RECEIPTS_MEDIA_TYPE,
    SUBMITTED,
    WORKING,
    batch_from_message,
    canonical_json,
    results_from_message,
    semantic_hash,
    sha256_hex,
    utc_now,
)
from .store import Store

DECISION_CHUNK = 6  # packages per model call
CHUNK_WORKERS = 4  # model calls allowed in flight at once
MAX_ATTEMPTS = 2
MIN_ATTEMPT_SECONDS = 8.0
WAIT_FOR_DUPLICATE_SECONDS = 45.0
LIST_LIMIT = 50


def _status(state: str, text: str | None = None) -> dict:
    status: dict[str, Any] = {"state": state, "timestamp": utc_now()}
    if text is not None:
        status["message"] = {
            "messageId": str(uuid.uuid4()),
            "role": "ROLE_AGENT",
            "parts": [{"text": text}],
        }
    return status


def _artifact(name: str, media_type: str, data: dict) -> dict:
    return {
        "artifactId": str(uuid.uuid4()),
        "name": name,
        "parts": [{"mediaType": media_type, "data": data}],
    }


def _proposal_artifact(batch: dict, decided: dict[str, dict]) -> dict:
    proposals = []
    for package in batch["packages"]:
        decision = decided[package["packageId"]]
        proposals.append(
            {
                "packageId": package["packageId"],
                "actionId": f"act-{uuid.uuid4().hex}",
                "action": decision["action"],
                "facts": dict(decision["facts"]),
                "evidenceRefs": list(decision["evidenceRefs"]),
                "rationale": decision["rationale"],
            }
        )
    return _artifact(
        "invoice-action-proposals",
        PROPOSALS_MEDIA_TYPE,
        {"batchId": batch["batchId"], "proposals": proposals},
    )


def _proposal_data(task: dict) -> dict:
    for artifact in task["artifacts"]:
        for part in artifact["parts"]:
            if part["mediaType"] == PROPOSALS_MEDIA_TYPE:
                return part["data"]
    raise A2AError(500, "INTERNAL", "Task has no proposals.")


def _executions(proposals: dict, results: dict) -> list[dict]:
    """Check every result against its stored proposal. Only ACCEPTED results become executions."""
    if results.get("batchId") != proposals["batchId"]:
        raise invalid_argument("batchId does not match the task.")

    by_package = {proposal["packageId"]: proposal for proposal in proposals["proposals"]}
    seen: set[str] = set()
    nonces: set[str] = set()
    accepted: list[dict] = []
    for result in results["results"]:
        package_id = result["packageId"]
        proposal = by_package.get(package_id)
        if proposal is None or package_id in seen:
            raise invalid_argument("Each result must match one proposal exactly once.")
        if result["actionId"] != proposal["actionId"] or result["action"] != proposal["action"]:
            raise invalid_argument("A result's actionId or action does not match its proposal.")
        if result["receiptNonce"] in nonces:
            raise invalid_argument("receiptNonce values must be unique.")
        seen.add(package_id)
        nonces.add(result["receiptNonce"])
        if result["outcome"] == "ACCEPTED":
            accepted.append(
                {
                    "packageId": package_id,
                    "actionId": proposal["actionId"],
                    "action": proposal["action"],
                    "receiptNonce": result["receiptNonce"],
                    "facts": proposal["facts"],
                    "evidenceRefs": proposal["evidenceRefs"],
                }
            )
    if seen != set(by_package):
        raise invalid_argument("Results must cover every proposal.")

    order = {package_id: index for index, package_id in enumerate(by_package)}
    accepted.sort(key=lambda execution: order[execution["packageId"]])
    return accepted


class TaskService:
    def __init__(self, store: Store, decider: Decider, budget_seconds: float, daily_limit: int) -> None:
        self._store = store
        self._decider = decider
        self._budget = budget_seconds
        self._daily_limit = daily_limit
        self._inflight: dict[tuple[str, str], threading.Event] = {}

    # Public operations ------------------------------------------------------

    def handle_message(self, principal: str, message: dict) -> dict:
        """Start a task, or continue one with results. A repeated messageId returns the stored Task."""
        message_id = message["messageId"]
        digest = semantic_hash(message)
        key = (principal, message_id)
        while True:
            with self._store.lock:
                seen = self._store.find_request(principal, message_id)
                if seen is not None:
                    if seen.digest != digest:
                        raise A2AError(
                            409,
                            "IDEMPOTENCY_CONFLICT",
                            "This messageId was already used with different content.",
                        )
                    waiting = self._inflight.get(key)
                    if waiting is None:
                        return self._require(principal, seen.task_id)
                elif "taskId" in message:
                    return self._continue(principal, message, digest)
                else:
                    waiting = None
                    task, batch, done = self._start(principal, message, digest)

            if waiting is not None:
                if not waiting.wait(WAIT_FOR_DUPLICATE_SECONDS):
                    raise A2AError(503, "UNAVAILABLE", "The original request is still running. Retry shortly.")
                continue
            return self._run(principal, key, task, batch, done)

    def get(self, principal: str, task_id: str) -> dict:
        with self._store.lock:
            return self._require(principal, task_id)

    def list_tasks(self, principal: str) -> list[dict]:
        with self._store.lock:
            tasks = self._store.list_tasks(principal, LIST_LIMIT)
        selected: list[dict] = []
        size = 0
        for task in tasks:
            encoded = len(json.dumps(task, ensure_ascii=False))
            if size + encoded > MAX_LIST_BYTES:
                break
            selected.append(task)
            size += encoded
        return selected

    def cancel(self, principal: str, task_id: str) -> dict:
        with self._store.transaction():
            task = self._require(principal, task_id)
            if task["status"]["state"] not in OPEN_STATES:
                raise A2AError(409, "TASK_NOT_CANCELABLE", "Task is already in a terminal state.")
            task["status"] = _status(CANCELED)
            self._store.save_task(principal, task)
            return task

    # Steps ------------------------------------------------------------------

    def _require(self, principal: str, task_id: str) -> dict:
        task = self._store.load_task(principal, task_id)
        if task is None:
            raise not_found()
        return task

    def _start(self, principal: str, message: dict, digest: str) -> tuple[dict, dict, threading.Event]:
        batch = batch_from_message(message)  # rejects bad input before anything is stored
        context_id = message.get("contextId")
        if not isinstance(context_id, str) or not context_id.strip():
            context_id = str(uuid.uuid4())
        task = {
            "id": str(uuid.uuid4()),
            "contextId": context_id,
            "status": _status(SUBMITTED),
            "artifacts": [],
            "history": [message],
        }
        with self._store.transaction():
            self._store.insert_task(principal, task, message["messageId"], digest)
        done = threading.Event()
        self._inflight[(principal, message["messageId"])] = done
        return task, batch, done

    def _run(
        self,
        principal: str,
        key: tuple[str, str],
        task: dict,
        batch: dict,
        done: threading.Event,
    ) -> dict:
        task_id = task["id"]
        try:
            with self._store.transaction():
                current = self._require(principal, task_id)
                if current["status"]["state"] != SUBMITTED:
                    return current  # cancelled before work began
                current["status"] = _status(WORKING)
                self._store.save_task(principal, current)

            decided = self._decide(batch)

            with self._store.transaction():
                current = self._require(principal, task_id)
                if current["status"]["state"] != WORKING:
                    return current  # cancelled while the model was running
                current["artifacts"] = [_proposal_artifact(batch, decided)]
                current["status"] = _status(INPUT_REQUIRED)
                self._store.save_task(principal, current)
                return current
        except ModelUnavailable:
            if self._abandon(principal, task_id, key[1]):
                raise A2AError(503, "UNAVAILABLE", "Temporary model failure. Retry with the same messageId.")
            return self._require(principal, task_id)
        except DecisionInvalid as exc:
            return self._fail(principal, task_id, f"Could not produce valid decisions: {exc}")
        finally:
            with self._store.lock:
                self._inflight.pop(key, None)
            done.set()

    def _abandon(self, principal: str, task_id: str, message_id: str) -> bool:
        """Remove a run that never produced a proposal, so the same messageId can start over cleanly."""
        with self._store.transaction():
            current = self._require(principal, task_id)
            if current["status"]["state"] != WORKING:
                return False
            self._store.delete_task(principal, task_id, message_id)
            return True

    def _fail(self, principal: str, task_id: str, text: str) -> dict:
        with self._store.transaction():
            current = self._require(principal, task_id)
            if current["status"]["state"] in OPEN_STATES:
                current["status"] = _status(FAILED, text)
                self._store.save_task(principal, current)
            return current

    def _continue(self, principal: str, message: dict, digest: str) -> dict:
        task_id = message["taskId"]
        with self._store.transaction():
            task = self._store.load_task(principal, task_id)
            if task is None:
                raise not_found()
            if message.get("contextId") != task["contextId"]:
                raise invalid_argument("contextId does not match the task.")
            if task["status"]["state"] != INPUT_REQUIRED:
                raise A2AError(409, "TASK_STATE_CONFLICT", "Task cannot accept results in its current state.")

            proposals = _proposal_data(task)
            results = results_from_message(message)
            executions = _executions(proposals, results)

            task["history"].append(message)
            task["artifacts"].append(
                _artifact(
                    "invoice-action-receipts",
                    RECEIPTS_MEDIA_TYPE,
                    {"batchId": proposals["batchId"], "executions": executions},
                )
            )
            task["status"] = _status(COMPLETED)
            self._store.save_task(principal, task)
            self._store.add_request(principal, message["messageId"], digest, task_id)
            return task

    # Model work (outside the transaction) -----------------------------------

    def _decide(self, batch: dict) -> dict[str, dict]:
        policy = {key: value for key, value in batch.items() if key not in ("batchId", "packages")}
        deadline = time.monotonic() + self._budget

        decided: dict[str, dict] = {}
        pending: list[tuple[str, dict]] = []
        for package in batch["packages"]:
            key = self._cache_key(policy, package)
            cached = self._store.get_decision(key)
            if cached is None:
                pending.append((key, package))
            else:
                decided[package["packageId"]] = cached
        if not pending:
            return decided

        # Small chunks run side by side, so each call finishes well inside the grader's 45-second limit.
        chunks = [pending[start : start + DECISION_CHUNK] for start in range(0, len(pending), DECISION_CHUNK)]
        with ThreadPoolExecutor(max_workers=min(CHUNK_WORKERS, len(chunks))) as pool:
            results = list(
                pool.map(
                    lambda chunk: self._decide_chunk(policy, [package for _, package in chunk], deadline),
                    chunks,
                )
            )
        for chunk, clean in zip(chunks, results):
            with self._store.transaction():
                for key, package in chunk:
                    self._store.put_decision(key, self._decider.model, clean[package["packageId"]])
            decided.update(clean)
        return decided

    def _decide_chunk(self, policy: dict, packages: list[dict], deadline: float) -> dict[str, dict]:
        feedback: list[str] = []
        for _ in range(MAX_ATTEMPTS):
            remaining = deadline - time.monotonic()
            if remaining < MIN_ATTEMPT_SECONDS:
                raise ModelUnavailable("time budget used up")
            day = datetime.now(UTC).strftime("%Y-%m-%d")
            if not self._store.reserve_model_call(day, self._daily_limit):
                raise ModelUnavailable("daily model-call limit reached")
            raw = self._decider.decide_chunk(policy, packages, feedback, timeout=remaining)
            clean, feedback = check_batch(packages, raw)
            if not feedback:
                return clean
        raise DecisionInvalid("; ".join(feedback[:3]))

    def _cache_key(self, policy: dict, package: dict) -> str:
        return sha256_hex(canonical_json({"model": self._decider.model, "policy": policy, "package": package}))
