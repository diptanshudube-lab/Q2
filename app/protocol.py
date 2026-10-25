"""A2A 1.0 wire rules: media types, the Agent Card, and validation of the messages this agent accepts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from .errors import invalid_argument

A2A_VERSION = "1.0"
A2A_MEDIA_TYPE = "application/a2a+json"
BATCH_MEDIA_TYPE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-receipts+json"
RESULTS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-results+json"

ACTIONS = ("settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception")
RESULT_OUTCOMES = ("ACCEPTED", "REJECTED")

SUBMITTED = "TASK_STATE_SUBMITTED"
WORKING = "TASK_STATE_WORKING"
INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
COMPLETED = "TASK_STATE_COMPLETED"
CANCELED = "TASK_STATE_CANCELED"
FAILED = "TASK_STATE_FAILED"
OPEN_STATES = (SUBMITTED, WORKING, INPUT_REQUIRED)

MAX_PACKAGES = 48
MAX_LIST_BYTES = 400_000


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    """Compact JSON with keys sorted at every depth, so equal content always gives equal text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def semantic_hash(message: dict) -> str:
    return sha256_hex(canonical_json(message))


def build_agent_card(base_url: str) -> dict:
    return {
        "name": "Invoice Action Agent",
        "description": (
            "Reads invoice claim batches, proposes one typed payment action per invoice with cited "
            "evidence, and records receipts only for proposals the grader accepts."
        ),
        "version": "1.0.0",
        "supportedInterfaces": [
            {"url": base_url, "protocolBinding": "HTTP+JSON", "protocolVersion": A2A_VERSION},
        ],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [BATCH_MEDIA_TYPE],
        "defaultOutputModes": [PROPOSALS_MEDIA_TYPE, RECEIPTS_MEDIA_TYPE],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "invoice_action_agent",
                "description": (
                    "Chooses settle_invoice, request_approval, hold_invoice, reject_duplicate or "
                    "open_exception for each invoice package, citing the decisive document references."
                ),
                "tags": ["invoice", "accounts-payable", "payments", "evidence", "human-in-the-loop"],
            }
        ],
        "securitySchemes": {"bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}},
        "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
    }


def _non_empty_str(value: Any, limit: int) -> bool:
    return isinstance(value, str) and 0 < len(value.strip()) and len(value) <= limit


def require_message(body: dict) -> dict:
    message = body.get("message")
    if not isinstance(message, dict):
        raise invalid_argument("Body must contain a message object.")
    if not _non_empty_str(message.get("messageId"), 256):
        raise invalid_argument("message.messageId is required.")
    if message.get("role") != "ROLE_USER":
        raise invalid_argument("message.role must be ROLE_USER.")
    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        raise invalid_argument("message.parts must be a non-empty list.")
    return message


def data_part(message: dict, media_type: str) -> dict:
    matches = [
        part
        for part in message["parts"]
        if isinstance(part, dict) and part.get("mediaType") == media_type
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("data"), dict):
        raise invalid_argument(f"Message needs exactly one {media_type} part with an object in data.")
    return matches[0]["data"]


def batch_from_message(message: dict) -> dict:
    data = data_part(message, BATCH_MEDIA_TYPE)
    if not _non_empty_str(data.get("batchId"), 256):
        raise invalid_argument("batchId is required.")
    if data.get("policyRevision") in (None, ""):
        raise invalid_argument("policyRevision is required.")
    packages = data.get("packages")
    if not isinstance(packages, list) or not 1 <= len(packages) <= MAX_PACKAGES:
        raise invalid_argument(f"packages must be a list of 1 to {MAX_PACKAGES} items.")
    seen: set[str] = set()
    for package in packages:
        if not isinstance(package, dict) or not _non_empty_str(package.get("packageId"), 256):
            raise invalid_argument("Every package needs a packageId.")
        if package["packageId"] in seen:
            raise invalid_argument("packageId values must be unique within the batch.")
        seen.add(package["packageId"])
    return data


def results_from_message(message: dict) -> dict:
    data = data_part(message, RESULTS_MEDIA_TYPE)
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise invalid_argument("results must be a non-empty list.")
    for result in results:
        if not isinstance(result, dict):
            raise invalid_argument("Each result must be an object.")
        for key in ("packageId", "actionId", "action", "receiptNonce"):
            if not _non_empty_str(result.get(key), 512):
                raise invalid_argument(f"Each result needs a non-empty {key}.")
        if result.get("outcome") not in RESULT_OUTCOMES:
            raise invalid_argument("outcome must be ACCEPTED or REJECTED.")
    return data
