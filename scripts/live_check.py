"""Run one small synthetic batch through the deployed agent, then accept its proposals.

This exercises the real Claude call and prints the timing, so you can see the 45-second limit is safe.
It does not use the grader's checks. It does use your Anthropic credits (two packages, a few cents at most).

Usage:
    python scripts/live_check.py --base https://your-service.onrender.com/a2a/ --token YOUR_TOKEN
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

BATCH_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
RESULTS_MEDIA = "application/vnd.ga5.invoice-action-results+json"
PROPOSALS_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"


def sample_packages() -> list[dict]:
    # Synthetic data only. Replace nothing here with real finance records.
    return [
        {
            "packageId": "SYN-PKG-001",
            "vendor": "Sample Stationery Co",
            "invoiceNumber": "SYN-INV-7001",
            "documents": [
                "Cover sheet for synthetic case one. [SYN-COVER-1]",
                "Invoice SYN-INV-7001 from Sample Stationery Co for Rs 4,500.00, dated 2026-10-01. [SYN-INV-REF-7001]",
                "Purchase order matched and goods receipt confirmed in full. [SYN-PO-REF-7001]",
                "Amount is within the autonomous limit of Rs 50,000.00 in the policy. [SYN-POLICY-REF-7001]",
            ],
        },
        {
            "packageId": "SYN-PKG-002",
            "vendor": "Sample Logistics Ltd",
            "invoiceNumber": "SYN-INV-7002",
            "documents": [
                "Cover sheet for synthetic case two. [SYN-COVER-2]",
                "Invoice SYN-INV-7002 from Sample Logistics Ltd for Rs 9,800.00. [SYN-INV-REF-7002]",
                "Bank details changed last week; verification call is still pending. [SYN-HOLD-REF-7002]",
                "Policy: payments to changed bank details wait for verification. [SYN-POLICY-REF-7002]",
            ],
        },
    ]


def call(method: str, url: str, token: str, body: dict | None = None) -> tuple[int, dict, float]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("A2A-Version", "1.0")
    request.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        request.add_header("Content-Type", "application/a2a+json")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
            return response.status, payload, time.monotonic() - started
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}"), time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Exact base URL, for example https://host/a2a/")
    parser.add_argument("--token", required=True, help="Bearer token for this check")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    message = {
        "messageId": f"live-{uuid.uuid4()}",
        "role": "ROLE_USER",
        "parts": [
            {
                "mediaType": BATCH_MEDIA,
                "data": {"batchId": f"live-batch-{uuid.uuid4().hex[:8]}", "policyRevision": "2026-10", "packages": sample_packages()},
            }
        ],
    }
    status, body, seconds = call("POST", f"{base}/message:send", args.token, {"message": message})
    if status != 200:
        print(f"message:send failed: HTTP {status} {body}")
        return 1
    task = body["task"]
    print(f"message:send -> HTTP {status} in {seconds:.1f}s, state {task['status']['state']}")
    if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
        print(json.dumps(task, indent=2))
        return 1

    proposal_part = task["artifacts"][0]["parts"][0]
    assert proposal_part["mediaType"] == PROPOSALS_MEDIA
    for proposal in proposal_part["data"]["proposals"]:
        print(f"  {proposal['packageId']}: {proposal['action']}  refs={proposal['evidenceRefs']}")
        print(f"    {proposal['rationale']}")

    results = [
        {
            "packageId": p["packageId"],
            "actionId": p["actionId"],
            "action": p["action"],
            "outcome": "ACCEPTED",
            "receiptNonce": f"live-nonce-{uuid.uuid4().hex}",
        }
        for p in proposal_part["data"]["proposals"]
    ]
    follow_up = {
        "messageId": f"live-result-{uuid.uuid4()}",
        "taskId": task["id"],
        "contextId": task["contextId"],
        "role": "ROLE_USER",
        "parts": [{"mediaType": RESULTS_MEDIA, "data": {"batchId": proposal_part["data"]["batchId"], "results": results}}],
    }
    status, body, seconds = call("POST", f"{base}/message:send", args.token, {"message": follow_up})
    if status != 200:
        print(f"results failed: HTTP {status} {body}")
        return 1
    done = body["task"]
    print(f"results -> HTTP {status} in {seconds:.1f}s, state {done['status']['state']}")
    receipts = [a for a in done["artifacts"] if a["parts"][0]["mediaType"] == RECEIPTS_MEDIA]
    if receipts:
        print(f"  receipts recorded for {len(receipts[0]['parts'][0]['data']['executions'])} package(s)")
    return 0 if done["status"]["state"] == "TASK_STATE_COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
