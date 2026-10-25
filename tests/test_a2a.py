"""Protocol and lifecycle tests. They use a fake decider, so no API key or network is needed."""

from __future__ import annotations

import copy
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, load_settings
from app.decider import ModelUnavailable, bracket_refs, check_decision
from app.errors import A2AError
from app.main import create_app
from app.protocol import (
    BATCH_MEDIA_TYPE,
    PROPOSALS_MEDIA_TYPE,
    RECEIPTS_MEDIA_TYPE,
    RESULTS_MEDIA_TYPE,
    canonical_json,
)
from app.service import TaskService
from app.store import Store

BASE_URL = "https://agent.example.test/a2a/"
ALICE = "alice-token-123"
BOB = "bob-token-456"
HEADERS = {"A2A-Version": "1.0", "Content-Type": "application/a2a+json"}


class FakeDecider:
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0
        self.delay = 0.0
        self.fail_next = False
        self.corrupt_next = 0
        self.action = "settle_invoice"
        self._lock = threading.Lock()

    def decide_chunk(self, policy, packages, feedback, timeout):
        with self._lock:
            self.calls += 1
            fail = self.fail_next
            corrupt = self.corrupt_next > 0
            self.fail_next = False
            if corrupt:
                self.corrupt_next -= 1
        time.sleep(self.delay)
        if fail:
            raise ModelUnavailable("fake outage")
        decisions = []
        for package in packages:
            refs = bracket_refs(package)[:3]
            if corrupt:
                refs = ["[NOT-IN-DOCUMENTS]", refs[0]]
            decisions.append(
                {
                    "packageId": package["packageId"],
                    "action": self.action,
                    "vendorName": package["vendor"],
                    "invoiceNumber": package["invoiceNumber"],
                    "amountMinor": 125000,
                    "currency": "INR",
                    "evidenceRefs": refs,
                    "reasoning": "The cited receipt and policy lines support this action for the test case.",
                }
            )
        return decisions


def make_package(package_id: str, invoice: str = "INV-1001") -> dict:
    return {
        "packageId": package_id,
        "vendor": "Acme Supplies Pvt Ltd",
        "invoiceNumber": invoice,
        "documents": [
            f"Invoice {invoice} from Acme Supplies Pvt Ltd for Rs 1,250.00. [INV-{package_id}-A]",
            f"Goods receipt confirmed for {invoice}. [GRN-{package_id}-B]",
            f"Amount is within delegated limit. [POL-{package_id}-C]",
            f"Cover sheet only. [COVER-{package_id}]",
        ],
    }


def batch_message(message_id="msg-1", batch_id="batch-1", packages=None) -> dict:
    return {
        "messageId": message_id,
        "role": "ROLE_USER",
        "parts": [
            {
                "mediaType": BATCH_MEDIA_TYPE,
                "data": {
                    "batchId": batch_id,
                    "policyRevision": "2026-10",
                    "packages": packages or [make_package("pkg-1"), make_package("pkg-2", "INV-1002")],
                },
            }
        ],
    }


def auth(token: str) -> dict:
    return {**HEADERS, "Authorization": f"Bearer {token}"}


def send(client, token, message, return_immediately=False):
    return client.post(
        "/a2a/message:send",
        headers=auth(token),
        json={
            "message": message,
            "configuration": {
                "returnImmediately": return_immediately,
                "historyLength": 20,
                "acceptedOutputModes": [PROPOSALS_MEDIA_TYPE, RECEIPTS_MEDIA_TYPE],
            },
        },
    )


def proposals_of(task: dict) -> dict:
    return task["artifacts"][0]["parts"][0]["data"]


def results_message(task: dict, outcomes: dict | None = None, message_id: str | None = None) -> dict:
    outcomes = outcomes or {}
    data = proposals_of(task)
    results = [
        {
            "packageId": proposal["packageId"],
            "actionId": proposal["actionId"],
            "action": proposal["action"],
            "outcome": outcomes.get(proposal["packageId"], "ACCEPTED"),
            "receiptNonce": f"nonce-{uuid.uuid4().hex}",
        }
        for proposal in data["proposals"]
    ]
    return {
        "messageId": message_id or f"res-{uuid.uuid4().hex}",
        "taskId": task["id"],
        "contextId": task["contextId"],
        "role": "ROLE_USER",
        "parts": [
            {"mediaType": RESULTS_MEDIA_TYPE, "data": {"batchId": data["batchId"], "results": results}}
        ],
    }


def settings_for(tmp_path) -> Settings:
    return Settings(
        public_base_url=BASE_URL,
        path_prefix="/a2a",
        database_path=str(tmp_path / "agent.sqlite3"),
        allowed_tokens=None,
        anthropic_api_key=None,
        model="fake-model",
        effort="medium",
        decision_budget_seconds=30.0,
        daily_model_call_limit=1000,
    )


@pytest.fixture
def env(tmp_path):
    decider = FakeDecider()
    app = create_app(settings_for(tmp_path), decider)
    client = TestClient(app, raise_server_exceptions=False)
    return client, decider


def send_ok(client, token, message):
    response = send(client, token, message)
    assert response.status_code == 200, response.text
    return response.json()["task"]


# Agent Card and envelope rules -------------------------------------------------


def test_agent_card_is_public_and_complete(env):
    client, _ = env
    response = client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    card = response.json()
    assert card["name"] and card["description"] and card["version"]
    assert isinstance(card["capabilities"], dict)
    assert card["supportedInterfaces"] == [
        {"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
    ]
    assert BATCH_MEDIA_TYPE in card["defaultInputModes"]
    assert PROPOSALS_MEDIA_TYPE in card["defaultOutputModes"]
    assert RECEIPTS_MEDIA_TYPE in card["defaultOutputModes"]
    skill = next(s for s in card["skills"] if s["id"] == "invoice_action_agent")
    assert skill["name"] and skill["description"] and skill["tags"]


def test_missing_or_bad_bearer_is_rejected(env):
    client, _ = env
    no_auth = client.post("/a2a/message:send", headers=HEADERS, json={})
    assert no_auth.status_code == 401
    assert client.get("/a2a/tasks", headers=HEADERS).status_code == 401
    assert client.get("/a2a/tasks", headers={**HEADERS, "Authorization": "Bearer "}).status_code == 401


def test_wrong_version_is_400(env):
    client, _ = env
    headers = {**auth(ALICE), "A2A-Version": "0.3"}
    assert client.get("/a2a/tasks", headers=headers).status_code == 400


def test_wrong_media_type_is_415(env):
    client, _ = env
    headers = {**auth(ALICE), "Content-Type": "application/json"}
    response = client.post("/a2a/message:send", headers=headers, json={"message": batch_message()})
    assert response.status_code == 415
    assert response.headers["content-type"].startswith("application/a2a+json")


def test_successful_responses_use_a2a_media_type(env):
    client, _ = env
    send_ok(client, ALICE, batch_message())
    listing = client.get("/a2a/tasks", headers=auth(ALICE))
    assert listing.headers["content-type"] == "application/a2a+json"


def test_no_redirect_for_trailing_slash(env):
    client, _ = env
    response = client.get("/a2a/tasks/", headers=auth(ALICE), follow_redirects=False)
    assert response.status_code == 404


# Proposal phase -----------------------------------------------------------------


def test_send_returns_input_required_with_cited_proposals(env):
    client, decider = env
    task = send_ok(client, ALICE, batch_message())
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert task["history"][0]["messageId"] == "msg-1"
    assert len(task["artifacts"]) == 1
    assert len(task["artifacts"][0]["parts"]) == 1
    data = proposals_of(task)
    assert data["batchId"] == "batch-1"
    assert [p["packageId"] for p in data["proposals"]] == ["pkg-1", "pkg-2"]
    action_ids = [p["actionId"] for p in data["proposals"]]
    assert len(set(action_ids)) == 2 and all(len(a) >= 12 for a in action_ids)
    for proposal in data["proposals"]:
        assert proposal["action"] in {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
        assert proposal["action"] in proposal["rationale"]
        assert 60 <= len(proposal["rationale"]) <= 1500
        assert len(proposal["evidenceRefs"]) == 3
        assert all(ref.startswith("[") for ref in proposal["evidenceRefs"])
        assert not any(ref.startswith("[COVER") for ref in proposal["evidenceRefs"])
    assert decider.calls == 1


def test_duplicate_message_returns_same_task_without_model_call(env):
    client, decider = env
    first = send_ok(client, ALICE, batch_message())
    same_content_reordered = {
        "parts": copy.deepcopy(batch_message()["parts"]),
        "role": "ROLE_USER",
        "messageId": "msg-1",
    }
    second = send(client, ALICE, same_content_reordered, return_immediately=True)
    assert second.status_code == 200
    assert second.json()["task"]["id"] == first["id"]
    assert second.json()["task"]["artifacts"] == first["artifacts"]
    assert decider.calls == 1


def test_conflicting_message_id_is_409_and_does_not_change_task(env):
    client, _ = env
    first = send_ok(client, ALICE, batch_message())
    changed = batch_message(packages=[make_package("pkg-9", "INV-9999")])
    response = send(client, ALICE, changed)
    assert response.status_code == 409
    assert response.json()["error"]["status"] == "IDEMPOTENCY_CONFLICT"
    again = client.get(f"/a2a/tasks/{first['id']}", headers=auth(ALICE)).json()
    assert again == first


def test_concurrent_equivalent_requests_share_one_model_call(tmp_path):
    decider = FakeDecider()
    decider.delay = 0.3
    service = TaskService(Store(str(tmp_path / "c.sqlite3")), decider, budget_seconds=30, daily_limit=100)
    message = batch_message(message_id="msg-race")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(service.handle_message, ALICE, copy.deepcopy(message)) for _ in range(6)]
        tasks = [future.result(timeout=20) for future in futures]
    assert len({task["id"] for task in tasks}) == 1
    assert all(task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED" for task in tasks)
    assert decider.calls == 1


def test_bad_batch_is_400_and_not_stored(env):
    client, _ = env
    bad = batch_message(message_id="msg-bad", packages=[make_package("dup"), make_package("dup")])
    response = send(client, ALICE, bad)
    assert response.status_code == 400
    listing = client.get("/a2a/tasks", headers=auth(ALICE)).json()
    assert listing == {"tasks": []}


def test_large_batch_is_split_into_chunks_and_every_package_is_proposed(env):
    client, decider = env
    packages = [make_package(f"pkg-{i:02d}", f"INV-{2000 + i}") for i in range(13)]
    task = send_ok(client, ALICE, batch_message(message_id="big-1", batch_id="big-batch", packages=packages))
    proposals = proposals_of(task)["proposals"]
    assert [p["packageId"] for p in proposals] == [f"pkg-{i:02d}" for i in range(13)]
    assert decider.calls == 3  # 6 + 6 + 1 packages


def test_invalid_model_output_is_repaired_by_feedback(env):
    client, decider = env
    decider.corrupt_next = 1
    task = send_ok(client, ALICE, batch_message())
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert decider.calls == 2


def test_persistent_invalid_output_fails_task(env):
    client, decider = env
    decider.corrupt_next = 10
    response = send(client, ALICE, batch_message())
    assert response.status_code == 200
    task = response.json()["task"]
    assert task["status"]["state"] == "TASK_STATE_FAILED"
    assert not task["artifacts"]


def test_model_outage_is_503_then_retry_with_same_message_succeeds(env):
    client, decider = env
    decider.fail_next = True
    response = send(client, ALICE, batch_message())
    assert response.status_code == 503
    assert client.get("/a2a/tasks", headers=auth(ALICE)).json() == {"tasks": []}
    task = send_ok(client, ALICE, batch_message())
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"


def test_decision_cache_is_reused_for_new_batch_and_message_ids(env):
    client, decider = env
    send_ok(client, ALICE, batch_message(message_id="check-1", batch_id="check-batch-1"))
    assert decider.calls == 1
    task = send_ok(client, ALICE, batch_message(message_id="save-1", batch_id="save-batch-1"))
    assert decider.calls == 1  # same package content, so no new model work
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert proposals_of(task)["batchId"] == "save-batch-1"


# Result phase -------------------------------------------------------------------


def test_all_accepted_results_complete_task_with_receipts(env):
    client, _ = env
    task = send_ok(client, ALICE, batch_message())
    response = send(client, ALICE, results_message(task))
    assert response.status_code == 200
    done = response.json()["task"]
    assert done["status"]["state"] == "TASK_STATE_COMPLETED"
    assert [a["name"] for a in done["artifacts"]] == ["invoice-action-proposals", "invoice-action-receipts"]
    receipts = done["artifacts"][1]["parts"]
    assert len(receipts) == 1 and receipts[0]["mediaType"] == RECEIPTS_MEDIA_TYPE
    executions = receipts[0]["data"]["executions"]
    assert [e["packageId"] for e in executions] == ["pkg-1", "pkg-2"]
    assert all(e["receiptNonce"].startswith("nonce-") for e in executions)
    assert len(done["history"]) == 2


def test_rejected_outcome_is_not_executed(env):
    client, _ = env
    task = send_ok(client, ALICE, batch_message())
    response = send(client, ALICE, results_message(task, {"pkg-2": "REJECTED"}))
    done = response.json()["task"]
    assert done["status"]["state"] == "TASK_STATE_COMPLETED"
    executions = done["artifacts"][1]["parts"][0]["data"]["executions"]
    assert [e["packageId"] for e in executions] == ["pkg-1"]


def test_mismatched_continuation_is_rejected_and_task_unchanged(env):
    client, _ = env
    task = send_ok(client, ALICE, batch_message())
    bad = results_message(task)
    bad["parts"][0]["data"]["results"][0]["action"] = "hold_invoice"
    response = send(client, ALICE, bad)
    assert response.status_code == 400
    current = client.get(f"/a2a/tasks/{task['id']}", headers=auth(ALICE)).json()
    assert current["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert len(current["artifacts"]) == 1


def test_wrong_context_is_rejected(env):
    client, _ = env
    task = send_ok(client, ALICE, batch_message())
    bad = results_message(task)
    bad["contextId"] = "some-other-context"
    assert send(client, ALICE, bad).status_code == 400


def test_terminal_replay_returns_same_task_and_new_result_is_409(env):
    client, _ = env
    task = send_ok(client, ALICE, batch_message())
    message = results_message(task, message_id="res-fixed")
    first = send(client, ALICE, message).json()["task"]
    replay = send(client, ALICE, copy.deepcopy(message))
    assert replay.status_code == 200
    assert replay.json()["task"] == first
    late = send(client, ALICE, results_message(task))
    assert late.status_code == 409


def test_replayed_initial_message_after_completion_returns_same_task(env):
    client, decider = env
    task = send_ok(client, ALICE, batch_message())
    send(client, ALICE, results_message(task))
    again = send_ok(client, ALICE, batch_message())
    assert again["status"]["state"] == "TASK_STATE_COMPLETED"
    assert decider.calls == 1


# Cancellation and races --------------------------------------------------------


def test_cancel_then_results_conflict(env):
    client, _ = env
    task = send_ok(client, ALICE, batch_message())
    canceled = client.post(f"/a2a/tasks/{task['id']}:cancel", headers=auth(ALICE))
    assert canceled.status_code == 200
    assert canceled.json()["status"]["state"] == "TASK_STATE_CANCELED"
    assert send(client, ALICE, results_message(task)).status_code == 409
    assert client.post(f"/a2a/tasks/{task['id']}:cancel", headers=auth(ALICE)).status_code == 409
    current = client.get(f"/a2a/tasks/{task['id']}", headers=auth(ALICE)).json()
    assert current["status"]["state"] == "TASK_STATE_CANCELED"
    assert len(current["artifacts"]) == 1


def test_results_then_cancel_conflict(env):
    client, _ = env
    task = send_ok(client, ALICE, batch_message())
    assert send(client, ALICE, results_message(task)).status_code == 200
    assert client.post(f"/a2a/tasks/{task['id']}:cancel", headers=auth(ALICE)).status_code == 409


def test_cancel_and_results_race_has_exactly_one_winner(tmp_path):
    decider = FakeDecider()
    service = TaskService(Store(str(tmp_path / "r.sqlite3")), decider, budget_seconds=30, daily_limit=100)
    task = service.handle_message(ALICE, batch_message(message_id="race-start"))
    message = results_message(task)
    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def cancel():
        barrier.wait()
        try:
            service.cancel(ALICE, task["id"])
            outcomes.append("cancel-200")
        except A2AError:
            outcomes.append("cancel-error")

    def result():
        barrier.wait()
        try:
            service.handle_message(ALICE, copy.deepcopy(message))
            outcomes.append("result-200")
        except A2AError:
            outcomes.append("result-error")

    threads = [threading.Thread(target=cancel), threading.Thread(target=result)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) in (["cancel-200", "result-error"], ["cancel-error", "result-200"])
    final = service.get(ALICE, task["id"])
    if outcomes and "result-200" in outcomes:
        assert final["status"]["state"] == "TASK_STATE_COMPLETED"
    else:
        assert final["status"]["state"] == "TASK_STATE_CANCELED"
        assert len(final["artifacts"]) == 1


# Isolation ----------------------------------------------------------------------


def test_users_cannot_read_continue_or_cancel_each_other(env):
    client, _ = env
    task = send_ok(client, ALICE, batch_message())
    not_found = {"error": {"code": 404, "status": "NOT_FOUND", "message": "Task not found."}}
    read = client.get(f"/a2a/tasks/{task['id']}", headers=auth(BOB))
    assert read.status_code == 404 and read.json() == not_found
    assert client.post(f"/a2a/tasks/{task['id']}:cancel", headers=auth(BOB)).status_code == 404
    assert send(client, BOB, results_message(task)).status_code == 404
    assert client.get("/a2a/tasks", headers=auth(BOB)).json() == {"tasks": []}
    still = client.get(f"/a2a/tasks/{task['id']}", headers=auth(ALICE)).json()
    assert still["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"


def test_same_message_id_under_different_users_is_independent(env):
    client, decider = env
    alice_task = send_ok(client, ALICE, batch_message(message_id="shared-id"))
    bob_task = send_ok(client, BOB, batch_message(message_id="shared-id"))
    assert alice_task["id"] != bob_task["id"]
    assert decider.calls == 1  # Bob's packages are identical, so the decision cache serves them
    assert [t["id"] for t in client.get("/a2a/tasks", headers=auth(BOB)).json()["tasks"]] == [bob_task["id"]]


def test_listing_returns_only_own_tasks(env):
    client, _ = env
    alice_task = send_ok(client, ALICE, batch_message(message_id="a-1"))
    listing = client.get("/a2a/tasks", headers=auth(ALICE)).json()["tasks"]
    assert [t["id"] for t in listing] == [alice_task["id"]]


def test_token_allowlist_rejects_unknown_tokens(tmp_path):
    settings = settings_for(tmp_path)
    settings = Settings(**{**settings.__dict__, "allowed_tokens": frozenset({ALICE})})
    client = TestClient(create_app(settings, FakeDecider()), raise_server_exceptions=False)
    assert client.get("/a2a/tasks", headers=auth(BOB)).status_code == 401
    assert client.get("/a2a/tasks", headers=auth(ALICE)).status_code == 200


# Pure checks ------------------------------------------------------------------


def test_check_decision_rejects_bad_amounts_and_unknown_refs():
    package = make_package("pkg-x")
    good_refs = bracket_refs(package)[:3]
    base = {
        "packageId": "pkg-x",
        "action": "settle_invoice",
        "vendorName": "Acme Supplies Pvt Ltd",
        "invoiceNumber": "INV-1001",
        "amountMinor": 125000,
        "currency": "INR",
        "evidenceRefs": good_refs,
        "reasoning": "The receipt and the policy line both support settlement in this case.",
    }
    decision, problem = check_decision(package, base)
    assert problem is None and decision["rationale"].startswith("Action settle_invoice:")
    assert check_decision(package, {**base, "amountMinor": 12.5})[1] is not None
    assert check_decision(package, {**base, "currency": "rupees"})[1] is not None
    assert check_decision(package, {**base, "evidenceRefs": good_refs[:2]})[1] is not None
    assert check_decision(package, {**base, "evidenceRefs": ["INV-PKG-X-A", "a", "b"]})[1] is not None
    cover_first = {"packageId": "pkg-y", "vendor": "Acme Supplies Pvt Ltd", "invoiceNumber": "INV-1001",
                   "documents": ["Cover sheet for this case. [COVER-Y]", "Invoice INV-1001 from Acme Supplies Pvt Ltd. [INV-Y]",
                                 "Receipt confirmed. [GRN-Y]", "Policy line. [POL-Y]"]}
    cover_refs = {**base, "packageId": "pkg-y", "evidenceRefs": ["[COVER-Y]", "[INV-Y]", "[GRN-Y]"]}
    assert "cover sheet" in check_decision(cover_first, cover_refs)[1]
    assert check_decision(cover_first, {**cover_refs, "evidenceRefs": ["[INV-Y]", "[GRN-Y]", "[POL-Y]"]})[1] is None


def test_canonical_json_sorts_nested_keys():
    assert canonical_json({"b": {"y": 1, "x": 2}, "a": [3]}) == '{"a":[3],"b":{"x":2,"y":1}}'


def test_load_settings_rejects_non_https_base():
    with pytest.raises(ValueError):
        load_settings({"PUBLIC_BASE_URL": "http://example.test/a2a/"})
    with pytest.raises(ValueError):
        load_settings({"PUBLIC_BASE_URL": "https://user:pw@example.test/a2a/"})
    assert load_settings({"PUBLIC_BASE_URL": "https://example.test/a2a/"}).path_prefix == "/a2a"
