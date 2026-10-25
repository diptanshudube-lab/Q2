"""Checks the real SDK call shape with a mocked HTTP transport. No network and no API key are needed."""

from __future__ import annotations

import json

import anthropic
import httpx2 as httpx  # anthropic 1.x runs on httpx2; an httpx client is rejected
import pytest

from app.decider import (
    DECISION_SCHEMA,
    SYSTEM_PROMPT,
    AnthropicDecider,
    ModelUnavailable,
)
from app.protocol import ACTIONS

PACKAGE = {
    "packageId": "pkg-1",
    "vendor": "Acme Supplies Pvt Ltd",
    "invoiceNumber": "INV-1001",
    "documents": [
        "Invoice INV-1001 from Acme Supplies Pvt Ltd for Rs 1,250.00. [INV-A]",
        "Goods receipt confirmed. [GRN-B]",
        "Within delegated limit. [POL-C]",
    ],
}


def model_reply(decisions: list[dict]) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": json.dumps({"decisions": decisions})}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


def decider_with(handler) -> AnthropicDecider:
    decider = AnthropicDecider("sk-ant-test-not-real", "claude-opus-5-5", "medium")
    decider._client = anthropic.Anthropic(
        api_key="sk-ant-test-not-real",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return decider


def test_request_uses_structured_output_and_effort():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json=model_reply([]))

    decider = decider_with(handler)
    decider.decide_chunk({"policyRevision": "2026-10"}, [PACKAGE], [], timeout=20.0)

    body = seen["body"]
    assert seen["url"].endswith("/v1/messages")
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": DECISION_SCHEMA}
    assert "tool_choice" not in body  # forced tool use is rejected by current models
    assert body["system"] == SYSTEM_PROMPT
    user_text = body["messages"][0]["content"]
    assert json.loads(user_text)["packages"][0]["packageId"] == "pkg-1"
    assert "sk-ant" not in user_text  # no credentials ever enter the prompt
    assert seen["key"] == "sk-ant-test-not-real"


def test_schema_enum_matches_actions():
    enum = DECISION_SCHEMA["properties"]["decisions"]["items"]["properties"]["action"]["enum"]
    assert tuple(enum) == ACTIONS


def test_parses_text_reply_into_decisions():
    decision = {
        "packageId": "pkg-1",
        "action": "settle_invoice",
        "vendorName": "Acme Supplies Pvt Ltd",
        "invoiceNumber": "INV-1001",
        "amountMinor": 125000,
        "currency": "INR",
        "evidenceRefs": ["[INV-A]", "[GRN-B]", "[POL-C]"],
        "reasoning": "The invoice matches the receipt and sits inside the delegated limit.",
    }
    decider = decider_with(lambda request: httpx.Response(200, json=model_reply([decision])))
    assert decider.decide_chunk({}, [PACKAGE], [], timeout=20.0) == [decision]


def test_api_errors_become_model_unavailable():
    decider = decider_with(lambda request: httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}}))
    with pytest.raises(ModelUnavailable):
        decider.decide_chunk({}, [PACKAGE], [], timeout=20.0)


def test_non_json_reply_returns_no_decisions():
    reply = model_reply([])
    reply["content"] = [{"type": "text", "text": "not json"}]
    decider = decider_with(lambda request: httpx.Response(200, json=reply))
    assert decider.decide_chunk({}, [PACKAGE], [], timeout=20.0) == []
