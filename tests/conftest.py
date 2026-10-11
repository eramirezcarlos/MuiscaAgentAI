"""Shared test helpers.

Nothing here touches the network: the Muisca API is replaced by canned responses with
the same shape as the real ones, and the model is replaced by a script of replies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from muisca_agent.muisca_api import MuiscaApi

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text())


ROUTES = {
    "/api/public/properties/search": "search",
    "/api/public/properties/28": "property_28",
    "/api/public/property-types/lookup": "types",
    "/api/public/property-statuses/lookup": "statuses",
    "/api/public/cities-with-properties": "cities",
    "/api/public/agents": "agents",
}


class FakeMuisca:
    """Stands in for the Laravel API and remembers every request it received."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        fixture = ROUTES.get(request.url.path)
        if fixture:
            return httpx.Response(200, json=load(fixture))
        if request.url.path.startswith("/api/public/properties/"):
            return httpx.Response(404, json={"message": "Property not found"})
        return httpx.Response(500, text="<html>Server Error</html>")

    @property
    def last_params(self) -> dict[str, str]:
        return dict(self.requests[-1].url.params)


@pytest.fixture
def muisca() -> FakeMuisca:
    return FakeMuisca()


@pytest.fixture
def api(muisca: FakeMuisca) -> MuiscaApi:
    http = httpx.Client(transport=httpx.MockTransport(muisca.handler), headers={"Accept": "application/json"})
    return MuiscaApi(base_url="http://muisca.test", http=http)


# ------------------------------------------------------------------ fake model


def tool_call(call_id: str, name: str, tool_input: dict[str, Any]) -> ToolUseBlock:
    return ToolUseBlock(id=call_id, name=name, input=tool_input, type="tool_use")


def reply(*blocks: Any, stop_reason: str = "end_turn") -> Message:
    """Builds a real SDK Message, so the agent is tested against the real types."""
    content = [TextBlock(text=b, type="text", citations=None) if isinstance(b, str) else b for b in blocks]
    return Message(
        id="msg_test",
        content=content,
        model="test-model",
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=100, output_tokens=20),
    )


class FakeLLM:
    """Returns scripted replies in order and records what the agent sent each time."""

    def __init__(self, *replies: Message) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.messages = self  # so that llm.messages.create(...) works like the SDK

    def create(self, **kwargs: Any) -> Message:
        # Copy the list: the agent keeps appending to the same one after this call.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._replies.pop(0)
