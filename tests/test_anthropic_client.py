import base64
from types import SimpleNamespace

import anthropic
import pytest

from app.core.errors import LlmError
from app.llm.anthropic_client import AnthropicChatClient, build_kwargs, parse_message
from app.llm.client import ChatRequest, read_cached, request_key

TOOL = {"name": "submit_verdicts", "description": "d", "input_schema": {"type": "object"}}


class FakeMessages:
    def __init__(self, content=None, error=None):
        self.calls = []
        self.content = content or []
        self.error = error

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.content)


def fake_sdk(content=None, error=None):
    messages = FakeMessages(content, error)
    return SimpleNamespace(messages=messages), messages


def test_build_kwargs_text_only():
    request = ChatRequest("sys", "hello", None, None, 50)
    kwargs = build_kwargs("model-x", request)
    assert kwargs == {
        "model": "model-x",
        "max_tokens": 50,
        "system": "sys",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    }


def test_build_kwargs_with_image_and_tool():
    request = ChatRequest("sys", "look", b"\xff\xd8jpeg", TOOL, 50)
    kwargs = build_kwargs("m", request)
    content = kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(content[0]["source"]["data"]) == b"\xff\xd8jpeg"
    assert content[1] == {"type": "text", "text": "look"}
    assert kwargs["tools"] == [TOOL]
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_verdicts"}


def test_parse_message_collects_text_and_tool_input():
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="a"),
            SimpleNamespace(type="tool_use", input={"accepted": False}),
            SimpleNamespace(type="text", text="b"),
        ]
    )
    response = parse_message(message)
    assert response.text == "ab"
    assert response.tool_input == {"accepted": False}
    assert response.backend == "anthropic"


def test_complete_records_response(tmp_path):
    sdk, messages = fake_sdk(content=[SimpleNamespace(type="text", text="ИТОГ: ПРИНЯТО")])
    client = AnthropicChatClient("m", tmp_path, sdk_client=sdk)
    request = ChatRequest("sys", "report", None, None, 20)
    response = client.complete(request)
    assert response.text == "ИТОГ: ПРИНЯТО"
    assert messages.calls[0]["model"] == "m"
    recorded = read_cached(tmp_path, request_key(request, "m"))
    assert recorded.text == "ИТОГ: ПРИНЯТО"
    assert recorded.backend == "recorded"


def test_api_error_becomes_llm_error(tmp_path):
    sdk, _ = fake_sdk(error=anthropic.AnthropicError("boom"))
    client = AnthropicChatClient("m", tmp_path, sdk_client=sdk)
    with pytest.raises(LlmError, match="boom"):
        client.complete(ChatRequest("sys", "report", None, None, 20))
    assert not list(tmp_path.glob("*.json"))
