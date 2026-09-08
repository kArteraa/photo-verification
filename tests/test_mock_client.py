from app.llm.anthropic_client import AnthropicChatClient
from app.llm.client import ChatRequest, ChatResponse, request_key, write_cached
from app.llm.factory import API_KEY_VARIABLE, build_client
from app.llm.mock_client import MockChatClient


def synthesize(request):
    return ChatResponse(f"synth:{request.text}", None, "mock")


def test_mock_synthesizes_on_cache_miss(tmp_path):
    client = MockChatClient("m", tmp_path, synthesize)
    response = client.complete(ChatRequest("s", "hello", None, None, 5))
    assert response == ChatResponse("synth:hello", None, "mock")


def test_mock_replays_recorded_response(tmp_path):
    request = ChatRequest("s", "hello", None, None, 5)
    write_cached(
        tmp_path, request_key(request, "m"), "m", request, ChatResponse("real", None, "anthropic")
    )
    client = MockChatClient("m", tmp_path, synthesize)
    response = client.complete(request)
    assert response.text == "real"
    assert response.backend == "recorded"
    other = MockChatClient("other-model", tmp_path, synthesize)
    assert other.complete(request).backend == "mock"


def test_factory_prefers_mock_flag(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_VARIABLE, "key")
    assert isinstance(build_client(True, "m", tmp_path, synthesize), MockChatClient)


def test_factory_falls_back_without_key(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)
    client = build_client(False, "m", tmp_path, synthesize)
    assert isinstance(client, MockChatClient)
    assert API_KEY_VARIABLE in caplog.text


def test_factory_builds_real_client_with_key(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_VARIABLE, "key")
    client = build_client(False, "m", tmp_path, synthesize)
    assert isinstance(client, AnthropicChatClient)
