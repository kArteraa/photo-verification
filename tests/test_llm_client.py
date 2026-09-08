import json

from app.llm.client import (
    BACKEND_RECORDED,
    ChatRequest,
    ChatResponse,
    cache_path,
    read_cached,
    request_key,
    write_cached,
)

TOOL = {"name": "submit", "description": "d", "input_schema": {"type": "object"}}


def request(**overrides):
    base = {"system": "s", "text": "t", "image_jpeg": None, "tool": None, "max_tokens": 10}
    base.update(overrides)
    return ChatRequest(**base)


def test_key_is_stable_and_sensitive_to_every_field():
    base = request_key(request(), "m")
    assert base == request_key(request(), "m")
    assert len(base) == 64
    assert base != request_key(request(), "other-model")
    assert base != request_key(request(text="u"), "m")
    assert base != request_key(request(system="x"), "m")
    assert base != request_key(request(image_jpeg=b"jpeg"), "m")
    assert base != request_key(request(tool=TOOL), "m")
    assert base != request_key(request(max_tokens=11), "m")


def test_cache_round_trip(tmp_path):
    key = request_key(request(tool=TOOL), "m")
    response = ChatResponse("ответ", {"accepted": True}, "anthropic")
    path = write_cached(tmp_path / "cache", key, "m", request(tool=TOOL), response)
    assert path == cache_path(tmp_path / "cache", key)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["model"] == "m"
    assert record["tool"] == "submit"
    assert record["has_image"] is False
    replayed = read_cached(tmp_path / "cache", key)
    assert replayed == ChatResponse("ответ", {"accepted": True}, BACKEND_RECORDED)


def test_missing_cache_entry(tmp_path):
    assert read_cached(tmp_path, "nope") is None
