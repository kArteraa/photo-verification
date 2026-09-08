"""Language model client interface and the on-disk cache of recorded responses.

Every request is identified by a key derived from its full content and the
model name, so a response recorded by the real client can later be replayed
by the mock client for the identical request.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

BACKEND_ANTHROPIC = "anthropic"
BACKEND_RECORDED = "recorded"
BACKEND_MOCK = "mock"


@dataclass(frozen=True)
class ChatRequest:
    """One request to a chat model."""

    system: str
    text: str
    image_jpeg: bytes | None
    tool: dict[str, Any] | None
    max_tokens: int


@dataclass(frozen=True)
class ChatResponse:
    """Model output: free text and, when a tool was requested, its structured input."""

    text: str
    tool_input: dict[str, Any] | None
    backend: str


class ChatClient(Protocol):
    """Anything that can answer a :class:`ChatRequest`."""

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Return the model response for the request."""


def request_key(request: ChatRequest, model: str) -> str:
    """Stable identifier of a request for the response cache."""
    image_digest = (
        hashlib.sha256(request.image_jpeg).hexdigest() if request.image_jpeg is not None else None
    )
    payload = {
        "model": model,
        "system": request.system,
        "text": request.text,
        "image": image_digest,
        "tool": request.tool,
        "max_tokens": request.max_tokens,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_path(cache_dir: Path, key: str) -> Path:
    """Location of the cached record for a request key."""
    return Path(cache_dir) / f"{key}.json"


def write_cached(
    cache_dir: Path, key: str, model: str, request: ChatRequest, response: ChatResponse
) -> Path:
    """Persist a response together with the request that produced it."""
    path = cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "key": key,
        "model": model,
        "system": request.system,
        "text": request.text,
        "has_image": request.image_jpeg is not None,
        "tool": request.tool["name"] if request.tool else None,
        "response": {
            "text": response.text,
            "tool_input": response.tool_input,
            "backend": response.backend,
        },
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_cached(cache_dir: Path, key: str) -> ChatResponse | None:
    """Load a recorded response, or None when the request was never recorded."""
    path = cache_path(cache_dir, key)
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    response = record["response"]
    return ChatResponse(response["text"], response["tool_input"], BACKEND_RECORDED)
