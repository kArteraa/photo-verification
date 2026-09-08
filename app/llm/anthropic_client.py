"""Chat client on the Anthropic Messages API that records every response."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import anthropic

from app.core.errors import LlmError
from app.llm.client import (
    BACKEND_ANTHROPIC,
    ChatRequest,
    ChatResponse,
    request_key,
    write_cached,
)

log = logging.getLogger(__name__)


class AnthropicChatClient:
    """Sends requests to the Messages API and caches the answers on disk."""

    def __init__(self, model: str, cache_dir: Path, sdk_client: Any | None = None) -> None:
        self._model = model
        self._cache_dir = Path(cache_dir)
        self._sdk = sdk_client if sdk_client is not None else anthropic.Anthropic()

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Call the API once and record the response."""
        try:
            message = self._sdk.messages.create(**build_kwargs(self._model, request))
        except anthropic.AnthropicError as exc:
            raise LlmError(f"anthropic request failed: {exc}") from exc
        response = parse_message(message)
        key = request_key(request, self._model)
        write_cached(self._cache_dir, key, self._model, request, response)
        log.debug("recorded response %s", key)
        return response


def build_kwargs(model: str, request: ChatRequest) -> dict[str, Any]:
    """Keyword arguments of ``messages.create`` for the request."""
    content: list[dict[str, Any]] = []
    if request.image_jpeg is not None:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(request.image_jpeg).decode("ascii"),
                },
            }
        )
    content.append({"type": "text", "text": request.text})
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_tokens,
        "system": request.system,
        "messages": [{"role": "user", "content": content}],
    }
    if request.tool is not None:
        kwargs["tools"] = [request.tool]
        kwargs["tool_choice"] = {"type": "tool", "name": request.tool["name"]}
    return kwargs


def parse_message(message: Any) -> ChatResponse:
    """Collect text blocks and the first tool call from an API message."""
    text = "".join(block.text for block in message.content if block.type == "text")
    tool_input = next(
        (dict(block.input) for block in message.content if block.type == "tool_use"), None
    )
    return ChatResponse(text, tool_input, BACKEND_ANTHROPIC)
