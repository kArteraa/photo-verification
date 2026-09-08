"""Offline chat client: replays recorded responses or synthesizes them."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from app.llm.client import ChatRequest, ChatResponse, read_cached, request_key

log = logging.getLogger(__name__)

Synthesizer = Callable[[ChatRequest], ChatResponse]


class MockChatClient:
    """Returns a recorded response when one exists, otherwise a synthesized one."""

    def __init__(self, model: str, cache_dir: Path, synthesize: Synthesizer) -> None:
        self._model = model
        self._cache_dir = Path(cache_dir)
        self._synthesize = synthesize

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Replay from the cache or synthesize."""
        key = request_key(request, self._model)
        recorded = read_cached(self._cache_dir, key)
        if recorded is not None:
            log.debug("replaying recorded response %s", key)
            return recorded
        return self._synthesize(request)
