"""Selection of the chat client: real API, or the offline mock when requested or keyless."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from app.llm.client import ChatClient
from app.llm.mock_client import MockChatClient, Synthesizer

log = logging.getLogger(__name__)

API_KEY_VARIABLE = "ANTHROPIC_API_KEY"


def build_client(mock: bool, model: str, cache_dir: Path, synthesize: Synthesizer) -> ChatClient:
    """Return the Anthropic client, or the mock when ``mock`` is set or no API key is present."""
    if mock:
        log.info("using the mock chat client")
        return MockChatClient(model, cache_dir, synthesize)
    if not os.environ.get(API_KEY_VARIABLE):
        log.warning("%s is not set; falling back to the mock chat client", API_KEY_VARIABLE)
        return MockChatClient(model, cache_dir, synthesize)
    from app.llm.anthropic_client import AnthropicChatClient

    return AnthropicChatClient(model, cache_dir)
