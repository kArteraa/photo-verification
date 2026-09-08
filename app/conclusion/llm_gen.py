"""Conclusion generation through a language model with verification and fallback.

The generator receives only the :class:`Report`; the image never reaches this
module.  The model answer is verified against the report, regenerated once
with the list of discrepancies, and finally replaced by the template text
when the model still disagrees with the facts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace

from app.conclusion.prompts import RETRY_PREFIX, SYSTEM_PROMPT
from app.conclusion.template_gen import render_conclusion
from app.conclusion.verify import VerificationResult, verify_conclusion
from app.core.errors import LlmError
from app.core.report import Report
from app.llm.client import BACKEND_MOCK, ChatClient, ChatRequest, ChatResponse

log = logging.getLogger(__name__)

MAX_TOKENS = 1024
SOURCE_TEMPLATE = "template"
RETRY_SUFFIX = "_retry"


@dataclass(frozen=True)
class Conclusion:
    """Conclusion text with its origin and verification outcome."""

    text: str
    source: str
    verification: VerificationResult


def conclusion_request(report: Report) -> ChatRequest:
    """Request carrying the system instruction and the JSON report only."""
    return ChatRequest(
        system=SYSTEM_PROMPT,
        text=report.to_json(),
        image_jpeg=None,
        tool=None,
        max_tokens=MAX_TOKENS,
    )


def generate_conclusion(report: Report, client: ChatClient) -> Conclusion:
    """Ask the model, verify, retry once, then fall back to the template."""
    request = conclusion_request(report)
    attempt = _verified_attempt(client, request, report)
    if attempt is not None:
        return attempt
    return _fallback(report)


def template_synthesizer(request: ChatRequest) -> ChatResponse:
    """Mock model behaviour: render the template conclusion for the report in the request."""
    report = Report.from_dict(json.loads(request.text[request.text.index("{") :]))
    return ChatResponse(render_conclusion(report), None, BACKEND_MOCK)


def _verified_attempt(
    client: ChatClient, request: ChatRequest, report: Report
) -> Conclusion | None:
    response = _complete(client, request)
    if response is None:
        return None
    result = verify_conclusion(response.text, report)
    if result.ok:
        return Conclusion(response.text, response.backend, result)
    log.warning("conclusion rejected by verification: %s", "; ".join(result.errors))
    retry = replace(
        request, text=RETRY_PREFIX.format(errors="; ".join(result.errors)) + request.text
    )
    response = _complete(client, retry)
    if response is None:
        return None
    result = verify_conclusion(response.text, report)
    if result.ok:
        return Conclusion(response.text, response.backend + RETRY_SUFFIX, result)
    log.warning("regenerated conclusion rejected again: %s", "; ".join(result.errors))
    return None


def _complete(client: ChatClient, request: ChatRequest) -> ChatResponse | None:
    try:
        return client.complete(request)
    except LlmError as exc:
        log.warning("language model unavailable: %s", exc)
        return None


def _fallback(report: Report) -> Conclusion:
    log.warning("using the template conclusion")
    text = render_conclusion(report)
    return Conclusion(text, SOURCE_TEMPLATE, verify_conclusion(text, report))
