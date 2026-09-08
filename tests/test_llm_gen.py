from app.conclusion.llm_gen import (
    conclusion_request,
    generate_conclusion,
    template_synthesizer,
)
from app.conclusion.prompts import RETRY_PREFIX, SYSTEM_PROMPT
from app.conclusion.template_gen import render_conclusion
from app.core.errors import LlmError
from app.llm.client import ChatResponse
from app.llm.mock_client import MockChatClient
from tests.test_template_gen import rejected_report


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResponse(item, None, "anthropic")


def good_text():
    return render_conclusion(rejected_report())


def bad_text():
    return good_text().replace("yaw=17.3°", "yaw=42.0°")


def test_request_contains_only_the_report():
    request = conclusion_request(rejected_report())
    assert request.system == SYSTEM_PROMPT
    assert request.image_jpeg is None
    assert request.tool is None
    assert '"frontal_pose"' in request.text


def test_first_answer_accepted():
    client = ScriptedClient([good_text()])
    conclusion = generate_conclusion(rejected_report(), client)
    assert conclusion.source == "anthropic"
    assert conclusion.verification.ok
    assert len(client.requests) == 1


def test_bad_then_good_uses_retry_with_errors():
    client = ScriptedClient([bad_text(), good_text()])
    conclusion = generate_conclusion(rejected_report(), client)
    assert conclusion.source == "anthropic_retry"
    assert conclusion.verification.ok
    retry = client.requests[1]
    assert retry.text.startswith(RETRY_PREFIX.split("{")[0])
    assert "42.0" in retry.text
    assert retry.text.endswith(client.requests[0].text)


def test_bad_twice_falls_back_to_template(caplog):
    client = ScriptedClient([bad_text(), bad_text()])
    conclusion = generate_conclusion(rejected_report(), client)
    assert conclusion.source == "template"
    assert conclusion.text == good_text()
    assert conclusion.verification.ok
    assert "template conclusion" in caplog.text


def test_llm_error_falls_back_to_template():
    client = ScriptedClient([LlmError("down")])
    conclusion = generate_conclusion(rejected_report(), client)
    assert conclusion.source == "template"
    assert len(client.requests) == 1


def test_template_synthesizer_through_mock_client(tmp_path):
    client = MockChatClient("m", tmp_path, template_synthesizer)
    conclusion = generate_conclusion(rejected_report(), client)
    assert conclusion.source == "mock"
    assert conclusion.text == good_text()
    assert conclusion.verification.ok
