"""Zero-shot vision-language baseline: prompt, structured answer schema and the mock oracle."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import cv2
import numpy as np

from app.conclusion.verify import extract_ids
from app.core.report import Kind
from app.core.spec import Requirement, Spec
from app.llm.client import BACKEND_MOCK, ChatRequest, ChatResponse

TOOL_NAME = "submit_verdicts"
MAX_SIDE = 512
JPEG_QUALITY = 85
MAX_TOKENS = 1024
MISS_PROBABILITY = 0.2
FALSE_ALARM_PROBABILITY = 0.03
INVENTED_TAG_PROBABILITY = 0.1

SYSTEM_PROMPT = """Ты проверяешь фотографию пользователя на соответствие набору требований. \
Осмотри изображение и для каждого требования вынеси вердикт pass или fail, кратко объяснив \
причину. Затем напиши связное заключение по-русски: перечисли нарушения, помечая каждое тегом \
в квадратных скобках с id требования, например [frontal_pose]. Требования, которые выполнены, \
тегами не помечай. Ответ верни только через инструмент submit_verdicts."""


def describe_requirement(requirement: Requirement) -> str:
    """One line of the requirement list shown to the model."""
    predicate = requirement.predicate
    kind = "обязательное" if requirement.kind is Kind.HARD else "рекомендательное"
    thresholds = ", ".join(
        f"{key} {predicate.op} {value}"
        for key, value in zip(
            predicate.keys
            if predicate.op == "abs_le"
            else (predicate.key,) * len(predicate.values),
            predicate.values,
            strict=False,
        )
    )
    return f"- {requirement.id} ({kind}): {requirement.title}; критерий: {thresholds}"


def user_text(spec: Spec) -> str:
    """Requirement list for the prompt."""
    lines = [f"Сценарий: {spec.description}.", "Требования:"]
    lines += [describe_requirement(requirement) for requirement in spec.requirements]
    return "\n".join(lines)


def tool_schema(spec: Spec) -> dict:
    """Tool definition forcing a structured answer."""
    return {
        "name": TOOL_NAME,
        "description": "Вердикты по каждому требованию и связное заключение.",
        "input_schema": {
            "type": "object",
            "properties": {
                "accepted": {"type": "boolean"},
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "enum": list(spec.ids)},
                            "status": {"type": "string", "enum": ["pass", "fail"]},
                            "reason": {"type": "string"},
                        },
                        "required": ["id", "status", "reason"],
                    },
                },
                "explanation": {"type": "string"},
            },
            "required": ["accepted", "verdicts", "explanation"],
        },
    }


def encode_image(bgr: np.ndarray) -> bytes:
    """Downscale to the maximum side and JPEG-encode for the API."""
    height, width = bgr.shape[:2]
    scale = min(1.0, MAX_SIDE / max(height, width))
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("cannot encode image")
    return buffer.tobytes()


def vlm_request(spec: Spec, image_jpeg: bytes) -> ChatRequest:
    """Request with the image, the requirement list and the answer tool."""
    return ChatRequest(
        system=SYSTEM_PROMPT,
        text=user_text(spec),
        image_jpeg=image_jpeg,
        tool=tool_schema(spec),
        max_tokens=MAX_TOKENS,
    )


def image_digest(image_jpeg: bytes) -> str:
    """Identifier of an encoded image."""
    return hashlib.sha256(image_jpeg).hexdigest()


def mock_oracle(
    spec: Spec, truth_by_digest: dict[str, str], seed: int
) -> Callable[[ChatRequest], ChatResponse]:
    """Synthesizer imitating a noisy model that knows the label of every image.

    The expected violation is missed with a fixed probability, every other
    requirement raises a false alarm with a small probability, and the
    explanation sometimes tags a requirement the verdicts did not fail.
    """

    def synthesize(request: ChatRequest) -> ChatResponse:
        digest = image_digest(request.image_jpeg or b"")
        truth = truth_by_digest.get(digest, "")
        rng = np.random.default_rng(int(digest[:8], 16) ^ seed)
        verdicts = []
        for requirement_id in spec.ids:
            if requirement_id == truth:
                failed = rng.random() >= MISS_PROBABILITY
            else:
                failed = rng.random() < FALSE_ALARM_PROBABILITY
            verdicts.append(
                {
                    "id": requirement_id,
                    "status": "fail" if failed else "pass",
                    "reason": "нарушение заметно на снимке" if failed else "требование выполнено",
                }
            )
        failed_ids = [item["id"] for item in verdicts if item["status"] == "fail"]
        hard_failed = [item for item in failed_ids if spec.by_id(item).kind is Kind.HARD]
        tagged = list(failed_ids)
        if rng.random() < INVENTED_TAG_PROBABILITY:
            passing = [item for item in spec.ids if item not in failed_ids]
            if passing:
                tagged.append(str(rng.choice(passing)))
        explanation = (
            "На снимке нарушены требования: " + ", ".join(f"[{item}]" for item in tagged) + "."
            if tagged
            else "Все требования выполнены."
        )
        payload = {"accepted": not hard_failed, "verdicts": verdicts, "explanation": explanation}
        return ChatResponse(explanation, payload, BACKEND_MOCK)

    return synthesize


def parse_answer(response: ChatResponse, spec: Spec) -> dict:
    """Normalize the model answer into statuses, acceptance and claimed ids."""
    payload = response.tool_input or {}
    statuses = {requirement_id: "missing" for requirement_id in spec.ids}
    for item in payload.get("verdicts", []):
        if item.get("id") in statuses and item.get("status") in ("pass", "fail"):
            statuses[item["id"]] = item["status"]
    failed = {requirement_id for requirement_id, status in statuses.items() if status == "fail"}
    explanation = str(payload.get("explanation", response.text))
    claimed = extract_ids(explanation, set(spec.ids))
    accepted = payload.get("accepted")
    if not isinstance(accepted, bool):
        accepted = not any(spec.by_id(item).kind is Kind.HARD for item in failed)
    return {"statuses": statuses, "failed": failed, "claimed": claimed, "accepted": accepted}
