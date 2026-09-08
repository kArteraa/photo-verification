from pathlib import Path

import numpy as np
import pandas as pd

from app.analyzers import REGISTRY
from app.core.spec import load_spec
from app.llm.client import ChatResponse
from experiments.run_vlm_baseline import stratified_sample
from experiments.vlm import (
    TOOL_NAME,
    encode_image,
    image_digest,
    mock_oracle,
    parse_answer,
    tool_schema,
    user_text,
    vlm_request,
)

SPEC = load_spec(Path("app/specs/document_photo.yaml"), REGISTRY)


def test_request_shape():
    request = vlm_request(SPEC, b"jpeg")
    assert request.image_jpeg == b"jpeg"
    assert request.tool["name"] == TOOL_NAME
    ids = request.tool["input_schema"]["properties"]["verdicts"]["items"]["properties"]["id"][
        "enum"
    ]
    assert ids == list(SPEC.ids)
    assert "frontal_pose" in user_text(SPEC)
    assert "abs_le" in user_text(SPEC)
    assert tool_schema(SPEC)["input_schema"]["required"] == ["accepted", "verdicts", "explanation"]


def test_encode_image_downscales():
    big = np.zeros((1200, 800, 3), np.uint8)
    data = encode_image(big)
    assert data[:2] == b"\xff\xd8"
    assert len(image_digest(data)) == 64


def test_mock_oracle_is_deterministic_and_label_aware():
    image = b"image-bytes"
    truth = {image_digest(image): "face_sharpness"}
    oracle = mock_oracle(SPEC, truth, seed=1)
    first = oracle(vlm_request(SPEC, image))
    second = oracle(vlm_request(SPEC, image))
    assert first == second
    assert first.backend == "mock"
    answer = parse_answer(first, SPEC)
    assert set(answer["statuses"]) == set(SPEC.ids)
    fails = 0
    for i in range(40):
        data = f"img{i}".encode()
        oracle = mock_oracle(SPEC, {image_digest(data): "face_sharpness"}, seed=1)
        fails += (
            parse_answer(oracle(vlm_request(SPEC, data)), SPEC)["statuses"]["face_sharpness"]
            == "fail"
        )
    assert fails >= 24


def test_parse_answer_handles_missing_fields():
    response = ChatResponse(
        "нарушено [eyes_open]",
        {"verdicts": [{"id": "eyes_open", "status": "fail", "reason": "r"}]},
        "x",
    )
    answer = parse_answer(response, SPEC)
    assert answer["failed"] == {"eyes_open"}
    assert answer["claimed"] == {"eyes_open"}
    assert answer["accepted"] is False
    assert answer["statuses"]["single_face"] == "missing"


def test_stratified_sample_keeps_every_class():
    labels = pd.DataFrame(
        {"cls": ["clean"] * 50 + ["blur_face"] * 30 + ["dark"] * 2, "file": range(82)}
    )
    sample = stratified_sample(labels, 20, seed=0)
    assert len(sample) <= 20
    assert set(sample["cls"]) == {"clean", "blur_face", "dark"}
    assert stratified_sample(labels, 0, seed=0).equals(labels)
