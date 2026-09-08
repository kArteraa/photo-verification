from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.analyzers import REGISTRY
from app.conclusion.template_gen import render_conclusion
from app.core.engine import Engine
from app.core.pipeline import check_image
from app.core.spec import load_spec
from app.imageio import write_bgr
from app.llm.client import ChatResponse
from experiments.run_llm_conclusions import conclusion_row, summarize_conclusions
from tests.helpers import FakeBackend, frontal_landmarks

SPEC = load_spec(Path("app/specs/document_photo.yaml"), REGISTRY)


def frame():
    return pd.DataFrame(
        {
            "file": ["a", "b", "c", "d", "e"],
            "cls": ["clean", "blur_face", "dark", "clean", "bright"],
            "expected_fail_id": ["", "face_sharpness", "exposure_ok", "", "exposure_ok"],
            "source": ["anthropic", "anthropic_retry", "template", "anthropic", "anthropic"],
            "backend": ["anthropic"] * 5,
            "verified_ok": [True] * 5,
            "claimed": ["", "face_sharpness", "exposure_ok", "headroom", "face_sharpness"],
            "failed": ["", "face_sharpness", "exposure_ok", "headroom", "face_sharpness"],
        }
    )


def test_summarize_paths_and_claims():
    summary = summarize_conclusions(frame(), SPEC)
    assert summary["n"] == 5
    assert summary["backends"] == ["anthropic"]
    assert summary["is_mock"] is False
    assert summary["first_try_rate"] == pytest.approx(3 / 5)
    assert summary["retry_rate"] == pytest.approx(1 / 5)
    assert summary["fallback_rate"] == pytest.approx(1 / 5)
    assert summary["verified_rate"] == 1.0
    claims = summary["claims"]
    assert claims["consistency"] == 1.0
    assert claims["completeness"] == pytest.approx(4 / 5)


def test_summarize_marks_mock():
    table = frame()
    table["backend"] = ["mock"] * 5
    table["source"] = ["mock"] * 5
    summary = summarize_conclusions(table, SPEC)
    assert summary["is_mock"] is True
    assert summary["first_try_rate"] == 1.0


class EchoTemplateClient:
    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        from app.conclusion.llm_gen import template_synthesizer

        return ChatResponse(template_synthesizer(request).text, None, "anthropic")


def test_conclusion_row_with_scripted_client(tmp_path):
    size = 512
    image = np.full((size, size, 3), 128, np.uint8)
    image[106:406, 106:406] = np.random.default_rng(0).integers(40, 220, (300, 300, 3), np.uint8)
    write_bgr(tmp_path / "img.jpg", image, quality=95)
    mask = np.zeros((size, size), np.float32)
    mask[80:512, 90:422] = 1.0
    backend = FakeBackend([frontal_landmarks((256, 256), 300, size, size)], mask)
    label = pd.Series({"file": "img.jpg", "cls": "clean", "expected_fail_id": ""})
    client = EchoTemplateClient()
    row, text = conclusion_row(label, tmp_path, Engine(SPEC), backend, client)
    assert client.calls == 1
    assert row["source"] == "anthropic"
    assert row["backend"] == "anthropic"
    assert row["verified_ok"] is True
    assert text.startswith("ИТОГ:")
    assert set(row["claimed"].split(";")) - {""} <= set(SPEC.ids)
    report = check_image(tmp_path / "img.jpg", Engine(SPEC), backend).report
    assert text == render_conclusion(report)
