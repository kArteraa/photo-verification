from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from app.analyzers import REGISTRY
from app.core.spec import load_spec
from experiments.calibrate import (
    calibrate_requirement,
    choose_threshold,
    clean_percentiles,
    score_sign,
    write_thresholds,
)

SPEC_PATH = Path(__file__).resolve().parent.parent / "app" / "specs" / "document_photo.yaml"


def test_choose_threshold_separable_by_f1():
    choice = choose_threshold(np.array([5.0, 6.0, 7.0]), np.array([1.0, 2.0, 3.0]), "f1", 0.05)
    assert 3.0 < choice.score < 5.0
    assert choice.f1 == 1.0
    assert choice.frr == 0.0


def test_choose_threshold_frr_mode_respects_bound():
    positives = np.array([2.5, 4.0, 5.0, 6.0])
    negatives = np.array([1.0, 2.0, 3.0, 3.5])
    choice = choose_threshold(positives, negatives, "frr", 0.0)
    assert choice.score > 3.5
    assert choice.frr == 0.0
    assert choice.tpr == pytest.approx(3 / 4)
    relaxed = choose_threshold(positives, negatives, "frr", 0.5)
    assert relaxed.tpr == 1.0
    assert relaxed.frr <= 0.5


def test_score_sign():
    assert score_sign("ge", None) == -1.0
    assert score_sign("le", None) == 1.0
    assert score_sign("range", "lo") == -1.0
    assert score_sign("range", "hi") == 1.0


def frame():
    sharp = [200.0, 180.0, 150.0, 120.0]
    blurred = [10.0, 20.0, 30.0, 55.0]
    bright_clean = [110.0, 130.0, 150.0, 170.0]
    bright = [215.0, 230.0, 240.0, 250.0]
    rows = []
    for value, mean in zip(sharp, bright_clean, strict=True):
        rows.append(("clean", value, mean))
    for value in blurred:
        rows.append(("blur_face", value, 120.0))
    for mean in bright:
        rows.append(("bright", 150.0, mean))
    return pd.DataFrame(
        rows,
        columns=["cls", "m.sharpness.face_laplacian_var.var", "m.exposure.stats.mean_brightness"],
    )


def test_calibrate_requirement_ge_and_range_high():
    spec = load_spec(SPEC_PATH, REGISTRY)
    blur = calibrate_requirement(frame(), spec, "blur_face", "f1", 0.05)
    assert blur["requirement"] == "face_sharpness"
    assert 55.0 < blur["new_value"] < 120.0
    assert blur["f1"] == 1.0
    assert blur["roc"]["fpr"][-1] == 1.0
    bright = calibrate_requirement(frame(), spec, "bright", "f1", 0.05)
    assert bright["side"] == "hi"
    assert bright["old_value"] == 200
    assert 170.0 < bright["new_value"] < 215.0


def test_write_thresholds_changes_only_predicate_lines(tmp_path):
    updates = [
        {"requirement": "face_sharpness", "op": "ge", "side": None, "new_value": 87.5},
        {"requirement": "exposure_ok", "op": "range", "side": "hi", "new_value": 205.0},
        {"requirement": "face_size", "op": "range", "side": "lo", "new_value": 0.08},
    ]
    text = write_thresholds(SPEC_PATH, updates)
    original = SPEC_PATH.read_text(encoding="utf-8")
    changed = [
        (a, b) for a, b in zip(original.splitlines(), text.splitlines(), strict=True) if a != b
    ]
    assert len(changed) == 3
    assert all("predicate:" in b for _, b in changed)
    out = tmp_path / "spec.yaml"
    out.write_text(text, encoding="utf-8")
    spec = load_spec(out, REGISTRY)
    assert spec.by_id("face_sharpness").predicate.values == (87.5,)
    assert spec.by_id("exposure_ok").predicate.values == (70, 205.0)
    assert spec.by_id("face_size").predicate.values == (0.08, 0.5)
    data = yaml.safe_load(text)
    assert data["requirements"][0]["title"] == "На снимке ровно одно лицо"


def test_clean_percentiles():
    spec = load_spec(SPEC_PATH, REGISTRY)
    table = clean_percentiles(frame(), spec)
    assert table["face_sharpness.var"]["p95"] == pytest.approx(197.0)
    assert "exposure_ok.mean_brightness" in table
