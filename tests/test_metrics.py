import numpy as np
import pandas as pd
import pytest

from app.analyzers import REGISTRY
from app.core.spec import parse_spec
from experiments.claims import ClaimRow, score_claims
from experiments.evaluate import measure_column, requirement_ids, split_ids, status_column
from experiments.metrics import (
    all_requirement_metrics,
    cross_trigger,
    error_rates,
    requirement_metrics,
    timing,
)
from tests.helpers import requirement, spec_data

SPEC = parse_spec(
    spec_data(
        requirement("face_sharpness", "sharpness.face_laplacian_var", {"op": "ge", "value": 60}),
        requirement(
            "exposure_ok",
            "exposure.stats",
            {"op": "range", "key": "mean_brightness", "value": [70, 200]},
        ),
    ),
    REGISTRY,
)


def frame():
    rows = [
        ("clean", "", "pass", 200.0, True, 10.0),
        ("clean", "", "pass", 150.0, True, 12.0),
        ("clean", "", "fail", 40.0, False, 14.0),
        ("clean", "", "undefined", np.nan, False, 16.0),
        ("blur_face", "face_sharpness", "fail", 20.0, False, 20.0),
        ("blur_face", "face_sharpness", "fail", 30.0, False, 22.0),
        ("blur_face", "face_sharpness", "pass", 90.0, True, 24.0),
        ("dark", "exposure_ok", "pass", 100.0, False, 26.0),
    ]
    table = pd.DataFrame(
        rows,
        columns=[
            "cls",
            "expected_fail_id",
            "status.face_sharpness",
            "m.sharpness.face_laplacian_var.var",
            "accepted",
            "elapsed_ms",
        ],
    )
    table["status.exposure_ok"] = ["pass"] * 7 + ["fail"]
    table["m.exposure.stats.mean_brightness"] = [120.0] * 7 + [30.0]
    table["file"] = [f"{i}.jpg" for i in range(len(table))]
    return table


def test_requirement_metrics_exact_values():
    metrics = requirement_metrics(frame(), SPEC, "face_sharpness", "blur_face")
    assert metrics.n_pos == 3
    assert metrics.n_neg == 4
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.undefined_rate == pytest.approx(1 / 7)
    assert metrics.auc == pytest.approx(8 / 9)


def test_all_requirement_metrics_cover_present_classes():
    items = all_requirement_metrics(frame(), SPEC)
    assert [(item.cls, item.requirement) for item in items] == [
        ("blur_face", "face_sharpness"),
        ("dark", "exposure_ok"),
    ]
    dark = items[1]
    assert dark.recall == 1.0
    assert np.isnan(dark.auc) or dark.auc == 1.0


def test_error_rates_and_timing():
    rates = error_rates(frame())
    assert rates.far == pytest.approx(1 / 4)
    assert rates.frr == pytest.approx(2 / 4)
    assert rates.per_class_far == {"blur_face": pytest.approx(1 / 3), "dark": 0.0}
    assert timing(frame()) == {"median_ms": 18.0, "mean_ms": 18.0}


def test_cross_trigger_table():
    table = cross_trigger(frame(), requirement_ids(frame().columns))
    assert table.loc["blur_face", "face_sharpness"] == pytest.approx(2 / 3)
    assert table.loc["dark", "exposure_ok"] == 1.0
    assert table.loc["clean", "exposure_ok"] == 0.0


def test_column_helpers():
    assert status_column("a") == "status.a"
    assert measure_column("pose.yaw_pitch", "yaw") == "m.pose.yaw_pitch.yaw"
    assert split_ids("a;b") == {"a", "b"}
    assert split_ids("") == set()


def test_score_claims():
    rows = [
        ClaimRow("1", frozenset({"a"}), frozenset({"a"}), frozenset({"a"})),
        ClaimRow("2", frozenset({"a", "b"}), frozenset({"a"}), frozenset({"a"})),
        ClaimRow("3", frozenset(), frozenset({"c"}), frozenset()),
    ]
    summary = score_claims(rows)
    assert summary.n == 3
    assert summary.consistency == pytest.approx(2 / 3)
    assert summary.completeness == pytest.approx(2 / 3)
    assert summary.invented_rate == pytest.approx(1 / 3)
    assert (summary.mentioned, summary.missed, summary.invented) == (2, 1, 1)
    assert score_claims([]).n == 0


def test_metrics_without_measurement_columns():
    table = frame().drop(columns=[c for c in frame().columns if c.startswith("m.")])
    metrics = requirement_metrics(table, SPEC, "face_sharpness", "blur_face")
    assert np.isnan(metrics.auc)
    assert metrics.f1 == pytest.approx(2 / 3)
