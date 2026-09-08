import json

from experiments.plots import f1_frame, load_real_metrics, summary_table


def ours():
    return {
        "per_requirement": [
            {
                "requirement": "face_sharpness",
                "cls": "blur_face",
                "precision": 0.9,
                "recall": 1.0,
                "f1": 0.95,
                "auc": 1.0,
                "undefined_rate": 0.0,
                "n_pos": 10,
                "n_neg": 10,
            }
        ],
        "far": 0.1,
        "frr": 0.2,
        "timing": {"median_ms": 21.0},
        "claims": {"consistency": 1.0, "completeness": 0.8, "invented_rate": 0.3},
    }


def vlm():
    return {
        "n_images": 100,
        "is_mock": False,
        "per_requirement": [
            {
                "requirement": "face_sharpness",
                "cls": "blur_face",
                "precision": 1.0,
                "recall": 0.5,
                "f1": 0.67,
            }
        ],
        "far": 0.15,
        "frr": 0.25,
        "claims": {"consistency": 0.9, "completeness": 0.7, "invented_rate": 0.2},
    }


def test_summary_table_without_baselines():
    table, integral = summary_table(ours(), None)
    assert list(table.columns) == [
        "requirement",
        "cls",
        "precision",
        "recall",
        "f1",
        "auc",
        "undefined_rate",
        "n_pos",
        "n_neg",
    ]
    assert list(integral.columns) == ["far", "frr", "median_ms", "consistency", "completeness"]
    assert "vlm" not in f1_frame(ours(), None)


def test_summary_table_with_real_baselines():
    llm = {
        "n": 100,
        "first_try_rate": 0.9,
        "retry_rate": 0.08,
        "fallback_rate": 0.02,
        "verified_rate": 1.0,
    }
    table, integral = summary_table(ours(), vlm(), llm)
    assert table.loc[0, "vlm_f1"] == 0.67
    assert integral.loc[0, "vlm_far"] == 0.15
    assert integral.loc[0, "vlm_n"] == 100
    assert integral.loc[0, "llm_first_try_rate"] == 0.9
    assert integral.loc[0, "llm_n"] == 100
    assert "vlm" in f1_frame(ours(), vlm())


def test_mock_metrics_are_ignored(tmp_path, caplog):
    path = tmp_path / "vlm_metrics.json"
    path.write_text(json.dumps({**vlm(), "is_mock": True}), encoding="utf-8")
    assert load_real_metrics(path) is None
    assert "mock client" in caplog.text
    path.write_text(json.dumps(vlm()), encoding="utf-8")
    assert load_real_metrics(path)["n_images"] == 100
    assert load_real_metrics(tmp_path / "missing.json") is None
