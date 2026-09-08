import json
import logging

import numpy as np
import pytest

from app import cli
from app.analyzers.context import FaceBackend
from app.imageio import write_bgr
from tests.helpers import FakeBackend, frontal_landmarks

SPEC = "app/specs/document_photo.yaml"


def fake_factory(landmarks, mask):
    def factory(models_dir) -> FaceBackend:
        return FakeBackend(landmarks, mask)

    return factory


def portrait_like(tmp_path, size=512):
    rng = np.random.default_rng(3)
    image = np.full((size, size, 3), 200, np.uint8)
    face = rng.integers(40, 220, size=(200, 160, 3), dtype=np.uint8)
    image[160:360, 176:336] = face
    path = tmp_path / "фото.jpg"
    write_bgr(path, image, quality=95)
    mask = np.zeros((size, size), np.float32)
    mask[120:512, 150:362] = 1.0
    return path, [frontal_landmarks((256, 260), 160, size, size)], mask


def test_parser_subcommands():
    args = cli.build_parser().parse_args(["check", "img.jpg", "--spec", SPEC, "--json", "o.json"])
    assert args.command == "check"
    assert str(args.json) == "o.json"
    args = cli.build_parser().parse_args(["models", "download", "--force"])
    assert (args.command, args.models_command, args.force) == ("models", "download", True)


def test_accepted_image_returns_zero_and_writes_json(tmp_path, capsys):
    path, landmarks, mask = portrait_like(tmp_path)
    out = tmp_path / "out.json"
    code = cli.main(
        ["check", str(path), "--spec", SPEC, "--json", str(out)],
        backend_factory=fake_factory(landmarks, mask),
    )
    assert code == cli.EXIT_ACCEPTED
    captured = capsys.readouterr().out
    assert "ИТОГ: ПРИНЯТО" in captured
    assert "single_face" in captured
    assert f"JSON: {out}" in captured
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["accepted"] is True
    assert report["spec"] == "document_photo"


def test_rejected_image_returns_one(tmp_path, capsys):
    path, _, mask = portrait_like(tmp_path)
    code = cli.main(["check", str(path), "--spec", SPEC], backend_factory=fake_factory([], mask))
    assert code == cli.EXIT_REJECTED
    captured = capsys.readouterr().out
    assert "ИТОГ: ОТКЛОНЕНО" in captured
    assert "[single_face]" in captured
    assert "заблокировано: single_face" in captured


def test_missing_image_returns_runtime_error(tmp_path, caplog):
    code = cli.main(
        ["check", str(tmp_path / "nope.jpg"), "--spec", SPEC],
        backend_factory=fake_factory([], np.zeros((4, 4))),
    )
    assert code == cli.EXIT_RUNTIME
    assert "cannot read image" in caplog.text


def test_bad_spec_returns_usage_error(tmp_path, caplog):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n", encoding="utf-8")
    path, landmarks, mask = portrait_like(tmp_path)
    code = cli.main(
        ["check", str(path), "--spec", str(bad)], backend_factory=fake_factory(landmarks, mask)
    )
    assert code == cli.EXIT_USAGE
    assert "cannot load specification" in caplog.text


def test_models_download_command(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_download(models_dir, force):
        calls.append((models_dir, force))
        return [models_dir / "a.task"]

    monkeypatch.setattr(cli, "download_models", fake_download)
    code = cli.main(["models", "download", "--models-dir", str(tmp_path), "--force"])
    assert code == cli.EXIT_ACCEPTED
    assert calls == [(tmp_path, True)]
    assert "a.task" in capsys.readouterr().out


@pytest.mark.mediapipe
def test_cli_on_fixture_portrait(fixtures_dir, tmp_path, capsys):
    out = tmp_path / "report.json"
    code = cli.main(
        ["check", str(fixtures_dir / "portrait_second.jpg"), "--spec", SPEC, "--json", str(out)]
    )
    captured = capsys.readouterr().out
    assert code in (cli.EXIT_ACCEPTED, cli.EXIT_REJECTED)
    assert "ИТОГ:" in captured
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["verdicts"][0]["id"] == "single_face"
    assert report["verdicts"][0]["status"] == "pass"


def test_llm_without_key_warns_and_uses_mock(tmp_path, capsys, caplog, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    caplog.set_level(logging.INFO)
    path, landmarks, mask = portrait_like(tmp_path)
    code = cli.main(
        ["check", str(path), "--spec", SPEC, "--llm", "--llm-cache", str(tmp_path / "cache")],
        backend_factory=fake_factory(landmarks, mask),
    )
    assert code == cli.EXIT_ACCEPTED
    assert "ANTHROPIC_API_KEY" in caplog.text
    assert "conclusion source: mock" in caplog.text
    assert "ИТОГ: ПРИНЯТО" in capsys.readouterr().out


def test_mock_requires_llm(tmp_path):
    path, landmarks, mask = portrait_like(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["check", str(path), "--spec", SPEC, "--mock"], fake_factory(landmarks, mask))
    assert exc.value.code == cli.EXIT_USAGE
