import json

from app.core.report import Kind, Measurement, Report, Status, Verdict


def verdict(**overrides):
    base = {
        "id": "eyes_open",
        "title": "Глаза открыты",
        "status": Status.PASS,
        "kind": Kind.HARD,
        "measured": {"min_ear": 0.27},
        "threshold": None,
        "reason": None,
        "fix_hint": None,
        "blocked_by": None,
    }
    base.update(overrides)
    return Verdict(**base)


def test_pass_verdict_serializes_only_measured():
    data = verdict().to_dict()
    assert data == {
        "id": "eyes_open",
        "title": "Глаза открыты",
        "status": "pass",
        "kind": "hard",
        "measured": {"min_ear": 0.27},
    }


def test_fail_verdict_serializes_evidence():
    data = verdict(
        status=Status.FAIL,
        measured={"yaw": 17.34567, "pitch": 2.1},
        threshold={"yaw": 10.0, "pitch": 10.0},
        reason="поворот головы yaw=17.3°",
        fix_hint="смотрите прямо в камеру",
    ).to_dict()
    assert data["status"] == "fail"
    assert data["measured"] == {"yaw": 17.3457, "pitch": 2.1}
    assert data["threshold"] == {"yaw": 10.0, "pitch": 10.0}
    assert data["reason"] == "поворот головы yaw=17.3°"
    assert data["fix_hint"] == "смотрите прямо в камеру"
    assert "blocked_by" not in data


def test_undefined_verdict_serializes_blocked_by():
    data = verdict(status=Status.UNDEFINED, measured=None, blocked_by="single_face").to_dict()
    assert data == {
        "id": "eyes_open",
        "title": "Глаза открыты",
        "status": "undefined",
        "kind": "hard",
        "blocked_by": "single_face",
    }


def test_integers_survive_rounding():
    data = verdict(measured={"count": 2}).to_dict()
    assert data["measured"] == {"count": 2}
    assert isinstance(data["measured"]["count"], int)


def test_report_round_trip():
    report = Report(
        image="img.jpg",
        spec="document_photo",
        accepted=False,
        verdicts=(verdict(), verdict(id="single_face", status=Status.FAIL, measured={"count": 2})),
        measurements={"eyes.min_ear": {"min_ear": 0.27, "left_ear": 0.3}},
    )
    restored = Report.from_dict(json.loads(report.to_json()))
    assert restored == report


def test_report_load_without_measurements(tmp_path):
    payload = {"image": "a.jpg", "spec": "s", "accepted": True, "verdicts": [verdict().to_dict()]}
    path = tmp_path / "отчёт.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    report = Report.load(path)
    assert report.measurements == {}
    assert report.by_id("eyes_open").status is Status.PASS


def test_save_keeps_cyrillic_readable(tmp_path):
    report = Report("a.jpg", "s", True, (verdict(),), {})
    path = tmp_path / "out.json"
    report.save(path)
    assert "Глаза открыты" in path.read_text(encoding="utf-8")


def test_filters():
    report = Report(
        "a.jpg",
        "s",
        False,
        (
            verdict(id="h", status=Status.FAIL, kind=Kind.HARD),
            verdict(id="s", status=Status.FAIL, kind=Kind.SOFT),
            verdict(id="u", status=Status.UNDEFINED, measured=None, blocked_by="h"),
            verdict(id="p"),
        ),
        {},
    )
    assert [v.id for v in report.failing(Kind.HARD)] == ["h"]
    assert [v.id for v in report.failing(Kind.SOFT)] == ["s"]
    assert [v.id for v in report.undefined()] == ["u"]
    assert [v.id for v in report.with_status(Status.PASS)] == ["p"]


def test_measurement_constructors():
    assert Measurement.of("face.count", {"count": 1}).applicable
    missing = Measurement.not_applicable("pose.yaw_pitch", "no face detected")
    assert not missing.applicable
    assert missing.values == {}
    assert missing.note == "no face detected"
