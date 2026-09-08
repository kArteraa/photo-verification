from app.core.engine import Engine
from app.core.report import Kind, Status
from app.core.spec import parse_spec
from tests.helpers import CallCounter, constant_measurer, registry_of, requirement, spec_data

POSE_TEMPLATE = "поворот головы yaw={yaw:.1f}°, pitch={pitch:.1f}° при допуске ±{tau:g}°"
POSE_HINT = "поверните голову {yaw_dir} примерно на {yaw_abs:.0f}°"


def engine_for(registry, *requirements):
    return Engine(parse_spec(spec_data(*requirements, name="doc"), registry))


def single_face(count=1):
    return constant_measurer("face.count", ("count",), {"count": count})


def pose(yaw=0.0, pitch=0.0, counter=None):
    values = {"yaw": yaw, "pitch": pitch, "roll": 0.0}
    return constant_measurer("pose.yaw_pitch", ("yaw", "pitch", "roll"), values, counter=counter)


SINGLE_FACE = requirement(
    "single_face",
    "face.count",
    {"op": "eq", "value": 1},
    reason_template="обнаружено лиц: {value} (требуется ровно 1)",
    fix_hint="оставьте в кадре только своё лицо",
)
FRONTAL = requirement(
    "frontal_pose",
    "pose.yaw_pitch",
    {"op": "abs_le", "keys": ["yaw", "pitch"], "value": [10.0, 10.0]},
    depends_on=["single_face"],
    reason_template=POSE_TEMPLATE,
    fix_hint=POSE_HINT,
)
HEADROOM = requirement(
    "headroom",
    "geometry.headroom_ratio",
    {"op": "range", "value": [0.04, 0.25]},
    kind="soft",
    depends_on=["single_face"],
    reason_template="отступ {value:.2f} {side} допуска [{lo}, {hi}] на {delta:.2f}",
)


def test_all_pass_is_accepted_and_records_measurements():
    registry = registry_of(single_face(), pose(3.0, -2.0))
    report = engine_for(registry, SINGLE_FACE, FRONTAL).evaluate(None, "img.jpg")
    assert report.accepted
    assert report.image == "img.jpg"
    assert report.spec == "doc"
    assert [v.status for v in report.verdicts] == [Status.PASS, Status.PASS]
    assert report.by_id("frontal_pose").measured == {"yaw": 3.0, "pitch": -2.0}
    assert report.measurements["pose.yaw_pitch"] == {"yaw": 3.0, "pitch": -2.0, "roll": 0.0}


def test_failed_dependency_blocks_and_skips_measurer():
    counter = CallCounter()
    registry = registry_of(single_face(count=2), pose(counter=counter))
    report = engine_for(registry, SINGLE_FACE, FRONTAL).evaluate(None, "img.jpg")
    assert not report.accepted
    face = report.by_id("single_face")
    assert face.status is Status.FAIL
    assert face.reason == "обнаружено лиц: 2 (требуется ровно 1)"
    assert face.threshold == {"count": 1}
    blocked = report.by_id("frontal_pose")
    assert blocked.status is Status.UNDEFINED
    assert blocked.blocked_by == "single_face"
    assert blocked.measured is None
    assert counter.calls == 0
    assert "pose.yaw_pitch" not in report.measurements


def test_not_applicable_measurer_yields_undefined_with_note():
    registry = registry_of(
        single_face(),
        constant_measurer(
            "pose.yaw_pitch", ("yaw", "pitch", "roll"), None, note="no face detected"
        ),
    )
    report = engine_for(registry, SINGLE_FACE, FRONTAL).evaluate(None, "img.jpg")
    verdict = report.by_id("frontal_pose")
    assert verdict.status is Status.UNDEFINED
    assert verdict.reason == "no face detected"
    assert verdict.blocked_by is None
    assert not report.accepted


def test_hard_fail_rejects_soft_fail_does_not():
    registry = registry_of(
        single_face(),
        pose(17.3, 2.1),
        constant_measurer("geometry.headroom_ratio", ("headroom_ratio",), {"headroom_ratio": 0.01}),
    )
    report = engine_for(registry, SINGLE_FACE, HEADROOM).evaluate(None, "img.jpg")
    assert report.accepted
    soft = report.by_id("headroom")
    assert soft.status is Status.FAIL
    assert soft.kind is Kind.SOFT
    assert soft.reason == "отступ 0.01 ниже допуска [0.04, 0.25] на 0.03"
    report = engine_for(registry, SINGLE_FACE, FRONTAL, HEADROOM).evaluate(None, "img.jpg")
    assert not report.accepted


def test_fail_reason_and_hint_formatting():
    registry = registry_of(single_face(), pose(17.34, 2.1))
    report = engine_for(registry, SINGLE_FACE, FRONTAL).evaluate(None, "img.jpg")
    verdict = report.by_id("frontal_pose")
    assert verdict.status is Status.FAIL
    assert verdict.reason == "поворот головы yaw=17.3°, pitch=2.1° при допуске ±10°"
    assert verdict.fix_hint == "поверните голову влево примерно на 17°"
    assert verdict.threshold == {"yaw": 10.0, "pitch": 10.0}
    assert verdict.measured == {"yaw": 17.34, "pitch": 2.1}


def test_shared_measurer_runs_once():
    counter = CallCounter()
    registry = registry_of(single_face(), pose(counter=counter))
    yaw_only = requirement(
        "yaw_only", "pose.yaw_pitch", {"op": "abs_le", "key": "yaw", "value": 5.0}
    )
    pitch_only = requirement(
        "pitch_only", "pose.yaw_pitch", {"op": "abs_le", "key": "pitch", "value": 5.0}
    )
    engine_for(registry, yaw_only, pitch_only).evaluate(None, "img.jpg")
    assert counter.calls == 1


def test_dependency_declared_after_dependant_is_still_evaluated_first():
    registry = registry_of(single_face(count=0), pose())
    report = engine_for(registry, FRONTAL, SINGLE_FACE).evaluate(None, "img.jpg")
    assert [v.id for v in report.verdicts] == ["frontal_pose", "single_face"]
    assert report.by_id("frontal_pose").blocked_by == "single_face"


def test_measure_all_and_evaluate_measured_match_lazy_evaluation():
    counter = CallCounter()
    registry = registry_of(single_face(count=2), pose(counter=counter))
    engine = engine_for(registry, SINGLE_FACE, FRONTAL)
    measurements = engine.measure_all(None)
    assert set(measurements) == {"face.count", "pose.yaw_pitch"}
    assert counter.calls == 1
    report = engine.evaluate_measured("img.jpg", measurements)
    assert report.verdicts == engine.evaluate(None, "img.jpg").verdicts
    assert [m.name for m in engine.measurers] == ["face.count", "pose.yaw_pitch"]
