import pytest
import yaml

from app.core.errors import SpecError, UnknownMeasurerError
from app.core.report import Kind
from app.core.spec import load_spec, parse_spec, topological_order
from tests.helpers import constant_measurer, registry_of, requirement, spec_data

REGISTRY = registry_of(
    constant_measurer("face.count", ("count",), {"count": 1}),
    constant_measurer(
        "pose.yaw_pitch", ("yaw", "pitch", "roll"), {"yaw": 0, "pitch": 0, "roll": 0}
    ),
    constant_measurer("eyes.min_ear", ("min_ear", "left_ear", "right_ear"), {"min_ear": 0.3}),
    constant_measurer("exposure.stats", ("mean_brightness", "clip_low"), {"mean_brightness": 120}),
)


def parse(*requirements):
    return parse_spec(spec_data(*requirements), REGISTRY)


def test_single_key_predicate_uses_primary_key():
    spec = parse(requirement("eyes_open", "eyes.min_ear", {"op": "ge", "value": 0.18}))
    predicate = spec.by_id("eyes_open").predicate
    assert predicate.keys == ("min_ear",)
    assert predicate.values == (0.18,)


def test_explicit_key_and_range():
    spec = parse(
        requirement(
            "exposure_ok",
            "exposure.stats",
            {"op": "range", "key": "mean_brightness", "value": [70, 200]},
            reason_template="{value} {lo} {hi} {side} {delta}",
        )
    )
    predicate = spec.by_id("exposure_ok").predicate
    assert predicate.op == "range"
    assert predicate.keys == ("mean_brightness",)
    assert predicate.values == (70, 200)


def test_abs_le_with_keys_and_hint_placeholders():
    spec = parse(
        requirement(
            "frontal_pose",
            "pose.yaw_pitch",
            {"op": "abs_le", "keys": ["yaw", "pitch"], "value": [10.0, 10.0]},
            reason_template="yaw={yaw:.1f} pitch={pitch:.1f} tau={tau} {tau_yaw} {tau_pitch}",
            fix_hint="поверните голову {yaw_dir} на {yaw_abs:.0f}°",
        )
    )
    predicate = spec.by_id("frontal_pose").predicate
    assert predicate.keys == ("yaw", "pitch")
    assert spec.by_id("frontal_pose").kind is Kind.HARD


def test_topological_order_keeps_yaml_order_for_ties():
    spec = parse(
        requirement("c", "face.count", {"op": "eq", "value": 1}, depends_on=["b"]),
        requirement("b", "face.count", {"op": "eq", "value": 1}, depends_on=["a"]),
        requirement("a", "face.count", {"op": "eq", "value": 1}),
        requirement("d", "face.count", {"op": "eq", "value": 1}, depends_on=["a"]),
    )
    assert [r.id for r in topological_order(spec)] == ["a", "b", "c", "d"]
    assert spec.ids == ("c", "b", "a", "d")


def test_load_spec_from_yaml(tmp_path):
    data = spec_data(
        requirement(
            "single_face",
            "face.count",
            {"op": "eq", "value": 1},
            reason_template="обнаружено лиц: {value}",
            title="На снимке ровно одно лицо",
        )
    )
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    spec = load_spec(path, REGISTRY)
    assert spec.by_id("single_face").title == "На снимке ровно одно лицо"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"name": "x", "requirements": []}, "missing top-level field: description"),
        (spec_data(), "non-empty list"),
        ("not a mapping", "must be a mapping"),
    ],
)
def test_top_level_errors(data, message):
    with pytest.raises(SpecError, match=message):
        parse_spec(data, REGISTRY)


def test_duplicate_id():
    with pytest.raises(SpecError, match="duplicate requirement id: a"):
        parse(
            requirement("a", "face.count", {"op": "eq", "value": 1}),
            requirement("a", "face.count", {"op": "eq", "value": 1}),
        )


def test_missing_field():
    item = requirement("a", "face.count", {"op": "eq", "value": 1})
    del item["fix_hint"]
    with pytest.raises(SpecError, match="a: missing field fix_hint"):
        parse(item)


def test_unknown_measurer_is_spec_error():
    with pytest.raises(UnknownMeasurerError, match=r"unknown measurer: nope\.x"):
        parse(requirement("a", "nope.x", {"op": "eq", "value": 1}))
    with pytest.raises(SpecError):
        parse(requirement("a", "nope.x", {"op": "eq", "value": 1}))


@pytest.mark.parametrize(
    ("predicate", "message"),
    [
        ({"op": "gt", "value": 1}, "unknown op gt"),
        ({"op": "abs_le", "keys": ["yaw", "pitch"], "value": [10.0]}, "list of 2 numbers"),
        ({"op": "range", "value": [1, 2, 3]}, "list of 2 numbers"),
        ({"op": "range", "value": [5, 5]}, "lower bound must be below"),
        ({"op": "ge", "key": "nope", "value": 1}, "key nope not produced by pose.yaw_pitch"),
        ({"op": "ge", "value": "1"}, "must be a number"),
        ({"op": "ge", "value": True}, "must be a number"),
        ({"value": 1}, "needs op and value"),
    ],
)
def test_predicate_errors(predicate, message):
    with pytest.raises(SpecError, match=message):
        parse(requirement("a", "pose.yaw_pitch", predicate))


def test_invalid_kind():
    with pytest.raises(SpecError, match="kind must be hard or soft"):
        parse(requirement("a", "face.count", {"op": "eq", "value": 1}, kind="strict"))


def test_dependency_errors():
    with pytest.raises(SpecError, match="a: unknown dependency zzz"):
        parse(requirement("a", "face.count", {"op": "eq", "value": 1}, depends_on=["zzz"]))
    with pytest.raises(SpecError, match="a: depends on itself"):
        parse(requirement("a", "face.count", {"op": "eq", "value": 1}, depends_on=["a"]))


def test_cycle_is_reported_with_path():
    with pytest.raises(SpecError, match="dependency cycle: a -> b -> a"):
        parse(
            requirement("a", "face.count", {"op": "eq", "value": 1}, depends_on=["b"]),
            requirement("b", "face.count", {"op": "eq", "value": 1}, depends_on=["a"]),
        )


@pytest.mark.parametrize(
    ("field", "template", "message"),
    [
        ("reason_template", "{nope}", "unknown placeholder {nope}"),
        ("reason_template", "{lo}", "unknown placeholder {lo}"),
        ("fix_hint", "{count_dir}", "unknown placeholder {count_dir}"),
        ("fix_hint", "{value", "malformed fix_hint"),
    ],
)
def test_template_errors(field, template, message):
    item = requirement("a", "face.count", {"op": "eq", "value": 1})
    item[field] = template
    with pytest.raises(SpecError, match=message):
        parse(item)


def test_range_predicate_rejects_tau_placeholder():
    item = requirement(
        "a",
        "exposure.stats",
        {"op": "range", "value": [70, 200]},
        reason_template="{value} при допуске {tau}",
    )
    with pytest.raises(SpecError, match=r"unknown placeholder {tau}"):
        parse(item)
