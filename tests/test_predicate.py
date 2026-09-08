import pytest

from app.core.predicate import evaluate, format_context, threshold_dict, violation_score
from app.core.spec import Predicate

EQ = Predicate("eq", ("count",), (1,))
GE = Predicate("ge", ("min_ear",), (0.18,))
LE = Predicate("le", ("cv",), (0.25,))
RANGE = Predicate("range", ("mean_brightness",), (70, 200))
ABS = Predicate("abs_le", ("yaw", "pitch"), (10.0, 10.0))


@pytest.mark.parametrize(
    ("predicate", "values", "expected"),
    [
        (EQ, {"count": 1}, True),
        (EQ, {"count": 2}, False),
        (EQ, {"count": 1 + 1e-12}, True),
        (GE, {"min_ear": 0.18}, True),
        (GE, {"min_ear": 0.1799}, False),
        (LE, {"cv": 0.25}, True),
        (LE, {"cv": 0.2501}, False),
        (RANGE, {"mean_brightness": 70}, True),
        (RANGE, {"mean_brightness": 200}, True),
        (RANGE, {"mean_brightness": 69.9}, False),
        (RANGE, {"mean_brightness": 200.1}, False),
        (ABS, {"yaw": -10.0, "pitch": 9.9}, True),
        (ABS, {"yaw": 10.1, "pitch": 0.0}, False),
        (ABS, {"yaw": 0.0, "pitch": -10.5}, False),
    ],
)
def test_evaluate(predicate, values, expected):
    assert evaluate(predicate, values) is expected


def test_threshold_dict_shapes():
    assert threshold_dict(EQ) == {"count": 1}
    assert threshold_dict(RANGE) == {"mean_brightness": [70, 200]}
    assert threshold_dict(ABS) == {"yaw": 10.0, "pitch": 10.0}


def test_format_context_for_abs_le():
    context = format_context(ABS, {"yaw": 17.3, "pitch": 2.1, "roll": -1.0})
    assert context["value"] == 17.3
    assert context["tau"] == 10.0
    assert context["tau_yaw"] == 10.0
    assert context["tau_pitch"] == 10.0
    assert context["roll"] == -1.0
    assert "lo" not in context


def test_format_context_for_range_below_and_above():
    below = format_context(RANGE, {"mean_brightness": 40})
    assert (below["lo"], below["hi"], below["side"], below["delta"]) == (70, 200, "ниже", 30)
    above = format_context(RANGE, {"mean_brightness": 230})
    assert (above["side"], above["delta"]) == ("выше", 30)
    assert "tau" not in above


@pytest.mark.parametrize(
    ("predicate", "passing", "violating"),
    [
        (EQ, {"count": 1}, {"count": 3}),
        (GE, {"min_ear": 0.3}, {"min_ear": 0.05}),
        (LE, {"cv": 0.1}, {"cv": 0.9}),
        (RANGE, {"mean_brightness": 120}, {"mean_brightness": 20}),
        (RANGE, {"mean_brightness": 120}, {"mean_brightness": 250}),
        (ABS, {"yaw": 1.0, "pitch": 1.0}, {"yaw": -25.0, "pitch": 1.0}),
    ],
)
def test_violation_score_grows_with_violation(predicate, passing, violating):
    assert violation_score(predicate, violating) > violation_score(predicate, passing)
