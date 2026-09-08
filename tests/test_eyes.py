import numpy as np
import pytest

from app.analyzers.context import AnalysisContext
from app.analyzers.eyes import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, measure_min_ear
from tests.helpers import FakeBackend

OPEN_EYE = np.array([[0, 0], [2, -1.5], [4, -1.5], [6, 0], [4, 1.5], [2, 1.5]], dtype=float)


def test_open_eye_ratio():
    assert eye_aspect_ratio(OPEN_EYE) == pytest.approx(0.5)


def test_vertical_compression_halves_ratio():
    squeezed = OPEN_EYE * np.array([1.0, 0.5])
    assert eye_aspect_ratio(squeezed) == pytest.approx(0.25)


def test_closed_eye_is_zero():
    closed = OPEN_EYE * np.array([1.0, 0.0])
    assert eye_aspect_ratio(closed) == 0.0


def test_degenerate_corners_give_zero():
    assert eye_aspect_ratio(np.zeros((6, 2))) == 0.0


def test_measurer_takes_minimum_over_both_eyes():
    landmarks = np.zeros((478, 3))
    width, height = 100, 100
    landmarks[list(LEFT_EYE), :2] = (OPEN_EYE + 10) / width
    landmarks[list(RIGHT_EYE), :2] = (OPEN_EYE * np.array([1.0, 0.5]) + 50) / width
    ctx = AnalysisContext(
        np.zeros((height, width, 3), np.uint8),
        FakeBackend([landmarks], np.zeros((height, width), np.float32)),
    )
    measurement = measure_min_ear(ctx)
    assert measurement.applicable
    assert measurement.values["left_ear"] == pytest.approx(0.5)
    assert measurement.values["right_ear"] == pytest.approx(0.25)
    assert measurement.values["min_ear"] == pytest.approx(0.25)


def test_measurer_without_face():
    ctx = AnalysisContext(np.zeros((10, 10, 3), np.uint8), FakeBackend([], np.zeros((10, 10))))
    measurement = measure_min_ear(ctx)
    assert not measurement.applicable
    assert measurement.note == "лицо не обнаружено"
