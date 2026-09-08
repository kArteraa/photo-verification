import numpy as np
import pytest

from app.analyzers.context import AnalysisContext, BBox
from app.analyzers.illumination import NOSE_BRIDGE, face_asymmetry, measure_face_asymmetry
from tests.helpers import FakeBackend, box_landmarks

SIZE = 100


def landmarks_with_bridge(x):
    landmarks = np.zeros((478, 2))
    landmarks[NOSE_BRIDGE] = (x, 50)
    return landmarks


def test_symmetric_face_is_zero():
    gray = np.full((SIZE, SIZE), 100, np.uint8)
    assert face_asymmetry(gray, landmarks_with_bridge(50), BBox(20, 20, 80, 80)) == 0.0


def test_left_lit_face_is_positive():
    gray = np.zeros((SIZE, SIZE), np.uint8)
    gray[:, :50] = 200
    gray[:, 50:] = 50
    value = face_asymmetry(gray, landmarks_with_bridge(50), BBox(20, 20, 80, 80))
    assert value == pytest.approx((200 - 50) / 250)
    mirrored = face_asymmetry(gray[:, ::-1], landmarks_with_bridge(50), BBox(20, 20, 80, 80))
    assert mirrored == pytest.approx(-value)


def test_black_face_and_empty_box():
    gray = np.zeros((SIZE, SIZE), np.uint8)
    assert face_asymmetry(gray, landmarks_with_bridge(50), BBox(20, 20, 80, 80)) == 0.0
    assert face_asymmetry(gray, landmarks_with_bridge(50), BBox(50, 50, 50, 50)) == 0.0


def test_measurer():
    gray = np.zeros((SIZE, SIZE), np.uint8)
    gray[:, :50] = 200
    bgr = np.repeat(gray[:, :, None], 3, axis=2)
    landmarks = box_landmarks(20, 20, 80, 80, SIZE, SIZE)
    landmarks[NOSE_BRIDGE, :2] = (0.5, 0.5)
    ctx = AnalysisContext(bgr, FakeBackend([landmarks], np.zeros((SIZE, SIZE), np.float32)))
    assert measure_face_asymmetry(ctx).values["asymmetry"] > 0.5
    empty = AnalysisContext(bgr, FakeBackend([], np.zeros((SIZE, SIZE), np.float32)))
    assert not measure_face_asymmetry(empty).applicable
