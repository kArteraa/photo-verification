import numpy as np
import pytest

from app.analyzers.context import AnalysisContext, BBox
from app.analyzers.geometry import (
    LEFT_IRIS,
    RIGHT_IRIS,
    center_offset,
    eye_line_ratio,
    face_area_ratio,
    headroom_ratio,
    measure_center_offset,
    measure_eye_line,
    measure_face_area_ratio,
    measure_headroom,
)
from tests.helpers import FakeBackend, box_landmarks

WIDTH, HEIGHT = 200, 100


def test_face_area_ratio():
    assert face_area_ratio(BBox(0, 0, 100, 50), WIDTH, HEIGHT) == pytest.approx(0.25)


def test_center_offset_signs():
    assert center_offset(BBox(90, 40, 110, 60), WIDTH, HEIGHT) == (0.0, 0.0, 0.0)
    dx, dy, offset = center_offset(BBox(130, 10, 170, 30), WIDTH, HEIGHT)
    assert dx == pytest.approx(0.25)
    assert dy == pytest.approx(-0.30)
    assert offset == pytest.approx(np.hypot(0.25, 0.30))


def test_headroom_uses_band_around_face():
    mask = np.zeros((HEIGHT, WIDTH), np.float32)
    mask[25:90, 80:120] = 1.0
    mask[0:90, 0:10] = 1.0
    assert headroom_ratio(mask, BBox(85, 40, 115, 70)) == pytest.approx(0.25)
    assert headroom_ratio(np.zeros((HEIGHT, WIDTH), np.float32), BBox(85, 40, 115, 70)) is None


def test_eye_line_ratio():
    landmarks = np.zeros((478, 2))
    landmarks[LEFT_IRIS] = (80, 30)
    landmarks[RIGHT_IRIS] = (120, 34)
    assert eye_line_ratio(landmarks, HEIGHT) == pytest.approx(0.32)


def test_measurers_with_and_without_face():
    mask = np.zeros((HEIGHT, WIDTH), np.float32)
    mask[20:100, 60:140] = 1.0
    landmarks = box_landmarks(80, 30, 120, 70, WIDTH, HEIGHT)
    ctx = AnalysisContext(np.zeros((HEIGHT, WIDTH, 3), np.uint8), FakeBackend([landmarks], mask))
    assert measure_face_area_ratio(ctx).values["face_area_ratio"] == pytest.approx(0.08)
    assert measure_center_offset(ctx).values["offset"] == pytest.approx(0.0)
    assert measure_headroom(ctx).values["headroom_ratio"] == pytest.approx(0.2)
    assert 0.0 <= measure_eye_line(ctx).values["eye_line_ratio"] <= 1.0
    empty = AnalysisContext(np.zeros((HEIGHT, WIDTH, 3), np.uint8), FakeBackend([], mask))
    for measure in (
        measure_face_area_ratio,
        measure_center_offset,
        measure_headroom,
        measure_eye_line,
    ):
        assert measure(empty).note == "лицо не обнаружено"
    no_person = AnalysisContext(
        np.zeros((HEIGHT, WIDTH, 3), np.uint8),
        FakeBackend([landmarks], np.zeros((HEIGHT, WIDTH), np.float32)),
    )
    assert measure_headroom(no_person).note == "человек не найден в кадре"
