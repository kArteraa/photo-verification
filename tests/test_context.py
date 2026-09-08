import numpy as np
import pytest

from app.analyzers.context import AnalysisContext, BBox, build_face
from tests.helpers import FakeBackend, box_landmarks

WIDTH, HEIGHT = 200, 100


def image(width=WIDTH, height=HEIGHT):
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255
    return bgr


def test_build_face_scales_and_clips_bbox():
    landmarks = np.array([[-0.1, 0.2, 0.0], [0.5, 0.5, 0.0], [1.2, 0.9, 0.0]])
    face = build_face(landmarks, WIDTH, HEIGHT)
    assert face.landmarks_px[0].tolist() == [-20.0, 20.0]
    assert face.bbox == BBox(0.0, 20.0, 200.0, 90.0)
    assert face.bbox.width == 200.0
    assert face.bbox.center == (100.0, 55.0)


def test_faces_sorted_by_area_and_detected_once():
    small = box_landmarks(10, 10, 30, 30, WIDTH, HEIGHT)
    large = box_landmarks(50, 10, 150, 90, WIDTH, HEIGHT)
    backend = FakeBackend([small, large], np.zeros((HEIGHT, WIDTH), dtype=np.float32))
    ctx = AnalysisContext(image(), backend)
    assert len(ctx.faces) == 2
    assert ctx.primary_face.bbox.area == pytest.approx(100 * 80)
    assert ctx.faces[1].bbox.area == pytest.approx(20 * 20)
    _ = ctx.faces, ctx.primary_face
    assert backend.detect_calls == 1


def test_no_faces():
    ctx = AnalysisContext(image(), FakeBackend([], np.zeros((HEIGHT, WIDTH), dtype=np.float32)))
    assert ctx.faces == ()
    assert ctx.primary_face is None


def test_person_confidence_cached_and_validated():
    mask = np.full((HEIGHT, WIDTH), 0.7, dtype=np.float32)
    backend = FakeBackend([], mask)
    ctx = AnalysisContext(image(), backend)
    assert ctx.person_confidence.dtype == np.float32
    assert float(ctx.person_confidence.mean()) == pytest.approx(0.7)
    _ = ctx.person_confidence
    assert backend.segment_calls == 1
    wrong = AnalysisContext(image(), FakeBackend([], np.zeros((10, 10), dtype=np.float32)))
    with pytest.raises(ValueError, match="person mask shape"):
        _ = wrong.person_confidence


def test_colour_conversions():
    ctx = AnalysisContext(image(), FakeBackend([], np.zeros((HEIGHT, WIDTH), dtype=np.float32)))
    assert ctx.width == WIDTH
    assert ctx.height == HEIGHT
    assert ctx.rgb[0, 0].tolist() == [0, 0, 255]
    assert ctx.rgb.flags["C_CONTIGUOUS"]
    assert ctx.gray.shape == (HEIGHT, WIDTH)
