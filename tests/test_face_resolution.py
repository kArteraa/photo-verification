import numpy as np
import pytest

from app.analyzers import REGISTRY
from app.analyzers.context import AnalysisContext
from app.analyzers.face import measure_face_count
from app.analyzers.resolution import measure_resolution, size_stats
from tests.helpers import FakeBackend, box_landmarks


def context(landmarks):
    return AnalysisContext(
        np.zeros((100, 160, 3), np.uint8), FakeBackend(landmarks, np.zeros((100, 160), np.float32))
    )


def test_face_count_zero_is_applicable():
    measurement = measure_face_count(context([]))
    assert measurement.applicable
    assert measurement.values == {"count": 0}


def test_face_count_two():
    faces = [box_landmarks(0, 0, 40, 40, 160, 100), box_landmarks(80, 0, 150, 60, 160, 100)]
    assert measure_face_count(context(faces)).values == {"count": 2}


def test_resolution():
    assert size_stats(640, 480) == {
        "width": 640,
        "height": 480,
        "min_side": 480,
        "aspect_ratio": pytest.approx(4 / 3),
    }
    assert measure_resolution(context([])).values["min_side"] == 100


def test_registry_contains_all_measurers():
    expected = {
        "face.count",
        "pose.yaw_pitch",
        "eyes.min_ear",
        "sharpness.face_laplacian_var",
        "sharpness.frame_laplacian_var",
        "exposure.stats",
        "background.cv",
        "background.mean_lightness",
        "geometry.face_area_ratio",
        "geometry.center_offset",
        "geometry.headroom_ratio",
        "geometry.eye_line_ratio",
        "resolution.size",
        "illumination.face_asymmetry",
    }
    assert expected <= set(REGISTRY)
    for spec in REGISTRY.values():
        assert spec.keys
