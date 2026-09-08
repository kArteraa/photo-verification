from itertools import pairwise

import cv2
import numpy as np

from app.analyzers.context import AnalysisContext, BBox
from app.analyzers.sharpness import (
    crop,
    laplacian_variance,
    measure_face_sharpness,
    measure_frame_sharpness,
)
from tests.helpers import FakeBackend, box_landmarks


def noise_image(seed=0, size=128):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 1.0)


def test_blur_reduces_variance_monotonically():
    gray = noise_image()
    variances = [laplacian_variance(gray)]
    for sigma in (1.5, 3.0, 5.0, 8.0):
        variances.append(laplacian_variance(cv2.GaussianBlur(gray, (0, 0), sigma)))
    assert all(a > b for a, b in pairwise(variances))


def test_flat_image_has_zero_variance():
    assert laplacian_variance(np.full((32, 32), 77, np.uint8)) == 0.0
    assert laplacian_variance(np.zeros((0, 0), np.uint8)) == 0.0


def test_crop_rounds_outwards_and_clips():
    gray = np.arange(100, dtype=np.uint8).reshape(10, 10)
    region = crop(gray, BBox(2.3, 1.7, 5.2, 4.1))
    assert region.shape == (4, 4)
    assert region[0, 0] == gray[1, 2]
    assert crop(gray, BBox(-5, -5, 50, 50)).shape == (10, 10)


def test_face_and_frame_measurers():
    gray = noise_image(size=64)
    gray[16:48, 16:48] = cv2.GaussianBlur(gray, (0, 0), 3.0)[16:48, 16:48]
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    landmarks = box_landmarks(16, 16, 47, 47, 64, 64)
    ctx = AnalysisContext(bgr, FakeBackend([landmarks], np.zeros((64, 64), np.float32)))
    face = measure_face_sharpness(ctx).values["var"]
    frame = measure_frame_sharpness(ctx).values["var"]
    assert 0 < face < frame
    assert face == laplacian_variance(crop(gray, ctx.primary_face.bbox))
    assert frame == laplacian_variance(gray)


def test_face_measurer_without_face():
    ctx = AnalysisContext(np.zeros((8, 8, 3), np.uint8), FakeBackend([], np.zeros((8, 8))))
    assert not measure_face_sharpness(ctx).applicable
    assert measure_frame_sharpness(ctx).applicable
