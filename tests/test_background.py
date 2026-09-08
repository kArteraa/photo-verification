import cv2
import numpy as np
import pytest

from app.analyzers.background import (
    background_mask,
    background_stats,
    erosion_radius,
    measure_background_cv,
    measure_background_lightness,
)
from app.analyzers.context import AnalysisContext
from tests.helpers import FakeBackend

SIZE = 64


def person_mask(size=SIZE):
    mask = np.zeros((size, size), np.float32)
    mask[8:56, 20:44] = 0.9
    return mask


def checkerboard(size=SIZE, cell=8):
    tiles = (np.indices((size, size)).sum(axis=0) // cell) % 2
    return (tiles * 200 + 30).astype(np.uint8)


def test_uniform_background_ignores_noisy_person():
    rng = np.random.default_rng(0)
    gray = np.full((SIZE, SIZE), 120, np.uint8)
    mask = person_mask()
    gray[mask > 0.5] = rng.integers(0, 256, size=int((mask > 0.5).sum()))
    stats = background_stats(gray, mask)
    assert stats.cv == pytest.approx(0.0)
    assert stats.mean_lightness == pytest.approx(120.0)
    assert 0.0 < stats.fraction < float((mask <= 0.5).mean())


def test_checkerboard_background_has_high_cv():
    stats = background_stats(checkerboard(), person_mask())
    assert stats.cv > 0.5


def test_all_person_is_none():
    assert background_stats(np.zeros((SIZE, SIZE), np.uint8), np.ones((SIZE, SIZE))) is None


def test_black_background_has_no_division_error():
    stats = background_stats(np.zeros((SIZE, SIZE), np.uint8), person_mask())
    assert stats.cv == 0.0
    assert stats.mean_lightness == 0.0


def test_background_mask_threshold_and_erosion():
    confidence = np.zeros((SIZE, SIZE), np.float32)
    confidence[20:44, 20:44] = 1.0
    confidence[10, 10] = 0.5
    mask = background_mask(confidence, 0.5)
    radius = erosion_radius(confidence.shape)
    assert radius == 1
    assert mask[10, 10]
    assert not mask[30, 30]
    assert not mask[20 - radius, 30]
    assert mask[20 - radius - 1, 30]


def test_boundary_halo_is_ignored():
    gray = np.full((SIZE, SIZE), 120, np.uint8)
    mask = person_mask()
    halo = cv2.dilate((mask > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    gray[halo & ~(mask > 0.5)] = 255
    assert background_stats(gray, mask).cv == pytest.approx(0.0)
    assert erosion_radius((512, 512)) == 8


def test_measurers():
    gray = checkerboard()
    bgr = np.repeat(gray[:, :, None], 3, axis=2)
    ctx = AnalysisContext(bgr, FakeBackend([], person_mask()))
    assert measure_background_cv(ctx).values["cv"] > 0.5
    assert 0 < measure_background_cv(ctx).values["fraction"] < 1
    assert measure_background_lightness(ctx).values["mean_lightness"] > 0
    full = AnalysisContext(bgr, FakeBackend([], np.ones((SIZE, SIZE), np.float32)))
    assert measure_background_cv(full).note == "фон не виден"
    assert not measure_background_lightness(full).applicable
