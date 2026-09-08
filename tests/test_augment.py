import cv2
import numpy as np
import pytest

from app.analyzers.sharpness import laplacian_variance
from experiments.augment import (
    background_color,
    blur,
    collage,
    compose_on_canvas,
    gain,
    normalize_background,
    replace_background,
    small_face_scale,
)

SIZE = 96


def portrait():
    rng = np.random.default_rng(0)
    image = np.full((SIZE, SIZE, 3), (200, 210, 220), np.uint8)
    image[24:80, 32:64] = rng.integers(0, 256, size=(56, 32, 3), dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), np.float32)
    mask[24:80, 32:64] = 1.0
    return image, mask


def test_blur_lowers_sharpness():
    image, _ = portrait()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.cvtColor(blur(image, 3.0), cv2.COLOR_BGR2GRAY)
    assert laplacian_variance(blurred) < laplacian_variance(gray)


def test_gain_moves_brightness_and_clips():
    image, _ = portrait()
    assert gain(image, 0.3).mean() < image.mean() < gain(image, 2.4).mean()
    assert gain(image, 2.4).max() == 255
    assert gain(image, 0.3).dtype == np.uint8


def test_background_color_is_median_of_background():
    image, mask = portrait()
    assert background_color(image, mask).tolist() == [200, 210, 220]
    assert background_color(image, np.ones_like(mask)).tolist() == [128, 128, 128]


def test_replace_background_keeps_person_pixels():
    image, mask = portrait()
    texture = np.zeros((SIZE, SIZE, 3), np.uint8)
    result = replace_background(image, mask, texture)
    assert np.array_equal(result[40:64, 44:52], image[40:64, 44:52])
    assert result[0:8, 0:8].max() == 0
    assert result.dtype == np.uint8


def test_normalize_background_flattens_background():
    image, mask = portrait()
    image[0:10, :] = (0, 0, 0)
    result = normalize_background(image, mask)
    assert result[0:8, 0:8].tolist()[0][0] == [200, 210, 220]


def test_compose_on_canvas_scales_and_shifts():
    image, mask = portrait()
    scale = small_face_scale(base_ratio=0.2, target_ratio=0.05)
    assert scale == pytest.approx(0.5)
    small = compose_on_canvas(image, mask, scale, (0.0, 0.0))
    dark = (small.astype(int).sum(axis=2) < 600) & (small.astype(int).sum(axis=2) != 630)
    ys, xs = np.nonzero(dark)
    assert ys.min() > 20 and ys.max() < 76
    assert xs.min() > 30 and xs.max() < 66
    shifted = compose_on_canvas(image, mask, 1.0, (0.25, 0.0))
    left_half = shifted[:, : SIZE // 2].astype(int).sum(axis=2) != 630
    right_half = shifted[:, SIZE // 2 :].astype(int).sum(axis=2) != 630
    assert right_half.sum() > left_half.sum()
    assert compose_on_canvas(image, mask, 1.0, (2.0, 2.0)).shape == image.shape


def test_collage_size():
    left, _ = portrait()
    right = np.zeros((48, 60, 3), np.uint8)
    result = collage(left, right)
    assert result.shape[0] == 48
    assert result.shape[1] == 48 + 60
