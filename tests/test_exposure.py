import numpy as np
import pytest

from app.analyzers.exposure import exposure_stats


def test_exact_clip_fractions_and_mean():
    gray = np.full((10, 10), 128, np.uint8)
    gray[0, :5] = 0
    gray[1, :3] = 255
    stats = exposure_stats(gray)
    assert stats["clip_low"] == pytest.approx(0.05)
    assert stats["clip_high"] == pytest.approx(0.03)
    assert stats["mean_brightness"] == pytest.approx(gray.mean())
    assert stats["std_brightness"] == pytest.approx(gray.std())


def test_gain_moves_mean_and_clipping():
    rng = np.random.default_rng(1)
    gray = rng.integers(60, 200, size=(50, 50)).astype(np.uint8)
    dark = np.clip(gray.astype(float) * 0.3, 0, 255).astype(np.uint8)
    bright = np.clip(gray.astype(float) * 2.4, 0, 255).astype(np.uint8)
    base, low, high = exposure_stats(gray), exposure_stats(dark), exposure_stats(bright)
    assert low["mean_brightness"] < base["mean_brightness"] < high["mean_brightness"]
    assert high["clip_high"] > base["clip_high"] == 0.0
    assert low["clip_low"] == 0.0
