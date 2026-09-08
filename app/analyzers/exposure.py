"""Exposure statistics from the grayscale histogram."""

from __future__ import annotations

import numpy as np

from app.analyzers.context import AnalysisContext
from app.analyzers.registry import register
from app.core.report import Measurement, Values

CLIP_LOW = 5
CLIP_HIGH = 250


def exposure_stats(gray: np.ndarray) -> Values:
    """Mean and standard deviation of brightness and the clipped-pixel fractions."""
    pixels = np.asarray(gray, dtype=np.float64)
    return {
        "mean_brightness": float(pixels.mean()),
        "std_brightness": float(pixels.std()),
        "clip_low": float(np.mean(pixels < CLIP_LOW)),
        "clip_high": float(np.mean(pixels > CLIP_HIGH)),
    }


@register("exposure.stats", ("mean_brightness", "std_brightness", "clip_low", "clip_high"))
def measure_exposure(ctx: AnalysisContext) -> Measurement:
    """Exposure statistics of the whole frame."""
    return Measurement.of("exposure.stats", exposure_stats(ctx.gray))
