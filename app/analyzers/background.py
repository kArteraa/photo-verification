"""Background uniformity and lightness behind the segmented person."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.analyzers.context import AnalysisContext
from app.analyzers.registry import register
from app.core.report import Measurement

PERSON_THRESHOLD = 0.5
MIN_BACKGROUND_FRACTION = 0.01
MIN_MEAN_FOR_CV = 1.0
EMPTY_BACKGROUND_NOTE = "фон не виден"


@dataclass(frozen=True)
class BackgroundStats:
    """Coefficient of variation and mean lightness of the background pixels."""

    cv: float
    mean_lightness: float
    fraction: float


def background_mask(person_confidence: np.ndarray, person_threshold: float) -> np.ndarray:
    """Boolean mask of pixels that are not part of the person."""
    return np.asarray(person_confidence) <= person_threshold


def background_stats(
    gray: np.ndarray,
    person_confidence: np.ndarray,
    person_threshold: float = PERSON_THRESHOLD,
    min_fraction: float = MIN_BACKGROUND_FRACTION,
) -> BackgroundStats | None:
    """Statistics of the background, or None when too little background is visible."""
    mask = background_mask(person_confidence, person_threshold)
    fraction = float(mask.mean())
    if fraction < min_fraction:
        return None
    pixels = np.asarray(gray, dtype=np.float64)[mask]
    mean = float(pixels.mean())
    cv = float(pixels.std() / max(mean, MIN_MEAN_FOR_CV))
    return BackgroundStats(cv, mean, fraction)


@register("background.cv", ("cv", "fraction"))
def measure_background_cv(ctx: AnalysisContext) -> Measurement:
    """Coefficient of variation of background brightness."""
    stats = background_stats(ctx.gray, ctx.person_confidence)
    if stats is None:
        return Measurement.not_applicable("background.cv", EMPTY_BACKGROUND_NOTE)
    return Measurement.of("background.cv", {"cv": stats.cv, "fraction": stats.fraction})


@register("background.mean_lightness", ("mean_lightness",))
def measure_background_lightness(ctx: AnalysisContext) -> Measurement:
    """Mean brightness of the background."""
    stats = background_stats(ctx.gray, ctx.person_confidence)
    if stats is None:
        return Measurement.not_applicable("background.mean_lightness", EMPTY_BACKGROUND_NOTE)
    return Measurement.of("background.mean_lightness", {"mean_lightness": stats.mean_lightness})
