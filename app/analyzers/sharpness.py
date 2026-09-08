"""Sharpness as the variance of the Laplacian, on the face region and on the frame."""

from __future__ import annotations

import cv2
import numpy as np

from app.analyzers.common import no_face
from app.analyzers.context import AnalysisContext, BBox
from app.analyzers.registry import register
from app.core.report import Measurement


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian response of a grayscale image."""
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop(gray: np.ndarray, bbox: BBox) -> np.ndarray:
    """Sub-image covered by the box, rounded outwards to whole pixels."""
    height, width = gray.shape[:2]
    x0 = max(0, int(np.floor(bbox.x0)))
    y0 = max(0, int(np.floor(bbox.y0)))
    x1 = min(width, int(np.ceil(bbox.x1)))
    y1 = min(height, int(np.ceil(bbox.y1)))
    return gray[y0:y1, x0:x1]


@register("sharpness.face_laplacian_var", ("var",))
def measure_face_sharpness(ctx: AnalysisContext) -> Measurement:
    """Laplacian variance inside the primary face box."""
    face = ctx.primary_face
    if face is None:
        return no_face("sharpness.face_laplacian_var")
    return Measurement.of(
        "sharpness.face_laplacian_var", {"var": laplacian_variance(crop(ctx.gray, face.bbox))}
    )


@register("sharpness.frame_laplacian_var", ("var",))
def measure_frame_sharpness(ctx: AnalysisContext) -> Measurement:
    """Laplacian variance over the whole frame."""
    return Measurement.of("sharpness.frame_laplacian_var", {"var": laplacian_variance(ctx.gray)})
