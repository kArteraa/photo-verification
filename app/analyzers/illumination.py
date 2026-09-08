"""Left-right illumination asymmetry of the face as a side-light indicator."""

from __future__ import annotations

import numpy as np

from app.analyzers.common import no_face
from app.analyzers.context import AnalysisContext, BBox
from app.analyzers.registry import register
from app.analyzers.sharpness import crop
from app.core.report import Measurement

NOSE_BRIDGE = 6
MIN_TOTAL_BRIGHTNESS = 1e-6


def face_asymmetry(gray: np.ndarray, landmarks_px: np.ndarray, bbox: BBox) -> float:
    """Normalized brightness difference between the image-left and image-right face halves.

    Positive values mean the left half (as seen in the image) is brighter.
    """
    region = np.asarray(crop(gray, bbox), dtype=np.float64)
    if region.size == 0:
        return 0.0
    split = round(float(landmarks_px[NOSE_BRIDGE][0]) - max(0.0, float(np.floor(bbox.x0))))
    split = min(max(split, 1), region.shape[1] - 1)
    left = region[:, :split].mean()
    right = region[:, split:].mean()
    total = left + right
    if total < MIN_TOTAL_BRIGHTNESS:
        return 0.0
    return float((left - right) / total)


@register("illumination.face_asymmetry", ("asymmetry",))
def measure_face_asymmetry(ctx: AnalysisContext) -> Measurement:
    """Illumination asymmetry of the primary face."""
    face = ctx.primary_face
    if face is None:
        return no_face("illumination.face_asymmetry")
    value = face_asymmetry(ctx.gray, face.landmarks_px, face.bbox)
    return Measurement.of("illumination.face_asymmetry", {"asymmetry": value})
