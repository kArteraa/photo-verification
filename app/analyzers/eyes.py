"""Eye openness via the eye aspect ratio (EAR) on face mesh landmarks.

Landmark indices follow the MediaPipe face mesh: six points per eye ordered as
outer corner, upper lid (two points), inner corner, lower lid (two points), so
that ``EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|)``.
"""

from __future__ import annotations

import numpy as np

from app.analyzers.common import no_face
from app.analyzers.context import AnalysisContext
from app.analyzers.registry import register
from app.core.report import Measurement

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MIN_CORNER_DISTANCE = 1e-6


def eye_aspect_ratio(points: np.ndarray) -> float:
    """EAR of one eye given its six landmarks as an ``(6, 2)`` array."""
    p1, p2, p3, p4, p5, p6 = np.asarray(points, dtype=np.float64)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < MIN_CORNER_DISTANCE:
        return 0.0
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    return float(vertical / (2.0 * horizontal))


@register("eyes.min_ear", ("min_ear", "left_ear", "right_ear"))
def measure_min_ear(ctx: AnalysisContext) -> Measurement:
    """Smallest EAR over both eyes of the primary face."""
    face = ctx.primary_face
    if face is None:
        return no_face("eyes.min_ear")
    left = eye_aspect_ratio(face.landmarks_px[list(LEFT_EYE)])
    right = eye_aspect_ratio(face.landmarks_px[list(RIGHT_EYE)])
    return Measurement.of(
        "eyes.min_ear", {"min_ear": min(left, right), "left_ear": left, "right_ear": right}
    )
