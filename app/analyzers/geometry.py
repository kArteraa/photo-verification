"""Frame geometry: face size, centring, headroom and eye-line position."""

from __future__ import annotations

import numpy as np

from app.analyzers.background import PERSON_THRESHOLD
from app.analyzers.common import no_face
from app.analyzers.context import AnalysisContext, BBox
from app.analyzers.registry import register
from app.core.report import Measurement

LEFT_IRIS = 468
RIGHT_IRIS = 473
HEADROOM_BAND_MARGIN = 0.25
NO_PERSON_NOTE = "человек не найден в кадре"


def face_area_ratio(bbox: BBox, width: int, height: int) -> float:
    """Fraction of the frame covered by the face box."""
    return float(bbox.area) / float(width * height)


def center_offset(bbox: BBox, width: int, height: int) -> tuple[float, float, float]:
    """Face-centre displacement ``(dx, dy, offset)`` as fractions of the frame size."""
    cx, cy = bbox.center
    dx = (cx - width / 2.0) / width
    dy = (cy - height / 2.0) / height
    return float(dx), float(dy), float(np.hypot(dx, dy))


def headroom_ratio(
    person_confidence: np.ndarray, bbox: BBox, person_threshold: float = PERSON_THRESHOLD
) -> float | None:
    """Distance from the top edge to the top of the person, as a fraction of the height.

    Only a vertical band around the face box is inspected so shoulders or a
    second person at the frame edge do not affect the result.
    """
    height, width = person_confidence.shape[:2]
    margin = bbox.width * HEADROOM_BAND_MARGIN
    x0 = max(0, int(np.floor(bbox.x0 - margin)))
    x1 = min(width, int(np.ceil(bbox.x1 + margin)))
    band = np.asarray(person_confidence)[:, x0:x1] > person_threshold
    rows = np.flatnonzero(band.any(axis=1))
    if rows.size == 0:
        return None
    return float(rows[0]) / float(height)


def eye_line_ratio(landmarks_px: np.ndarray, height: int) -> float:
    """Vertical position of the eye line (mean of both iris centres) as a fraction of the height."""
    eye_y = (landmarks_px[LEFT_IRIS][1] + landmarks_px[RIGHT_IRIS][1]) / 2.0
    return float(eye_y / height)


@register("geometry.face_area_ratio", ("face_area_ratio",))
def measure_face_area_ratio(ctx: AnalysisContext) -> Measurement:
    """Fraction of the frame covered by the primary face."""
    face = ctx.primary_face
    if face is None:
        return no_face("geometry.face_area_ratio")
    ratio = face_area_ratio(face.bbox, ctx.width, ctx.height)
    return Measurement.of("geometry.face_area_ratio", {"face_area_ratio": ratio})


@register("geometry.center_offset", ("offset", "dx", "dy"))
def measure_center_offset(ctx: AnalysisContext) -> Measurement:
    """Displacement of the primary face from the frame centre."""
    face = ctx.primary_face
    if face is None:
        return no_face("geometry.center_offset")
    dx, dy, offset = center_offset(face.bbox, ctx.width, ctx.height)
    return Measurement.of("geometry.center_offset", {"offset": offset, "dx": dx, "dy": dy})


@register("geometry.headroom_ratio", ("headroom_ratio",))
def measure_headroom(ctx: AnalysisContext) -> Measurement:
    """Space above the head of the primary face."""
    face = ctx.primary_face
    if face is None:
        return no_face("geometry.headroom_ratio")
    ratio = headroom_ratio(ctx.person_confidence, face.bbox)
    if ratio is None:
        return Measurement.not_applicable("geometry.headroom_ratio", NO_PERSON_NOTE)
    return Measurement.of("geometry.headroom_ratio", {"headroom_ratio": ratio})


@register("geometry.eye_line_ratio", ("eye_line_ratio",))
def measure_eye_line(ctx: AnalysisContext) -> Measurement:
    """Vertical eye-line position of the primary face."""
    face = ctx.primary_face
    if face is None:
        return no_face("geometry.eye_line_ratio")
    ratio = eye_line_ratio(face.landmarks_px, ctx.height)
    return Measurement.of("geometry.eye_line_ratio", {"eye_line_ratio": ratio})
