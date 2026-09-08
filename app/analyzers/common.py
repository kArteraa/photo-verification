"""Helpers shared by the measurer wrappers."""

from __future__ import annotations

from app.core.report import Measurement

NO_FACE_NOTE = "лицо не обнаружено"


def no_face(measurer: str) -> Measurement:
    """Measurement reported when a face-dependent measurer finds no face."""
    return Measurement.not_applicable(measurer, NO_FACE_NOTE)
