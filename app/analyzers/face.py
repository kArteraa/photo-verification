"""Face count measurer."""

from __future__ import annotations

from collections.abc import Sequence

from app.analyzers.context import AnalysisContext, Face
from app.analyzers.registry import register
from app.core.report import Measurement


def count_faces(faces: Sequence[Face]) -> int:
    """Number of detected faces."""
    return len(faces)


@register("face.count", ("count",))
def measure_face_count(ctx: AnalysisContext) -> Measurement:
    """Count faces; zero faces is a legitimate value, not an undefined measurement."""
    return Measurement.of("face.count", {"count": count_faces(ctx.faces)})
