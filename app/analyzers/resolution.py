"""Image size and aspect ratio."""

from __future__ import annotations

from app.analyzers.context import AnalysisContext
from app.analyzers.registry import register
from app.core.report import Measurement, Values


def size_stats(width: int, height: int) -> Values:
    """Width, height, shorter side and width-to-height ratio."""
    return {
        "width": width,
        "height": height,
        "min_side": min(width, height),
        "aspect_ratio": width / height,
    }


@register("resolution.size", ("min_side", "width", "height", "aspect_ratio"))
def measure_resolution(ctx: AnalysisContext) -> Measurement:
    """Frame dimensions."""
    return Measurement.of("resolution.size", size_stats(ctx.width, ctx.height))
