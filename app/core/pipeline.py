"""End-to-end check of one image file: read, analyze, apply rules, time it."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from app.analyzers.context import AnalysisContext, FaceBackend
from app.core.engine import Engine
from app.core.report import Report
from app.imageio import read_bgr


@dataclass(frozen=True)
class CheckResult:
    """Report for one image and the wall-clock time spent producing it."""

    report: Report
    elapsed_ms: float


def check_image(path: Path, engine: Engine, backend: FaceBackend) -> CheckResult:
    """Run the full facts-and-rules pipeline on an image file."""
    started = time.perf_counter()
    ctx = AnalysisContext(read_bgr(path), backend)
    report = engine.evaluate(ctx, Path(path).name)
    return CheckResult(report, (time.perf_counter() - started) * 1000.0)
