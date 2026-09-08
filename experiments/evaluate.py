"""Run the full pipeline over a labelled dataset and collect one row per image."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from app.analyzers.context import AnalysisContext, FaceBackend
from app.conclusion.template_gen import RECOMMENDATIONS_LABEL, VIOLATIONS_LABEL, render_conclusion
from app.conclusion.verify import extract_footer, verify_conclusion
from app.core.engine import Engine
from app.core.report import Measurement, Report, Status
from app.imageio import read_bgr
from experiments.common import LABEL_COLUMNS

log = logging.getLogger(__name__)

STATUS_PREFIX = "status."
MEASURE_PREFIX = "m."
CLAIMED_SEPARATOR = ";"


def status_column(requirement_id: str) -> str:
    """Column holding the verdict status of a requirement."""
    return f"{STATUS_PREFIX}{requirement_id}"


def measure_column(measurer: str, key: str) -> str:
    """Column holding one raw measurement value."""
    return f"{MEASURE_PREFIX}{measurer}.{key}"


def evaluate_image(path: Path, engine: Engine, backend: FaceBackend) -> tuple[Report, dict, float]:
    """Measure everything once, build the report and time the whole step."""
    started = time.perf_counter()
    ctx = AnalysisContext(read_bgr(path), backend)
    measurements = engine.measure_all(ctx)
    report = engine.evaluate_measured(path.name, measurements)
    return report, measurements, (time.perf_counter() - started) * 1000.0


def row_for(
    label: pd.Series,
    report: Report,
    measurements: dict[str, Measurement],
    elapsed_ms: float,
    engine: Engine,
) -> dict:
    """Flatten one evaluation into a table row."""
    row = {column: label[column] for column in LABEL_COLUMNS}
    row["accepted"] = report.accepted
    row["elapsed_ms"] = elapsed_ms
    for verdict in report.verdicts:
        row[status_column(verdict.id)] = verdict.status.value
    for measurer in engine.measurers:
        measurement = measurements[measurer.name]
        for key in measurer.keys:
            value = measurement.values.get(key) if measurement.applicable else None
            row[measure_column(measurer.name, key)] = np.nan if value is None else value
    conclusion = render_conclusion(report)
    row["conclusion_ok"] = verify_conclusion(conclusion, report).ok
    claimed = set()
    for label in (VIOLATIONS_LABEL, RECOMMENDATIONS_LABEL):
        claimed |= extract_footer(conclusion, label) or set()
    row["claimed"] = CLAIMED_SEPARATOR.join(sorted(claimed))
    row["failed"] = CLAIMED_SEPARATOR.join(
        verdict.id for verdict in report.verdicts if verdict.status is Status.FAIL
    )
    return row


def evaluate_dataset(
    labels: pd.DataFrame, images_dir: Path, engine: Engine, backend: FaceBackend
) -> pd.DataFrame:
    """Evaluate every labelled image and return the prediction table."""
    rows = []
    for index, (_, label) in enumerate(labels.iterrows(), start=1):
        path = Path(images_dir) / label["file"]
        report, measurements, elapsed_ms = evaluate_image(path, engine, backend)
        rows.append(row_for(label, report, measurements, elapsed_ms, engine))
        if index % 100 == 0:
            log.info("evaluated %d/%d", index, len(labels))
    return pd.DataFrame(rows)


def split_ids(value: str) -> set[str]:
    """Parse a semicolon-joined id list."""
    return {item for item in str(value).split(CLAIMED_SEPARATOR) if item}


def requirement_ids(columns: Iterable[str]) -> list[str]:
    """Requirement ids present as status columns."""
    return [column[len(STATUS_PREFIX) :] for column in columns if column.startswith(STATUS_PREFIX)]
