"""Shared test doubles: a fake perception backend, stub measurers and spec builders."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np

from app.analyzers.pose import camera_matrix, canonical_model
from app.analyzers.registry import MeasurerSpec
from app.core.report import Measurement, Number


class FakeBackend:
    """Backend returning hand-made landmarks and masks while counting calls."""

    def __init__(self, landmarks: list[np.ndarray], mask: np.ndarray) -> None:
        self.landmarks = landmarks
        self.mask = mask
        self.detect_calls = 0
        self.segment_calls = 0

    def detect_landmarks(self, rgb: np.ndarray) -> list[np.ndarray]:
        self.detect_calls += 1
        return [np.array(item, dtype=np.float64) for item in self.landmarks]

    def person_confidence(self, rgb: np.ndarray) -> np.ndarray:
        self.segment_calls += 1
        return self.mask


class CallCounter:
    """Mutable counter shared with stub measurers."""

    def __init__(self) -> None:
        self.calls = 0


def constant_measurer(
    name: str,
    keys: tuple[str, ...],
    values: dict[str, Number] | None,
    note: str = "not applicable",
    counter: CallCounter | None = None,
) -> MeasurerSpec:
    """Stub measurer returning fixed values, or a non-applicable measurement when values is None."""

    def measure(ctx: Any) -> Measurement:
        if counter is not None:
            counter.calls += 1
        if values is None:
            return Measurement.not_applicable(name, note)
        return Measurement.of(name, dict(values))

    return MeasurerSpec(name, keys, measure)


def registry_of(*measurers: MeasurerSpec) -> dict[str, MeasurerSpec]:
    return {measurer.name: measurer for measurer in measurers}


def requirement(
    requirement_id: str,
    measurer: str,
    predicate: dict[str, Any],
    kind: str = "hard",
    depends_on: Iterable[str] = (),
    reason_template: str = "value={value}",
    fix_hint: str = "fix it",
    title: str | None = None,
) -> dict[str, Any]:
    return {
        "id": requirement_id,
        "title": title or requirement_id,
        "measurer": measurer,
        "predicate": predicate,
        "kind": kind,
        "depends_on": list(depends_on),
        "reason_template": reason_template,
        "fix_hint": fix_hint,
    }


def spec_data(*requirements: dict[str, Any], name: str = "test") -> dict[str, Any]:
    return {"name": name, "description": "test specification", "requirements": list(requirements)}


def box_landmarks(
    x0: float, y0: float, x1: float, y1: float, width: int, height: int, count: int = 478
) -> np.ndarray:
    """Normalized landmarks filling a pixel box, for bbox-driven logic."""
    side = int(np.ceil(np.sqrt(count)))
    xs = np.linspace(x0, x1, side) / width
    ys = np.linspace(y0, y1, side) / height
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float64)
    return points[:count]


def frontal_landmarks(
    center: tuple[float, float], face_width: float, width: int, height: int
) -> np.ndarray:
    """Normalized 478-point landmarks of a frontal canonical face placed in the frame."""
    model = canonical_model()
    unit, _ = cv2.projectPoints(
        model, np.zeros(3), np.array([0.0, 0.0, 60.0]), camera_matrix(width, height), None
    )
    unit = unit.reshape(-1, 2)
    span = unit[:, 0].max() - unit[:, 0].min()
    points = (unit - unit.mean(axis=0)) * (face_width / span) + np.array(center)
    iris = np.array([points[[33, 133]].mean(axis=0), points[[362, 263]].mean(axis=0)])
    extra = np.repeat(iris, 5, axis=0)
    pixels = np.vstack([points, extra])
    normalized = pixels / np.array([width, height], dtype=np.float64)
    return np.hstack([normalized, np.zeros((len(normalized), 1))])
