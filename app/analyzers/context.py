"""Per-image analysis context sharing face detection and segmentation between measurers.

The context owns colour-space conversions and caches the backend outputs, so
face landmarks and the person mask are computed at most once per image no
matter how many measurers consume them.  The backend is injected, which lets
tests substitute a fake that returns hand-made landmarks and masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in pixel coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """Horizontal extent."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Vertical extent."""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Box centre as ``(x, y)``."""
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


@dataclass(frozen=True)
class Face:
    """One detected face: normalized landmarks, pixel landmarks and the clipped box."""

    landmarks_norm: np.ndarray
    landmarks_px: np.ndarray
    bbox: BBox


class FaceBackend(Protocol):
    """Perception backend: landmark detection and person segmentation on RGB images."""

    def detect_landmarks(self, rgb: np.ndarray) -> list[np.ndarray]:
        """Return one ``(N, 3)`` array of normalized landmarks per detected face."""

    def person_confidence(self, rgb: np.ndarray) -> np.ndarray:
        """Return an ``(H, W)`` float32 map of person probability in ``[0, 1]``."""


def build_face(landmarks_norm: np.ndarray, width: int, height: int) -> Face:
    """Convert normalized landmarks into a :class:`Face` with a frame-clipped bounding box."""
    scale = np.array([width, height], dtype=np.float64)
    landmarks_px = np.asarray(landmarks_norm, dtype=np.float64)[:, :2] * scale
    x0, y0 = landmarks_px.min(axis=0)
    x1, y1 = landmarks_px.max(axis=0)
    bbox = BBox(
        max(0.0, float(x0)),
        max(0.0, float(y0)),
        min(float(width), float(x1)),
        min(float(height), float(y1)),
    )
    return Face(np.asarray(landmarks_norm, dtype=np.float64), landmarks_px, bbox)


class AnalysisContext:
    """Lazy, cached view of one BGR image for the measurers."""

    def __init__(self, bgr: np.ndarray, backend: FaceBackend) -> None:
        self.bgr = bgr
        self.height, self.width = bgr.shape[:2]
        self._backend = backend

    @cached_property
    def rgb(self) -> np.ndarray:
        """Contiguous RGB copy of the image."""
        return np.ascontiguousarray(cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB))

    @cached_property
    def gray(self) -> np.ndarray:
        """Grayscale view of the image."""
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)

    @cached_property
    def faces(self) -> tuple[Face, ...]:
        """Detected faces sorted by descending box area."""
        faces = [
            build_face(landmarks, self.width, self.height)
            for landmarks in self._backend.detect_landmarks(self.rgb)
        ]
        return tuple(sorted(faces, key=lambda face: -face.bbox.area))

    @property
    def primary_face(self) -> Face | None:
        """The largest detected face, if any."""
        return self.faces[0] if self.faces else None

    @cached_property
    def person_confidence(self) -> np.ndarray:
        """Person probability map with the image's shape."""
        confidence = np.asarray(self._backend.person_confidence(self.rgb), dtype=np.float32)
        if confidence.shape != (self.height, self.width):
            expected = (self.height, self.width)
            raise ValueError(f"person mask shape {confidence.shape} differs from image {expected}")
        return confidence
