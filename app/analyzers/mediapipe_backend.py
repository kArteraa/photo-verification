"""Perception backend on the MediaPipe Tasks API: face landmarks and selfie segmentation."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from app.analyzers.models import model_path

log = logging.getLogger(__name__)

LANDMARKER_MODEL = "face_landmarker.task"
SEGMENTER_MODEL = "selfie_segmenter.tflite"
PERSON_MASK_INDEX = 0


class MediaPipeBackend:
    """Lazy wrapper around FaceLandmarker and ImageSegmenter loaded from model buffers."""

    def __init__(self, models_dir: Path, num_faces: int, min_detection_confidence: float) -> None:
        self._models_dir = Path(models_dir)
        self._num_faces = num_faces
        self._min_detection_confidence = min_detection_confidence
        logging.getLogger("absl").setLevel(logging.ERROR)

    @cached_property
    def _landmarker(self) -> vision.FaceLandmarker:
        buffer = model_path(self._models_dir, LANDMARKER_MODEL).read_bytes()
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=buffer),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self._num_faces,
            min_face_detection_confidence=self._min_detection_confidence,
        )
        log.debug("loading face landmarker (%d faces max)", self._num_faces)
        return vision.FaceLandmarker.create_from_options(options)

    @cached_property
    def _segmenter(self) -> vision.ImageSegmenter:
        buffer = model_path(self._models_dir, SEGMENTER_MODEL).read_bytes()
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_buffer=buffer),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        log.debug("loading selfie segmenter")
        return vision.ImageSegmenter.create_from_options(options)

    def detect_landmarks(self, rgb: np.ndarray) -> list[np.ndarray]:
        """Normalized ``(478, 3)`` landmarks for every detected face."""
        result = self._landmarker.detect(_to_image(rgb))
        return [
            np.array([[point.x, point.y, point.z] for point in face], dtype=np.float64)
            for face in result.face_landmarks
        ]

    def person_confidence(self, rgb: np.ndarray) -> np.ndarray:
        """Person probability map of the image."""
        result = self._segmenter.segment(_to_image(rgb))
        mask = result.confidence_masks[PERSON_MASK_INDEX].numpy_view()
        return np.array(mask.reshape(mask.shape[:2]), dtype=np.float32, copy=True)


def _to_image(rgb: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
