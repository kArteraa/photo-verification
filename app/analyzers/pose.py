"""Head pose from face landmarks by solving the Perspective-n-Point problem.

The 468 MediaPipe face mesh landmarks are matched against the canonical face
model shipped with MediaPipe (``data/canonical_face_model.json``, centimetres,
+X right, +Y up, +Z towards the viewer).  The model is flipped into the OpenCV
camera frame (+Y down, +Z into the scene) so that a frontal face yields the
identity rotation.  A frontal initial guess keeps the iterative solver in the
physically meaningful minimum.

Angle conventions (degrees):

* ``yaw > 0`` – the face is turned towards the left edge of the image;
* ``pitch > 0`` – the chin is raised (face looks up);
* ``roll > 0`` – the head is tilted clockwise as seen in the image.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np

from app.analyzers.common import no_face
from app.analyzers.context import AnalysisContext
from app.analyzers.registry import register
from app.core.report import Measurement

MODEL_PATH = Path(__file__).parent / "data" / "canonical_face_model.json"
TO_CAMERA_FRAME = np.array([1.0, -1.0, -1.0])
INITIAL_DISTANCE_CM = 60.0


@dataclass(frozen=True)
class HeadPose:
    """Head orientation in degrees."""

    yaw: float
    pitch: float
    roll: float


@cache
def canonical_model() -> np.ndarray:
    """Canonical face vertices in the OpenCV camera frame, shape ``(468, 3)``."""
    vertices = np.array(json.loads(MODEL_PATH.read_text(encoding="utf-8")), dtype=np.float64)
    return vertices * TO_CAMERA_FRAME


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics with focal length equal to the image width."""
    return np.array(
        [[width, 0.0, width / 2.0], [0.0, width, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def angles_from_rotation(rotation: np.ndarray) -> HeadPose:
    """Yaw, pitch and roll from a rotation matrix, avoiding Euler-order ambiguity."""
    forward = rotation @ np.array([0.0, 0.0, -1.0])
    right = rotation @ np.array([1.0, 0.0, 0.0])
    yaw = math.atan2(-forward[0], -forward[2])
    pitch = math.atan2(-forward[1], math.hypot(forward[0], forward[2]))
    roll = math.atan2(right[1], right[0])
    return HeadPose(math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def estimate_head_pose(landmarks_px: np.ndarray, width: int, height: int) -> HeadPose | None:
    """Solve PnP between pixel landmarks and the canonical model."""
    model = canonical_model()
    image_points = np.ascontiguousarray(landmarks_px[: len(model)], dtype=np.float64)
    if image_points.shape != (len(model), 2):
        return None
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [INITIAL_DISTANCE_CM]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        model,
        image_points,
        camera_matrix(width, height),
        None,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return angles_from_rotation(rotation)


@register("pose.yaw_pitch", ("yaw", "pitch", "roll"))
def measure_head_pose(ctx: AnalysisContext) -> Measurement:
    """Head pose of the primary face."""
    face = ctx.primary_face
    if face is None:
        return no_face("pose.yaw_pitch")
    pose = estimate_head_pose(face.landmarks_px, ctx.width, ctx.height)
    if pose is None:
        return Measurement.not_applicable("pose.yaw_pitch", "поза головы не определена")
    return Measurement.of(
        "pose.yaw_pitch", {"yaw": pose.yaw, "pitch": pose.pitch, "roll": pose.roll}
    )
