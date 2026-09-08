"""Controlled augmentations that turn a valid portrait into a labelled violation."""

from __future__ import annotations

import cv2
import numpy as np

BLUR_SIGMAS = (1.5, 3.0, 5.0, 8.0)
DARK_GAINS = (0.3, 0.5)
BRIGHT_GAINS = (1.8, 2.4)
SMALL_FACE_RATIOS = (0.15, 0.25)
SHIFT_RANGE = (0.15, 0.30)
FEATHER_SIGMA = 3.0
PERSON_THRESHOLD = 0.5
CANVAS_BORDER = 8


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of the whole frame."""
    return cv2.GaussianBlur(image, (0, 0), sigma)


def gain(image: np.ndarray, factor: float) -> np.ndarray:
    """Linear brightness scaling with clipping."""
    return np.clip(image.astype(np.float64) * factor, 0, 255).astype(np.uint8)


def feathered_alpha(person_confidence: np.ndarray) -> np.ndarray:
    """Soft person alpha in ``[0, 1]`` with shape ``(H, W, 1)``."""
    hard = (person_confidence > PERSON_THRESHOLD).astype(np.float32)
    soft = cv2.GaussianBlur(hard, (0, 0), FEATHER_SIGMA)
    return np.clip(soft, 0.0, 1.0)[:, :, None]


def replace_background(
    image: np.ndarray, person_confidence: np.ndarray, background: np.ndarray
) -> np.ndarray:
    """Composite the person over another background of the same size."""
    alpha = feathered_alpha(person_confidence)
    mixed = image.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def background_color(image: np.ndarray, person_confidence: np.ndarray) -> np.ndarray:
    """Median colour of the background pixels, or mid grey when the person fills the frame."""
    mask = person_confidence <= PERSON_THRESHOLD
    if mask.sum() == 0:
        return np.array([128, 128, 128], dtype=np.uint8)
    return np.median(image[mask], axis=0).astype(np.uint8)


def uniform_canvas(height: int, width: int, color: np.ndarray) -> np.ndarray:
    """Flat canvas of one colour."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def normalize_background(image: np.ndarray, person_confidence: np.ndarray) -> np.ndarray:
    """Person on a flat canvas of the original background colour."""
    height, width = image.shape[:2]
    canvas = uniform_canvas(height, width, background_color(image, person_confidence))
    return replace_background(image, person_confidence, canvas)


def compose_on_canvas(
    image: np.ndarray,
    person_confidence: np.ndarray,
    scale: float,
    shift: tuple[float, float],
) -> np.ndarray:
    """Scale the person cut-out and paste it, shifted by frame fractions, on a flat canvas."""
    height, width = image.shape[:2]
    canvas = uniform_canvas(height, width, background_color(image, person_confidence))
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(feathered_alpha(person_confidence), (new_w, new_h))[:, :, None]
    left = round((width - new_w) / 2 + shift[0] * width)
    top = round((height - new_h) / 2 + shift[1] * height)
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + new_w, width), min(top + new_h, height)
    if x1 <= x0 or y1 <= y0:
        return canvas
    src = small[y0 - top : y1 - top, x0 - left : x1 - left].astype(np.float32)
    a = alpha[y0 - top : y1 - top, x0 - left : x1 - left]
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(np.rint(src * a + region * (1.0 - a)), 0, 255).astype(np.uint8)
    return canvas


def small_face_scale(base_ratio: float, target_ratio: float) -> float:
    """Uniform scale that brings the face area ratio from ``base_ratio`` to ``target_ratio``."""
    return float(np.sqrt(target_ratio / base_ratio))


def collage(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Two portraits side by side at a common height."""
    height = min(left.shape[0], right.shape[0])

    def fit(image: np.ndarray) -> np.ndarray:
        scale = height / image.shape[0]
        return cv2.resize(image, (round(image.shape[1] * scale), height))

    return np.hstack([fit(left), fit(right)])
