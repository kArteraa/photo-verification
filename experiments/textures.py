"""Procedural busy backgrounds used in place of interior photographs."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

Texture = Callable[[int, int, np.random.Generator], np.ndarray]


def random_color(rng: np.random.Generator) -> np.ndarray:
    """Random BGR colour with enough contrast to be visible."""
    return rng.integers(20, 236, size=3).astype(np.uint8)


def checkerboard(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Two-colour checkerboard with a random cell size."""
    cell = int(rng.integers(16, 64))
    tiles = (np.indices((height, width)).sum(axis=0) // cell) % 2
    return np.where(tiles[:, :, None] == 1, random_color(rng), random_color(rng)).astype(np.uint8)


def stripes(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Diagonal stripes with a random period and angle."""
    period = int(rng.integers(12, 48))
    angle = rng.uniform(0, np.pi)
    ys, xs = np.indices((height, width))
    phase = ((xs * np.cos(angle) + ys * np.sin(angle)) // period) % 2
    return np.where(phase[:, :, None] == 1, random_color(rng), random_color(rng)).astype(np.uint8)


def smooth_noise(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency colour noise upsampled from a coarse grid."""
    scale = int(rng.integers(4, 16))
    coarse = rng.integers(0, 256, size=(scale, scale, 3)).astype(np.uint8)
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)


def gradient(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Linear gradient between two colours at a random angle."""
    a, b = random_color(rng).astype(float), random_color(rng).astype(float)
    angle = rng.uniform(0, 2 * np.pi)
    ys, xs = np.indices((height, width))
    t = (xs * np.cos(angle) + ys * np.sin(angle)).astype(float)
    t = (t - t.min()) / max(t.max() - t.min(), 1.0)
    return (a + (b - a) * t[:, :, None]).astype(np.uint8)


def blobs(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Random filled circles over a flat colour."""
    canvas = np.full((height, width, 3), random_color(rng), dtype=np.uint8)
    for _ in range(int(rng.integers(15, 40))):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        radius = int(rng.integers(8, max(9, min(height, width) // 6)))
        cv2.circle(canvas, center, radius, random_color(rng).tolist(), thickness=-1)
    return canvas


def bricks(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Brick wall pattern with mortar lines."""
    brick_h, brick_w = int(rng.integers(16, 40)), int(rng.integers(40, 96))
    mortar = random_color(rng)
    canvas = np.full((height, width, 3), mortar, dtype=np.uint8)
    for row, y in enumerate(range(0, height, brick_h)):
        offset = (brick_w // 2) if row % 2 else 0
        for x in range(-offset, width, brick_w):
            color = random_color(rng).tolist()
            cv2.rectangle(canvas, (x + 2, y + 2), (x + brick_w - 2, y + brick_h - 2), color, -1)
    return canvas


def grid(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Thin grid lines over a flat colour."""
    canvas = np.full((height, width, 3), random_color(rng), dtype=np.uint8)
    step = int(rng.integers(20, 60))
    color = random_color(rng).tolist()
    for x in range(0, width, step):
        cv2.line(canvas, (x, 0), (x, height), color, 2)
    for y in range(0, height, step):
        cv2.line(canvas, (0, y), (width, y), color, 2)
    return canvas


def speckle(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """High-frequency pixel noise around a mean colour."""
    base = random_color(rng).astype(float)
    noise = rng.normal(0, 60, size=(height, width, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


TEXTURES: dict[str, Texture] = {
    "checkerboard": checkerboard,
    "stripes": stripes,
    "smooth_noise": smooth_noise,
    "gradient": gradient,
    "blobs": blobs,
    "bricks": bricks,
    "grid": grid,
    "speckle": speckle,
}


def random_texture(height: int, width: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """A texture chosen uniformly among the generators, with its name."""
    name = str(rng.choice(list(TEXTURES)))
    return TEXTURES[name](height, width, rng), name
