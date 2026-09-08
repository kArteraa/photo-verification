"""Image reading and writing that tolerate non-ASCII paths on Windows."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.errors import ImageReadError


def read_bgr(path: Path) -> np.ndarray:
    """Decode an image file into a BGR array."""
    path = Path(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ImageReadError(f"cannot read image: {path}") from exc
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageReadError(f"cannot decode image: {path}")
    return image


def write_bgr(path: Path, image: np.ndarray, quality: int) -> None:
    """Encode a BGR array by the file extension and write it, JPEG at ``quality``."""
    path = Path(path)
    ok, buffer = cv2.imencode(path.suffix, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ImageReadError(f"cannot encode image: {path}")
    buffer.tofile(path)
