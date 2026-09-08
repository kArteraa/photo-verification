import numpy as np
import pytest

from app.core.errors import ImageReadError
from app.imageio import read_bgr, write_bgr


def test_round_trip_with_cyrillic_path(tmp_path):
    folder = tmp_path / "фото"
    folder.mkdir()
    path = folder / "портрет.png"
    image = np.zeros((8, 6, 3), dtype=np.uint8)
    image[2, 3] = (10, 20, 30)
    write_bgr(path, image, quality=95)
    restored = read_bgr(path)
    assert restored.shape == (8, 6, 3)
    assert restored[2, 3].tolist() == [10, 20, 30]


def test_jpeg_quality_is_applied(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    low, high = tmp_path / "low.jpg", tmp_path / "high.jpg"
    write_bgr(low, image, quality=10)
    write_bgr(high, image, quality=95)
    assert low.stat().st_size < high.stat().st_size


def test_missing_and_undecodable_files(tmp_path):
    with pytest.raises(ImageReadError, match="cannot read image"):
        read_bgr(tmp_path / "nope.jpg")
    garbage = tmp_path / "garbage.jpg"
    garbage.write_bytes(b"not an image")
    with pytest.raises(ImageReadError, match="cannot decode image"):
        read_bgr(garbage)
