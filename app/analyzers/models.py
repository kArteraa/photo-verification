"""Model files required by the MediaPipe backend and their download."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from app.core.errors import ModelMissingError

log = logging.getLogger(__name__)

MODEL_BASE_URL = "https://storage.googleapis.com/mediapipe-models"
MODEL_URLS = {
    "face_landmarker.task": (
        f"{MODEL_BASE_URL}/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "selfie_segmenter.tflite": (
        f"{MODEL_BASE_URL}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    ),
}
DOWNLOAD_HINT = "run: photocheck models download"


def model_path(models_dir: Path, name: str) -> Path:
    """Path of a model file, raising :class:`ModelMissingError` when it is absent."""
    path = Path(models_dir) / name
    if not path.is_file() or path.stat().st_size == 0:
        raise ModelMissingError(f"model file missing: {path} ({DOWNLOAD_HINT})")
    return path


def models_available(models_dir: Path) -> bool:
    """True when every required model file is present."""
    try:
        for name in MODEL_URLS:
            model_path(models_dir, name)
    except ModelMissingError:
        return False
    return True


def download_models(models_dir: Path, force: bool) -> list[Path]:
    """Fetch the model files into ``models_dir`` and return their paths."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, url in MODEL_URLS.items():
        target = models_dir / name
        if target.is_file() and target.stat().st_size > 0 and not force:
            log.info("model already present: %s", target)
        else:
            log.info("downloading %s", url)
            partial = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(url, partial)
            if partial.stat().st_size == 0:
                partial.unlink()
                raise ModelMissingError(f"empty download: {url}")
            partial.replace(target)
            log.info("saved %s (%d bytes)", target, target.stat().st_size)
        paths.append(target)
    return paths
