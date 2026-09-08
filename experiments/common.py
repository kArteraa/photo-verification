"""Constants and helpers shared by the experiment scripts."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.analyzers.context import FaceBackend
from app.analyzers.mediapipe_backend import MediaPipeBackend

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_VALID = ROOT / "data" / "valid"
DATA_GENERATED = ROOT / "data" / "generated"
RESULTS = ROOT / "results"
RESULTS_MOCK = RESULTS / "mock"
FIGURES = RESULTS / "figures"
MODELS_DIR = ROOT / "models"
SPECS_DIR = ROOT / "app" / "specs"
DOCUMENT_SPEC = SPECS_DIR / "document_photo.yaml"
LLM_MODEL = "claude-sonnet-4-6"
LLM_CACHE = RESULTS / "llm_cache"

CLEAN = "clean"
CLASSES = (
    CLEAN,
    "blur_face",
    "dark",
    "bright",
    "busy_background",
    "face_small",
    "face_shifted",
    "multi_face",
    "non_frontal",
    "eyes_closed",
)
CLASS_TO_REQUIREMENT = {
    "blur_face": "face_sharpness",
    "dark": "exposure_ok",
    "bright": "exposure_ok",
    "busy_background": "background_uniform",
    "face_small": "face_size",
    "face_shifted": "face_centered",
    "multi_face": "single_face",
    "non_frontal": "frontal_pose",
    "eyes_closed": "eyes_open",
}
LABEL_COLUMNS = ("file", "source", "cls", "param", "split", "expected_fail_id")
SPLIT_CALIB = "calib"
SPLIT_TEST = "test"


def configure_logging(verbose: bool) -> None:
    """Root logging setup shared by the scripts."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def make_backend(models_dir: Path = MODELS_DIR) -> FaceBackend:
    """MediaPipe backend with the CLI defaults."""
    return MediaPipeBackend(models_dir, num_faces=5, min_detection_confidence=0.5)


def rng(seed: int = SEED) -> np.random.Generator:
    """Seeded random generator."""
    return np.random.default_rng(seed)


def load_labels(path: Path) -> pd.DataFrame:
    """Read labels.csv keeping empty strings for missing expected ids."""
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def image_files(folder: Path) -> list[Path]:
    """Sorted image files in a folder."""
    return sorted(
        p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
