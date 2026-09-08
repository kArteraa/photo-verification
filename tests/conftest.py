from pathlib import Path

import pytest

from app.analyzers.models import models_available

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
MODELS_DIR = ROOT / "models"
DATA_VALID = ROOT / "data" / "valid"


def pytest_collection_modifyitems(config, items):
    skip_mediapipe = pytest.mark.skip(reason="MediaPipe models not downloaded")
    skip_dataset = pytest.mark.skip(reason="data/valid is empty")
    has_models = models_available(MODELS_DIR)
    has_dataset = DATA_VALID.is_dir() and any(DATA_VALID.glob("*.jpg"))
    for item in items:
        if "mediapipe" in item.keywords and not has_models:
            item.add_marker(skip_mediapipe)
        if "dataset" in item.keywords and not (has_models and has_dataset):
            item.add_marker(skip_dataset)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return MODELS_DIR


@pytest.fixture(scope="session")
def backend(models_dir):
    from app.analyzers.mediapipe_backend import MediaPipeBackend

    return MediaPipeBackend(models_dir, num_faces=5, min_detection_confidence=0.5)


@pytest.fixture(scope="session")
def data_valid_dir() -> Path:
    return DATA_VALID
