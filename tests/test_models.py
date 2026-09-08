import pytest

from app.analyzers import models
from app.core.errors import ModelMissingError


def fake_urlretrieve(payload):
    def retrieve(url, target):
        target.write_bytes(payload)

    return retrieve


def test_model_path_requires_non_empty_file(tmp_path):
    with pytest.raises(ModelMissingError, match="photocheck models download"):
        models.model_path(tmp_path, "face_landmarker.task")
    (tmp_path / "face_landmarker.task").write_bytes(b"")
    with pytest.raises(ModelMissingError):
        models.model_path(tmp_path, "face_landmarker.task")
    (tmp_path / "face_landmarker.task").write_bytes(b"x")
    assert models.model_path(tmp_path, "face_landmarker.task").name == "face_landmarker.task"


def test_download_and_skip_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(models.urllib.request, "urlretrieve", fake_urlretrieve(b"model"))
    paths = models.download_models(tmp_path / "models", force=False)
    assert [p.name for p in paths] == list(models.MODEL_URLS)
    assert all(p.read_bytes() == b"model" for p in paths)
    assert models.models_available(tmp_path / "models")
    monkeypatch.setattr(models.urllib.request, "urlretrieve", fake_urlretrieve(b"new"))
    models.download_models(tmp_path / "models", force=False)
    assert paths[0].read_bytes() == b"model"
    models.download_models(tmp_path / "models", force=True)
    assert paths[0].read_bytes() == b"new"


def test_empty_download_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(models.urllib.request, "urlretrieve", fake_urlretrieve(b""))
    with pytest.raises(ModelMissingError, match="empty download"):
        models.download_models(tmp_path, force=True)
    assert not models.models_available(tmp_path)
    assert not list(tmp_path.glob("*.part"))
