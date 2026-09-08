from pathlib import Path

import pytest

from app.analyzers import REGISTRY
from app.core.report import Kind
from app.core.spec import load_spec, topological_order

SPECS_DIR = Path(__file__).resolve().parent.parent / "app" / "specs"
SPEC_FILES = sorted(SPECS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", SPEC_FILES, ids=[p.stem for p in SPEC_FILES])
def test_spec_loads_and_orders(path):
    spec = load_spec(path, REGISTRY)
    assert spec.name == path.stem
    ordered = topological_order(spec)
    placed = set()
    for requirement in ordered:
        assert set(requirement.depends_on) <= placed
        placed.add(requirement.id)
    assert len(ordered) == len(spec.requirements)


def test_three_scenarios_exist():
    assert [p.stem for p in SPEC_FILES] == ["aesthetic", "avatar", "document_photo"]


def test_aesthetic_is_advisory_only():
    spec = load_spec(SPECS_DIR / "aesthetic.yaml", REGISTRY)
    assert all(requirement.kind is Kind.SOFT for requirement in spec.requirements)


def test_avatar_is_looser_than_document_photo():
    document = load_spec(SPECS_DIR / "document_photo.yaml", REGISTRY)
    avatar = load_spec(SPECS_DIR / "avatar.yaml", REGISTRY)
    assert (
        avatar.by_id("frontal_pose").predicate.values[0]
        > document.by_id("frontal_pose").predicate.values[0]
    )
    assert (
        avatar.by_id("face_size").predicate.values[1]
        > document.by_id("face_size").predicate.values[1]
    )
    assert (
        avatar.by_id("min_resolution").predicate.values[0]
        < document.by_id("min_resolution").predicate.values[0]
    )
    assert len(document.requirements) == 10
