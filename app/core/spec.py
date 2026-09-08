"""Requirement specification: schema, validation and dependency ordering.

A specification ``R = {r_1, ..., r_n}`` is loaded from YAML.  Each requirement
``r_i = (μ_i, π_i, τ_i, κ_i, D_i)`` maps to :class:`Requirement` with a
measurer, a :class:`Predicate` holding the operator and thresholds, a kind and
the dependency ids.  Parsing resolves measurers against a registry and rejects
every malformed field early, so the engine can assume a valid, acyclic graph.
"""

from __future__ import annotations

import string
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.analyzers.registry import MeasurerSpec, get_measurer
from app.core.errors import SpecError
from app.core.hints import hint_placeholders
from app.core.report import Kind, Number

OPS = ("eq", "ge", "le", "range", "abs_le")
REQUIRED_FIELDS = (
    "id",
    "title",
    "measurer",
    "predicate",
    "kind",
    "depends_on",
    "reason_template",
    "fix_hint",
)
TOP_LEVEL_FIELDS = ("name", "description", "requirements")
RANGE_PLACEHOLDERS = {"lo", "hi", "side", "delta"}


@dataclass(frozen=True)
class Predicate:
    """Decision predicate π_i: an operator applied to measurement keys with thresholds τ_i."""

    op: str
    keys: tuple[str, ...]
    values: tuple[Number, ...]

    @property
    def key(self) -> str:
        """The single key inspected by every operator except multi-key ``abs_le``."""
        return self.keys[0]


@dataclass(frozen=True)
class Requirement:
    """One requirement r_i of the specification."""

    id: str
    title: str
    measurer: MeasurerSpec
    predicate: Predicate
    kind: Kind
    depends_on: tuple[str, ...]
    reason_template: str
    fix_hint: str


@dataclass(frozen=True)
class Spec:
    """A named, validated set of requirements."""

    name: str
    description: str
    requirements: tuple[Requirement, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        """Requirement ids in declaration order."""
        return tuple(requirement.id for requirement in self.requirements)

    def by_id(self, requirement_id: str) -> Requirement:
        """Return the requirement with the given id."""
        for requirement in self.requirements:
            if requirement.id == requirement_id:
                return requirement
        raise KeyError(requirement_id)


def load_spec(path: Path, registry: Mapping[str, MeasurerSpec]) -> Spec:
    """Read a YAML file and parse it into a :class:`Spec`."""
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return parse_spec(data, registry)


def parse_spec(data: Any, registry: Mapping[str, MeasurerSpec]) -> Spec:
    """Validate raw specification data and resolve measurers against ``registry``."""
    if not isinstance(data, dict):
        raise SpecError("specification must be a mapping")
    for field in TOP_LEVEL_FIELDS:
        if field not in data:
            raise SpecError(f"missing top-level field: {field}")
    raw_requirements = data["requirements"]
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise SpecError("requirements must be a non-empty list")
    ids = _collect_ids(raw_requirements)
    requirements = tuple(_parse_requirement(item, registry, ids) for item in raw_requirements)
    spec = Spec(str(data["name"]), str(data["description"]), requirements)
    topological_order(spec)
    return spec


def topological_order(spec: Spec) -> tuple[Requirement, ...]:
    """Order requirements so that dependencies precede dependants.

    At every step the earliest YAML entry whose dependencies are already placed is taken,
    which keeps the result deterministic and close to the declaration order.
    """
    remaining = list(spec.requirements)
    done: list[Requirement] = []
    done_ids: set[str] = set()
    while remaining:
        ready = next(
            (req for req in remaining if all(dep in done_ids for dep in req.depends_on)),
            None,
        )
        if ready is None:
            raise SpecError(f"dependency cycle: {_describe_cycle(remaining)}")
        remaining.remove(ready)
        done.append(ready)
        done_ids.add(ready.id)
    return tuple(done)


def _describe_cycle(remaining: list[Requirement]) -> str:
    by_id = {req.id: req for req in remaining}
    path = [remaining[0].id]
    while True:
        current = by_id[path[-1]]
        following = next(dep for dep in current.depends_on if dep in by_id)
        if following in path:
            path.append(following)
            return " -> ".join(path[path.index(following) :])
        path.append(following)


def _collect_ids(raw_requirements: list[Any]) -> set[str]:
    ids: set[str] = set()
    for item in raw_requirements:
        if not isinstance(item, dict) or "id" not in item:
            raise SpecError("every requirement must be a mapping with an id")
        requirement_id = str(item["id"])
        if requirement_id in ids:
            raise SpecError(f"duplicate requirement id: {requirement_id}")
        ids.add(requirement_id)
    return ids


def _parse_requirement(
    item: dict[str, Any], registry: Mapping[str, MeasurerSpec], ids: set[str]
) -> Requirement:
    requirement_id = str(item["id"])
    for field in REQUIRED_FIELDS:
        if field not in item:
            raise SpecError(f"{requirement_id}: missing field {field}")
    measurer = get_measurer(registry, str(item["measurer"]))
    predicate = _parse_predicate(requirement_id, item["predicate"], measurer)
    kind = _parse_kind(requirement_id, item["kind"])
    depends_on = _parse_depends_on(requirement_id, item["depends_on"], ids)
    allowed = _allowed_placeholders(measurer, predicate)
    reason_template = _check_template(requirement_id, "reason_template", item, allowed)
    fix_hint = _check_template(requirement_id, "fix_hint", item, allowed)
    return Requirement(
        id=requirement_id,
        title=str(item["title"]),
        measurer=measurer,
        predicate=predicate,
        kind=kind,
        depends_on=depends_on,
        reason_template=reason_template,
        fix_hint=fix_hint,
    )


def _parse_kind(requirement_id: str, raw: Any) -> Kind:
    try:
        return Kind(raw)
    except ValueError as exc:
        raise SpecError(f"{requirement_id}: kind must be hard or soft, got {raw!r}") from exc


def _parse_depends_on(requirement_id: str, raw: Any, ids: set[str]) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise SpecError(f"{requirement_id}: depends_on must be a list")
    depends_on = tuple(str(dep) for dep in raw)
    for dep in depends_on:
        if dep == requirement_id:
            raise SpecError(f"{requirement_id}: depends on itself")
        if dep not in ids:
            raise SpecError(f"{requirement_id}: unknown dependency {dep}")
    return depends_on


def _parse_predicate(requirement_id: str, raw: Any, measurer: MeasurerSpec) -> Predicate:
    if not isinstance(raw, dict) or "op" not in raw or "value" not in raw:
        raise SpecError(f"{requirement_id}: predicate needs op and value")
    op = str(raw["op"])
    if op not in OPS:
        raise SpecError(f"{requirement_id}: unknown op {op}")
    if op == "abs_le" and "keys" in raw:
        keys = tuple(str(key) for key in raw["keys"])
        values = _numbers(requirement_id, raw["value"], len(keys))
    elif op == "range":
        keys = (_single_key(raw, measurer),)
        values = _numbers(requirement_id, raw["value"], 2)
        if values[0] >= values[1]:
            raise SpecError(f"{requirement_id}: range lower bound must be below upper bound")
    else:
        keys = (_single_key(raw, measurer),)
        values = (_number(requirement_id, raw["value"]),)
    for key in keys:
        if key not in measurer.keys:
            raise SpecError(f"{requirement_id}: key {key} not produced by {measurer.name}")
    return Predicate(op, keys, values)


def _single_key(raw: dict[str, Any], measurer: MeasurerSpec) -> str:
    return str(raw["key"]) if "key" in raw else measurer.primary_key


def _number(requirement_id: str, raw: Any) -> Number:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise SpecError(f"{requirement_id}: threshold must be a number, got {raw!r}")
    return raw


def _numbers(requirement_id: str, raw: Any, count: int) -> tuple[Number, ...]:
    if not isinstance(raw, list) or len(raw) != count:
        raise SpecError(f"{requirement_id}: expected a list of {count} numbers, got {raw!r}")
    return tuple(_number(requirement_id, value) for value in raw)


def _allowed_placeholders(measurer: MeasurerSpec, predicate: Predicate) -> set[str]:
    allowed = set(measurer.keys) | {"value"} | hint_placeholders(measurer.keys)
    if predicate.op == "range":
        return allowed | RANGE_PLACEHOLDERS
    allowed.add("tau")
    allowed.update(f"tau_{key}" for key in predicate.keys)
    return allowed


def _check_template(
    requirement_id: str, field: str, item: dict[str, Any], allowed: set[str]
) -> str:
    template = str(item[field])
    try:
        fields = [name for _, name, _, _ in string.Formatter().parse(template) if name]
    except ValueError as exc:
        raise SpecError(f"{requirement_id}: malformed {field}: {exc}") from exc
    for name in fields:
        if name not in allowed:
            raise SpecError(f"{requirement_id}: {field} uses unknown placeholder {{{name}}}")
    return template
