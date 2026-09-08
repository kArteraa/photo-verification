"""Registry of measurers μ_i addressed by dotted names from the specification."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.errors import UnknownMeasurerError
from app.core.report import Measurement

if TYPE_CHECKING:
    from app.analyzers.context import AnalysisContext

MeasureFn = Callable[["AnalysisContext"], Measurement]


@dataclass(frozen=True)
class MeasurerSpec:
    """A registered measurer: its name, the keys it produces and the function."""

    name: str
    keys: tuple[str, ...]
    fn: MeasureFn

    @property
    def primary_key(self) -> str:
        """Key used by single-key predicates that do not name one explicitly."""
        return self.keys[0]


REGISTRY: dict[str, MeasurerSpec] = {}


def register(name: str, keys: tuple[str, ...]) -> Callable[[MeasureFn], MeasureFn]:
    """Register the decorated function as measurer ``name`` producing ``keys``."""
    if not keys:
        raise ValueError(f"measurer {name} must declare at least one key")

    def decorator(fn: MeasureFn) -> MeasureFn:
        if name in REGISTRY:
            raise ValueError(f"measurer already registered: {name}")
        REGISTRY[name] = MeasurerSpec(name, keys, fn)
        return fn

    return decorator


def get_measurer(registry: Mapping[str, MeasurerSpec], name: str) -> MeasurerSpec:
    """Look up a measurer, raising :class:`UnknownMeasurerError` when absent."""
    try:
        return registry[name]
    except KeyError as exc:
        raise UnknownMeasurerError(f"unknown measurer: {name}") from exc
