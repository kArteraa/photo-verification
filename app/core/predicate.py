"""Evaluation of decision predicates π_i and helpers derived from thresholds τ_i."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.core.report import Number
from app.core.spec import Predicate

EQ_TOLERANCE = 1e-9


def evaluate(predicate: Predicate, values: Mapping[str, Number]) -> bool:
    """Return True when the measured values satisfy the predicate."""
    if predicate.op == "abs_le":
        return all(
            abs(values[key]) <= limit
            for key, limit in zip(predicate.keys, predicate.values, strict=True)
        )
    measured = values[predicate.key]
    if predicate.op == "eq":
        return abs(measured - predicate.values[0]) < EQ_TOLERANCE
    if predicate.op == "ge":
        return measured >= predicate.values[0]
    if predicate.op == "le":
        return measured <= predicate.values[0]
    low, high = predicate.values
    return low <= measured <= high


def threshold_dict(predicate: Predicate) -> dict[str, Any]:
    """Thresholds keyed by measurement key, as written into the report."""
    if predicate.op == "range":
        return {predicate.key: list(predicate.values)}
    return dict(zip(predicate.keys, predicate.values, strict=True))


def format_context(predicate: Predicate, values: Mapping[str, Number]) -> dict[str, Any]:
    """Placeholders available to ``reason_template`` and ``fix_hint``."""
    context: dict[str, Any] = dict(values)
    context["value"] = values[predicate.key]
    if predicate.op == "range":
        low, high = predicate.values
        measured = values[predicate.key]
        context["lo"] = low
        context["hi"] = high
        context["side"] = "ниже" if measured < low else "выше"
        context["delta"] = max(low - measured, measured - high)
        return context
    context["tau"] = predicate.values[0]
    for key, limit in zip(predicate.keys, predicate.values, strict=True):
        context[f"tau_{key}"] = limit
    return context


def violation_score(predicate: Predicate, values: Mapping[str, Number]) -> float:
    """Scalar that grows monotonically with the degree of violation, for ROC analysis."""
    if predicate.op == "abs_le":
        return max(abs(values[key]) for key in predicate.keys)
    measured = values[predicate.key]
    if predicate.op == "eq":
        return abs(measured - predicate.values[0])
    if predicate.op == "ge":
        return -measured
    if predicate.op == "le":
        return measured
    low, high = predicate.values
    return max(low - measured, measured - high)
