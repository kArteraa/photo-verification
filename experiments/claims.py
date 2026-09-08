"""Consistency and completeness of conclusions relative to verdicts and ground truth."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ClaimRow:
    """Claims of one conclusion: what it says, what is true, what the system itself decided."""

    file: str
    claimed: frozenset[str]
    truth: frozenset[str]
    own_fails: frozenset[str]


@dataclass(frozen=True)
class ClaimSummary:
    """Aggregate claim quality over many conclusions."""

    n: int
    consistency: float
    completeness: float
    invented_rate: float
    mentioned: int
    missed: int
    invented: int


def score_claims(rows: Iterable[ClaimRow]) -> ClaimSummary:
    """Aggregate consistency with own verdicts and agreement with ground truth."""
    items = list(rows)
    if not items:
        return ClaimSummary(0, float("nan"), float("nan"), float("nan"), 0, 0, 0)
    consistent = sum(row.claimed <= row.own_fails for row in items)
    complete = sum(row.truth <= row.claimed for row in items)
    with_invented = sum(bool(row.claimed - row.truth) for row in items)
    mentioned = sum(len(row.claimed & row.truth) for row in items)
    missed = sum(len(row.truth - row.claimed) for row in items)
    invented = sum(len(row.claimed - row.truth) for row in items)
    return ClaimSummary(
        n=len(items),
        consistency=consistent / len(items),
        completeness=complete / len(items),
        invented_rate=with_invented / len(items),
        mentioned=mentioned,
        missed=missed,
        invented=invented,
    )
