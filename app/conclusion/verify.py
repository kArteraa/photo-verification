"""Automatic check of a conclusion text against the report it must describe.

The conclusion is consistent when every requirement it tags is actually
failed or undefined in the report, every failed requirement is mentioned, the
footer lists match the verdicts exactly, the header matches the acceptance
flag and every number in the text occurs in the report (measured values,
thresholds or the numbers inside generated reasons).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass

from app.conclusion.template_gen import (
    ACCEPTED_WORD,
    HEADER,
    NONE_WORD,
    RECOMMENDATIONS_LABEL,
    REJECTED_WORD,
    UNCHECKED_LABEL,
    VIOLATIONS_LABEL,
)
from app.core.report import Kind, Report, Status

TAG_PATTERN = re.compile(r"\[([a-z][a-z0-9_]*)\]")
NUMBER_PATTERN = re.compile(r"(?<![\w\[\-.,])[-+]?\d+(?:[.,]\d+)?(?![\w\]])")
FOOTER_LABELS = (VIOLATIONS_LABEL, RECOMMENDATIONS_LABEL, UNCHECKED_LABEL)


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of checking a conclusion text against a report."""

    ok: bool
    header_matches: bool
    invented: frozenset[str]
    missing: frozenset[str]
    foreign_numbers: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ClaimScore:
    """Agreement between a set of claimed violations and the true ones."""

    correct: int
    missed: int
    invented: int


def extract_ids(text: str, known_ids: Collection[str]) -> set[str]:
    """Requirement ids tagged as ``[id]`` anywhere in the text."""
    return {match for match in TAG_PATTERN.findall(text) if match in known_ids}


def extract_footer(text: str, label: str) -> set[str] | None:
    """Ids listed after ``label:`` in the footer, or None when the line is absent."""
    for line in text.splitlines():
        if line.startswith(f"{label}:"):
            payload = line.split(":", 1)[1].strip()
            if payload in ("", NONE_WORD):
                return set()
            return {item.strip() for item in payload.split(",") if item.strip()}
    return None


def extract_numbers(text: str) -> list[str]:
    """Number literals in the text, with either decimal separator."""
    body = "\n".join(
        line for line in text.splitlines() if not line.startswith(tuple(FOOTER_LABELS))
    )
    body = TAG_PATTERN.sub("", body)
    return NUMBER_PATTERN.findall(body)


def allowed_numbers(report: Report) -> set[float]:
    """Every number the conclusion may legitimately quote."""
    allowed: set[float] = set()
    for verdict in report.verdicts:
        allowed.update(_flatten(verdict.measured or {}))
        allowed.update(_flatten(verdict.threshold or {}))
        for text in (verdict.reason, verdict.fix_hint):
            if text:
                allowed.update(_parse(token) for token in NUMBER_PATTERN.findall(text))
    for values in report.measurements.values():
        allowed.update(_flatten(values))
    return allowed


def number_matches(token: str, allowed: Iterable[float]) -> bool:
    """True when the token equals some allowed number after rounding to its own precision."""
    value = _parse(token)
    tolerance = 0.5 * 10 ** (-_decimals(token)) + 1e-9
    return any(abs(value - candidate) <= tolerance for candidate in allowed)


def verify_conclusion(text: str, report: Report) -> VerificationResult:
    """Check the conclusion text against the report."""
    known = {verdict.id for verdict in report.verdicts}
    failed = {verdict.id for verdict in report.with_status(Status.FAIL)}
    undefined = {verdict.id for verdict in report.undefined()}
    mentioned = extract_ids(text, known)
    invented = frozenset(mentioned - failed - undefined)
    missing = frozenset(failed - mentioned)
    errors: list[str] = []
    header_matches = _header_matches(text, report.accepted)
    if not header_matches:
        errors.append("header does not match the acceptance flag")
    if invented:
        errors.append("tags requirements that did not fail: " + ", ".join(sorted(invented)))
    if missing:
        errors.append("does not mention failed requirements: " + ", ".join(sorted(missing)))
    errors += _footer_errors(text, report)
    allowed = allowed_numbers(report)
    foreign = tuple(token for token in extract_numbers(text) if not number_matches(token, allowed))
    if foreign:
        errors.append("quotes numbers absent from the report: " + ", ".join(foreign))
    return VerificationResult(
        ok=not errors,
        header_matches=header_matches,
        invented=invented,
        missing=missing,
        foreign_numbers=foreign,
        errors=tuple(errors),
    )


def compare_claims(claimed: Collection[str], truth: Collection[str]) -> ClaimScore:
    """Count correct, missed and invented claims relative to the ground truth."""
    claimed_set, truth_set = set(claimed), set(truth)
    return ClaimScore(
        correct=len(claimed_set & truth_set),
        missed=len(truth_set - claimed_set),
        invented=len(claimed_set - truth_set),
    )


def _header_matches(text: str, accepted: bool) -> bool:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    expected = f"{HEADER}: {ACCEPTED_WORD if accepted else REJECTED_WORD}"
    return first == expected


def _footer_errors(text: str, report: Report) -> list[str]:
    expected = {
        VIOLATIONS_LABEL: {verdict.id for verdict in report.failing(Kind.HARD)},
        RECOMMENDATIONS_LABEL: {verdict.id for verdict in report.failing(Kind.SOFT)},
        UNCHECKED_LABEL: {verdict.id for verdict in report.undefined()},
    }
    errors = []
    for label, ids in expected.items():
        listed = extract_footer(text, label)
        if listed is None:
            errors.append(f"missing footer line {label}")
        elif listed != ids:
            errors.append(f"footer {label} lists {sorted(listed)} instead of {sorted(ids)}")
    return errors


def _flatten(values: dict) -> list[float]:
    flat: list[float] = []
    for value in values.values():
        if isinstance(value, list | tuple):
            flat.extend(float(item) for item in value)
        elif isinstance(value, int | float) and not isinstance(value, bool):
            flat.append(float(value))
    return flat


def _parse(token: str) -> float:
    return float(token.replace(",", "."))


def _decimals(token: str) -> int:
    digits = token.replace(",", ".")
    return len(digits.split(".")[1]) if "." in digits else 0
