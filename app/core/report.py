"""Report data structures shared by all pipeline stages.

A :class:`Report` is the machine-readable contract between the rules engine
and the conclusion generators: the verdict vector ``v``, the measurement
vector ``m`` and the acceptance flag.  Floats are rounded at serialization
time so that the conclusion generator and the verifier see identical numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

FLOAT_DIGITS = 4

Number = int | float
Values = dict[str, Number]


class Status(StrEnum):
    """Outcome of a decision predicate: 1, 0 or ⊥."""

    PASS = "pass"
    FAIL = "fail"
    UNDEFINED = "undefined"


class Kind(StrEnum):
    """Requirement kind κ_i."""

    HARD = "hard"
    SOFT = "soft"


def round_values(values: Values) -> Values:
    """Round float values to the serialization precision, leaving integers intact."""
    return {
        key: round(value, FLOAT_DIGITS) if isinstance(value, float) else value
        for key, value in values.items()
    }


@dataclass(frozen=True)
class Measurement:
    """Output of a measurer μ_i: named numbers plus an applicability flag."""

    measurer: str
    values: Values
    applicable: bool
    note: str | None

    @classmethod
    def of(cls, measurer: str, values: Values) -> Measurement:
        """Build an applicable measurement."""
        return cls(measurer, values, True, None)

    @classmethod
    def not_applicable(cls, measurer: str, note: str) -> Measurement:
        """Build a measurement that could not be taken, with the reason."""
        return cls(measurer, {}, False, note)


@dataclass(frozen=True)
class Verdict:
    """Decision on a single requirement together with its evidence."""

    id: str
    title: str
    status: Status
    kind: Kind
    measured: Values | None
    threshold: dict[str, Any] | None
    reason: str | None
    fix_hint: str | None
    blocked_by: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON contract, omitting absent fields."""
        data: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "kind": self.kind.value,
        }
        if self.measured is not None:
            data["measured"] = round_values(self.measured)
        for name in ("threshold", "reason", "fix_hint", "blocked_by"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Verdict:
        """Deserialize from the JSON contract."""
        return cls(
            id=data["id"],
            title=data["title"],
            status=Status(data["status"]),
            kind=Kind(data["kind"]),
            measured=data.get("measured"),
            threshold=data.get("threshold"),
            reason=data.get("reason"),
            fix_hint=data.get("fix_hint"),
            blocked_by=data.get("blocked_by"),
        )


@dataclass(frozen=True)
class Report:
    """Structured result of checking one image against one specification.

    ``measurements`` maps measurer names to their raw values (the vector ``m``);
    it is optional in serialized form because the published contract omits it.
    """

    image: str
    spec: str
    accepted: bool
    verdicts: tuple[Verdict, ...]
    measurements: dict[str, Values]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON contract."""
        return {
            "image": self.image,
            "spec": self.spec,
            "accepted": self.accepted,
            "verdicts": [verdict.to_dict() for verdict in self.verdicts],
            "measurements": {
                name: round_values(values) for name, values in self.measurements.items()
            },
        }

    def to_json(self) -> str:
        """Serialize to a JSON string with Cyrillic kept readable."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save(self, path: Path) -> None:
        """Write the JSON report to ``path``."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Report:
        """Deserialize from the JSON contract."""
        return cls(
            image=data["image"],
            spec=data["spec"],
            accepted=data["accepted"],
            verdicts=tuple(Verdict.from_dict(item) for item in data["verdicts"]),
            measurements=data.get("measurements", {}),
        )

    @classmethod
    def load(cls, path: Path) -> Report:
        """Read a JSON report from ``path``."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def by_id(self, requirement_id: str) -> Verdict:
        """Return the verdict for a requirement id."""
        for verdict in self.verdicts:
            if verdict.id == requirement_id:
                return verdict
        raise KeyError(requirement_id)

    def with_status(self, status: Status) -> tuple[Verdict, ...]:
        """Return verdicts having the given status."""
        return tuple(verdict for verdict in self.verdicts if verdict.status is status)

    def failing(self, kind: Kind) -> tuple[Verdict, ...]:
        """Return failed verdicts of the given kind."""
        return tuple(verdict for verdict in self.with_status(Status.FAIL) if verdict.kind is kind)

    def undefined(self) -> tuple[Verdict, ...]:
        """Return verdicts that could not be evaluated."""
        return self.with_status(Status.UNDEFINED)
