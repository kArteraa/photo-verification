"""Rules engine: applies a specification to an image context and produces a report.

Requirements are evaluated in topological order.  A requirement whose
dependency did not pass receives ⊥ with ``blocked_by`` set and its measurer is
not invoked; a requirement whose measurer is not applicable receives ⊥ with the
measurer's note.  The image is accepted only when every hard requirement
passed, so ⊥ on a hard requirement is a rejection by construction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from app.analyzers.context import AnalysisContext
from app.analyzers.registry import MeasurerSpec
from app.core.hints import hint_context
from app.core.predicate import evaluate, format_context, threshold_dict
from app.core.report import Kind, Measurement, Report, Status, Values, Verdict
from app.core.spec import Requirement, Spec, topological_order

Measure = Callable[[MeasurerSpec], Measurement]


class Engine:
    """Evaluates one specification against analysis contexts."""

    def __init__(self, spec: Spec) -> None:
        self._spec = spec
        self._order = topological_order(spec)

    @property
    def spec(self) -> Spec:
        """The specification this engine applies."""
        return self._spec

    @property
    def measurers(self) -> tuple[MeasurerSpec, ...]:
        """Distinct measurers referenced by the specification, in first-use order."""
        seen: dict[str, MeasurerSpec] = {}
        for requirement in self._spec.requirements:
            seen.setdefault(requirement.measurer.name, requirement.measurer)
        return tuple(seen.values())

    def evaluate(self, ctx: AnalysisContext, image_name: str) -> Report:
        """Measure lazily (only what unblocked requirements need) and build the report."""
        cache: dict[str, Measurement] = {}

        def measure(measurer: MeasurerSpec) -> Measurement:
            if measurer.name not in cache:
                cache[measurer.name] = measurer.fn(ctx)
            return cache[measurer.name]

        return self._build_report(image_name, measure)

    def measure_all(self, ctx: AnalysisContext) -> dict[str, Measurement]:
        """Run every measurer of the specification once, regardless of dependencies."""
        return {measurer.name: measurer.fn(ctx) for measurer in self.measurers}

    def evaluate_measured(self, image_name: str, measurements: Mapping[str, Measurement]) -> Report:
        """Build the report from measurements computed in advance."""
        return self._build_report(image_name, lambda measurer: measurements[measurer.name])

    def _build_report(self, image_name: str, measure: Measure) -> Report:
        verdicts: dict[str, Verdict] = {}
        used: dict[str, Measurement] = {}
        for requirement in self._order:
            verdicts[requirement.id] = self._verdict(requirement, measure, verdicts, used)
        ordered = tuple(verdicts[requirement.id] for requirement in self._spec.requirements)
        accepted = all(
            verdict.status is Status.PASS for verdict in ordered if verdict.kind is Kind.HARD
        )
        measurements = {
            name: dict(measurement.values)
            for name, measurement in used.items()
            if measurement.applicable
        }
        return Report(image_name, self._spec.name, accepted, ordered, measurements)

    def _verdict(
        self,
        requirement: Requirement,
        measure: Measure,
        verdicts: Mapping[str, Verdict],
        used: dict[str, Measurement],
    ) -> Verdict:
        blocker = next(
            (dep for dep in requirement.depends_on if verdicts[dep].status is not Status.PASS),
            None,
        )
        if blocker is not None:
            return _undefined(requirement, blocked_by=blocker, reason=None)
        measurement = measure(requirement.measurer)
        used[requirement.measurer.name] = measurement
        if not measurement.applicable:
            return _undefined(requirement, blocked_by=None, reason=measurement.note)
        values = measurement.values
        measured = {key: values[key] for key in requirement.predicate.keys}
        if evaluate(requirement.predicate, values):
            return _passed(requirement, measured)
        return _failed(requirement, measured, values)


def _passed(requirement: Requirement, measured: Values) -> Verdict:
    return Verdict(
        id=requirement.id,
        title=requirement.title,
        status=Status.PASS,
        kind=requirement.kind,
        measured=measured,
        threshold=None,
        reason=None,
        fix_hint=None,
        blocked_by=None,
    )


def _failed(requirement: Requirement, measured: Values, values: Values) -> Verdict:
    context = {**format_context(requirement.predicate, values), **hint_context(values)}
    return Verdict(
        id=requirement.id,
        title=requirement.title,
        status=Status.FAIL,
        kind=requirement.kind,
        measured=measured,
        threshold=threshold_dict(requirement.predicate),
        reason=requirement.reason_template.format(**context),
        fix_hint=requirement.fix_hint.format(**context),
        blocked_by=None,
    )


def _undefined(requirement: Requirement, blocked_by: str | None, reason: str | None) -> Verdict:
    return Verdict(
        id=requirement.id,
        title=requirement.title,
        status=Status.UNDEFINED,
        kind=requirement.kind,
        measured=None,
        threshold=None,
        reason=reason,
        fix_hint=None,
        blocked_by=blocked_by,
    )
