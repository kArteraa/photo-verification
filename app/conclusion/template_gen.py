"""Deterministic Russian conclusion rendered from a report without any model.

The text follows the machine-checkable layout shared with the LLM generator:
a header line with the outcome, a body where every failed or undefined
requirement is tagged as ``[requirement_id]``, and footer lines listing the
hard violations, the soft recommendations and the unchecked requirements.
"""

from __future__ import annotations

from app.core.report import Kind, Report, Status, Verdict

HEADER = "ИТОГ"
ACCEPTED_WORD = "ПРИНЯТО"
REJECTED_WORD = "ОТКЛОНЕНО"
VIOLATIONS_LABEL = "НАРУШЕНИЯ"
RECOMMENDATIONS_LABEL = "РЕКОМЕНДАЦИИ"
UNCHECKED_LABEL = "НЕ ПРОВЕРЕНО"
NONE_WORD = "нет"


def render_conclusion(report: Report) -> str:
    """Build the conclusion text for a report."""
    hard = report.failing(Kind.HARD)
    soft = report.failing(Kind.SOFT)
    undefined = report.undefined()
    passed = report.with_status(Status.PASS)
    lines = [f"{HEADER}: {ACCEPTED_WORD if report.accepted else REJECTED_WORD}", _summary(report)]
    if hard:
        lines += ["", "Обязательные требования не выполнены:"]
        lines += [_violation_line(verdict) for verdict in hard]
    if undefined:
        lines += ["", "Не удалось проверить:"]
        lines += [_undefined_line(verdict) for verdict in undefined]
    if soft:
        lines += ["", "Можно улучшить:"]
        lines += [_violation_line(verdict) for verdict in soft]
    if passed:
        lines += ["", "Выполнено: " + ", ".join(verdict.title.lower() for verdict in passed) + "."]
    lines += [
        "",
        _footer(VIOLATIONS_LABEL, hard),
        _footer(RECOMMENDATIONS_LABEL, soft),
        _footer(UNCHECKED_LABEL, undefined),
    ]
    return "\n".join(lines)


def _summary(report: Report) -> str:
    hard_count = len(report.failing(Kind.HARD))
    undefined_count = len(report.undefined())
    if report.accepted:
        return "Фотография соответствует обязательным требованиям сценария."
    parts = []
    if hard_count:
        parts.append(f"нарушено обязательных требований: {hard_count}")
    if undefined_count:
        parts.append(f"не удалось проверить: {undefined_count}")
    return "Фотография отклонена: " + "; ".join(parts) + "."


def _violation_line(verdict: Verdict) -> str:
    return f"— [{verdict.id}] {verdict.title}: {verdict.reason}. Рекомендация: {verdict.fix_hint}."


def _undefined_line(verdict: Verdict) -> str:
    if verdict.blocked_by is not None:
        detail = f"проверка невозможна, пока не выполнено требование [{verdict.blocked_by}]"
    else:
        detail = verdict.reason or "измерение недоступно"
    return f"— [{verdict.id}] {verdict.title}: {detail}."


def _footer(label: str, verdicts: tuple[Verdict, ...]) -> str:
    ids = ", ".join(verdict.id for verdict in verdicts) if verdicts else NONE_WORD
    return f"{label}: {ids}"
