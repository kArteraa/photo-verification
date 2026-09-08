import pytest

from app.conclusion.template_gen import render_conclusion
from app.conclusion.verify import (
    compare_claims,
    extract_footer,
    extract_ids,
    extract_numbers,
    number_matches,
    verify_conclusion,
)
from tests.test_template_gen import rejected_report


def conclusion():
    return render_conclusion(rejected_report())


def test_extract_ids_only_known():
    text = "нарушено [frontal_pose] и [unknown_id], см. [eyes_open]"
    assert extract_ids(text, {"frontal_pose", "eyes_open"}) == {"frontal_pose", "eyes_open"}


def test_extract_footer():
    text = "НАРУШЕНИЯ: a, b\nРЕКОМЕНДАЦИИ: нет\n"
    assert extract_footer(text, "НАРУШЕНИЯ") == {"a", "b"}
    assert extract_footer(text, "РЕКОМЕНДАЦИИ") == set()
    assert extract_footer(text, "НЕ ПРОВЕРЕНО") is None


def test_extract_numbers_skips_tags_footers_and_identifiers():
    text = (
        "ИТОГ: ОТКЛОНЕНО\nyaw=17.3°, pitch=-2,1° [face_size2] img_001.jpg ±10° 0.25\nНАРУШЕНИЯ: a1"
    )
    assert extract_numbers(text) == ["17.3", "-2,1", "10", "0.25"]


@pytest.mark.parametrize(
    ("token", "expected"),
    [("17.3", True), ("17,3", True), ("17", True), ("17.30", True), ("17.4", False), ("18", False)],
)
def test_number_matches_with_rounding_tolerance(token, expected):
    assert number_matches(token, {17.3}) is expected


def test_template_passes():
    result = verify_conclusion(conclusion(), rejected_report())
    assert result.ok
    assert result.header_matches
    assert not result.invented and not result.missing and not result.foreign_numbers


def test_invented_violation_is_caught():
    text = conclusion().replace("Выполнено: single face.", "Нарушено также [single_face].")
    result = verify_conclusion(text, rejected_report())
    assert not result.ok
    assert result.invented == frozenset({"single_face"})
    assert any("did not fail" in error for error in result.errors)


def test_missing_violation_is_caught():
    text = "\n".join(line for line in conclusion().splitlines() if "[frontal_pose]" not in line)
    result = verify_conclusion(text, rejected_report())
    assert result.missing == frozenset({"frontal_pose"})
    assert not result.ok


def test_verdict_counts_are_allowed_but_wrong_counts_are_not():
    text = conclusion()
    assert "нарушено обязательных требований: 1; не удалось проверить: 2." in text
    assert verify_conclusion(text, rejected_report()).ok
    wrong = text.replace("не удалось проверить: 2.", "не удалось проверить: 9.")
    assert verify_conclusion(wrong, rejected_report()).foreign_numbers == ("9",)


def test_foreign_number_is_caught():
    text = conclusion().replace("yaw=17.3°", "yaw=42.0°")
    result = verify_conclusion(text, rejected_report())
    assert result.foreign_numbers == ("42.0",)
    assert not result.ok


def test_header_mismatch_is_caught():
    text = conclusion().replace("ИТОГ: ОТКЛОНЕНО", "ИТОГ: ПРИНЯТО")
    result = verify_conclusion(text, rejected_report())
    assert not result.header_matches
    assert not result.ok


def test_footer_mismatch_is_caught():
    text = conclusion().replace("РЕКОМЕНДАЦИИ: headroom", "РЕКОМЕНДАЦИИ: нет")
    result = verify_conclusion(text, rejected_report())
    assert any("footer РЕКОМЕНДАЦИИ" in error for error in result.errors)
    missing_line = "\n".join(
        line for line in conclusion().splitlines() if not line.startswith("НЕ ПРОВЕРЕНО")
    )
    result = verify_conclusion(missing_line, rejected_report())
    assert any("missing footer line" in error for error in result.errors)


def test_compare_claims():
    score = compare_claims({"a", "b", "x"}, {"a", "b", "c"})
    assert (score.correct, score.missed, score.invented) == (2, 1, 1)
