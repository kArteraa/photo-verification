from app.conclusion.template_gen import render_conclusion
from app.conclusion.verify import verify_conclusion
from app.core.report import Kind, Report, Status, Verdict


def verdict(requirement_id, status, kind=Kind.HARD, **fields):
    base = {
        "measured": None,
        "threshold": None,
        "reason": None,
        "fix_hint": None,
        "blocked_by": None,
    }
    base.update(fields)
    return Verdict(
        requirement_id, requirement_id.replace("_", " ").capitalize(), status, kind, **base
    )


def rejected_report():
    return Report(
        image="img.jpg",
        spec="document_photo",
        accepted=False,
        verdicts=(
            verdict("single_face", Status.PASS, measured={"count": 1}),
            verdict(
                "frontal_pose",
                Status.FAIL,
                measured={"yaw": 17.3, "pitch": 2.1},
                threshold={"yaw": 10.0, "pitch": 10.0},
                reason="поворот головы yaw=17.3°, pitch=2.1° при допуске ±10°",
                fix_hint="смотрите прямо в камеру",
            ),
            verdict("eyes_open", Status.UNDEFINED, blocked_by="frontal_pose"),
            verdict("background_uniform", Status.UNDEFINED, reason="фон не виден"),
            verdict(
                "headroom",
                Status.FAIL,
                Kind.SOFT,
                measured={"headroom_ratio": 0.01},
                threshold={"headroom_ratio": [0.04, 0.25]},
                reason="отступ над головой 0.01 ниже рекомендуемого [0.04, 0.25]",
                fix_hint="оставьте место над головой",
            ),
        ),
        measurements={"pose.yaw_pitch": {"yaw": 17.3, "pitch": 2.1, "roll": -0.4}},
    )


def accepted_report():
    return Report(
        "img.jpg",
        "document_photo",
        True,
        (verdict("single_face", Status.PASS, measured={"count": 1}),),
        {},
    )


def test_rejected_conclusion_structure():
    text = render_conclusion(rejected_report())
    lines = text.splitlines()
    assert lines[0] == "ИТОГ: ОТКЛОНЕНО"
    assert "нарушено обязательных требований: 1" in lines[1]
    assert "не удалось проверить: 2" in lines[1]
    assert "— [frontal_pose] Frontal pose: поворот головы yaw=17.3°" in text
    assert "Рекомендация: смотрите прямо в камеру." in text
    assert (
        "— [eyes_open] Eyes open: проверка невозможна, пока не выполнено требование [frontal_pose]."
        in text
    )
    assert "— [background_uniform] Background uniform: фон не виден." in text
    assert "Можно улучшить:" in text
    assert "— [headroom] Headroom: отступ над головой 0.01" in text
    assert "Выполнено: single face." in text
    assert "НАРУШЕНИЯ: frontal_pose" in lines
    assert "РЕКОМЕНДАЦИИ: headroom" in lines
    assert "НЕ ПРОВЕРЕНО: eyes_open, background_uniform" in lines


def test_accepted_conclusion():
    text = render_conclusion(accepted_report())
    lines = text.splitlines()
    assert lines[0] == "ИТОГ: ПРИНЯТО"
    assert "соответствует обязательным требованиям" in lines[1]
    assert "Обязательные требования не выполнены" not in text
    assert "НАРУШЕНИЯ: нет" in lines
    assert "РЕКОМЕНДАЦИИ: нет" in lines
    assert "НЕ ПРОВЕРЕНО: нет" in lines


def test_template_output_passes_verification():
    for report in (rejected_report(), accepted_report()):
        result = verify_conclusion(render_conclusion(report), report)
        assert result.ok, result.errors


def test_rendering_is_deterministic():
    assert render_conclusion(rejected_report()) == render_conclusion(rejected_report())
