"""Figure 1: architecture of the facts, rules and verbalization pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BOX_STYLE = "round,pad=0.02,rounding_size=0.02"
STAGES = (
    ("Изображение x", 0.085),
    ("Анализаторы μ_i\n(лицо, поза, глаза,\nрезкость, экспозиция,\nфон, геометрия)", 0.251),
    ("Измерения m\n(числа и флаги\nприменимости)", 0.417),
    ("Движок правил\nспецификация R (YAML),\nграф зависимостей,\nвердикты {0, 1, ⊥}", 0.583),
    ("Отчёт (v, m)\nJSON", 0.749),
    ("Заключение e\nшаблон или LLM\n+ автоматическая сверка", 0.915),
)
BOX_WIDTH = 0.15
BOX_HEIGHT = 0.5
SHADED = {1, 3, 5}


def draw_architecture(path: Path, dpi: int = 300) -> Path:
    """Render the pipeline diagram as a grayscale-friendly JPEG."""
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for index, (label, x) in enumerate(STAGES):
        face = "#d9d9d9" if index in SHADED else "white"
        ax.add_patch(
            FancyBboxPatch(
                (x - BOX_WIDTH / 2, 0.5 - BOX_HEIGHT / 2),
                BOX_WIDTH,
                BOX_HEIGHT,
                boxstyle=BOX_STYLE,
                linewidth=1.2,
                edgecolor="black",
                facecolor=face,
            )
        )
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=8.5)
        if index:
            previous = STAGES[index - 1][1]
            ax.add_patch(
                FancyArrowPatch(
                    (previous + BOX_WIDTH / 2, 0.5),
                    (x - BOX_WIDTH / 2, 0.5),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=1.2,
                    color="black",
                )
            )
    ax.text(
        0.5,
        0.06,
        "Языковая модель получает только отчёт (v, m); изображение ей недоступно",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pil_kwargs={"quality": 95})
    plt.close(fig)
    return path
