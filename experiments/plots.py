"""Figures 2 to 4 and the summary table for the paper."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import FIGURES, RESULTS, configure_logging
from experiments.plots_arch import draw_architecture

log = logging.getLogger("plots")

DPI = 300
ROC_REQUIREMENTS = ("face_sharpness", "background_uniform", "exposure_ok", "face_size")
OURS_LABEL = "предложенный метод"
VLM_LABEL = "VLM zero-shot"
MOCK_SUFFIX = " (mock)"
GRAY_DARK = "#404040"
GRAY_LIGHT = "#b0b0b0"


def load_json(path: Path) -> dict | None:
    """Read a JSON file when it exists."""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def vlm_label(vlm: dict | None) -> str:
    """Legend label of the baseline, marked when its numbers come from the mock."""
    if vlm is None:
        return VLM_LABEL
    return VLM_LABEL + (MOCK_SUFFIX if vlm.get("is_mock") else "")


def save(fig: plt.Figure, path: Path) -> None:
    """Save a figure as a 300 dpi JPEG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pil_kwargs={"quality": 95})
    plt.close(fig)
    log.info("saved %s", path)


def plot_roc(calibration: dict, path: Path) -> None:
    """Figure 2: ROC curves of the key measurers with the calibrated thresholds."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    styles = ("-", "--", "-.", ":")
    updates = {item["requirement"]: item for item in calibration["updates"]}
    for style, requirement in zip(styles, ROC_REQUIREMENTS, strict=False):
        item = updates.get(requirement)
        if item is None:
            continue
        label = f"{requirement} ({item['cls']})"
        ax.plot(item["roc"]["fpr"], item["roc"]["tpr"], style, color="black", label=label)
        point = item["chosen_point"]
        ax.plot(point["fpr"], point["tpr"], "o", color="black", markersize=6)
    ax.plot([0, 1], [0, 1], color=GRAY_LIGHT, linewidth=0.8)
    ax.set_xlabel("Доля ложных отклонений (FPR)")
    ax.set_ylabel("Доля обнаруженных нарушений (TPR)")
    ax.set_title("ROC-кривые измерителей, точки — калиброванные пороги τ")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linewidth=0.4, color=GRAY_LIGHT)
    save(fig, path)


def f1_frame(ours: dict, vlm: dict | None) -> pd.DataFrame:
    """Per-requirement F1 of both methods, indexed by requirement and class."""
    rows = {
        (item["requirement"], item["cls"]): {"ours": item["f1"]} for item in ours["per_requirement"]
    }
    if vlm is not None:
        for item in vlm["per_requirement"]:
            rows.setdefault((item["requirement"], item["cls"]), {})["vlm"] = item["f1"]
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = [f"{req}\n({cls})" for req, cls in frame.index]
    return frame


def plot_f1_bars(ours: dict, vlm: dict | None, path: Path) -> None:
    """Figure 3: grouped F1 bars per requirement."""
    frame = f1_frame(ours, vlm)
    positions = np.arange(len(frame))
    width = 0.38 if "vlm" in frame else 0.6
    fig, ax = plt.subplots(figsize=(max(6, 1.35 * len(frame)), 4.6))
    ax.bar(
        positions - (width / 2 if "vlm" in frame else 0),
        frame["ours"],
        width,
        color=GRAY_DARK,
        label=OURS_LABEL,
    )
    if "vlm" in frame:
        ax.bar(
            positions + width / 2,
            frame["vlm"].fillna(0),
            width,
            color=GRAY_LIGHT,
            edgecolor="black",
            label=vlm_label(vlm),
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(frame.index, fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 по требованию")
    ax.set_title("Качество обнаружения нарушений по требованиям")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="y", linewidth=0.4, color=GRAY_LIGHT)
    save(fig, path)


def plot_consistency(ours: dict, vlm: dict | None, path: Path) -> None:
    """Figure 4: consistency, completeness and invented claims of the conclusions."""
    metrics = ("consistency", "completeness", "invented_rate")
    labels = (
        "консистентность\nс вердиктами",
        "полнота\n(нарушение упомянуто)",
        "доля заключений\nс лишними утверждениями",
    )
    ours_values = [ours["claims"][metric] for metric in metrics]
    positions = np.arange(len(metrics))
    width = 0.38 if vlm else 0.6
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(
        positions - (width / 2 if vlm else 0), ours_values, width, color=GRAY_DARK, label=OURS_LABEL
    )
    if vlm is not None:
        vlm_values = [vlm["claims"][metric] for metric in metrics]
        ax.bar(
            positions + width / 2,
            vlm_values,
            width,
            color=GRAY_LIGHT,
            edgecolor="black",
            label=vlm_label(vlm),
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Доля изображений")
    ax.set_title("Свойства заключений")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", linewidth=0.4, color=GRAY_LIGHT)
    save(fig, path)


def summary_table(ours: dict, vlm: dict | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Table 2: per-requirement metrics of both methods and the integral metrics."""
    rows = []
    vlm_items = {
        (item["requirement"], item["cls"]): item for item in (vlm or {}).get("per_requirement", [])
    }
    for item in ours["per_requirement"]:
        row = {
            "requirement": item["requirement"],
            "cls": item["cls"],
            "precision": item["precision"],
            "recall": item["recall"],
            "f1": item["f1"],
            "auc": item["auc"],
            "undefined_rate": item["undefined_rate"],
            "n_pos": item["n_pos"],
            "n_neg": item["n_neg"],
        }
        baseline = vlm_items.get((item["requirement"], item["cls"]))
        if baseline is not None:
            row.update({f"vlm_{k}": baseline[k] for k in ("precision", "recall", "f1")})
        rows.append(row)
    integral = {
        "far": ours["far"],
        "frr": ours["frr"],
        "median_ms": ours["timing"]["median_ms"],
        "consistency": ours["claims"]["consistency"],
        "completeness": ours["claims"]["completeness"],
    }
    if vlm is not None:
        integral.update(
            {
                "vlm_far": vlm["far"],
                "vlm_frr": vlm["frr"],
                "vlm_consistency": vlm["claims"]["consistency"],
                "vlm_completeness": vlm["claims"]["completeness"],
                "vlm_backends": ",".join(vlm["backends"]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame([integral])


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    ours = load_json(args.results / "metrics.json")
    if ours is None:
        raise SystemExit("results/metrics.json is missing; run run_eval.py first")
    vlm = load_json(args.results / "vlm_metrics.json")
    calibration = load_json(args.results / "calibration.json")
    draw_architecture(args.figures / "fig1_architecture.jpg", DPI)
    if calibration is not None:
        plot_roc(calibration, args.figures / "fig2_roc.jpg")
    else:
        log.warning("calibration.json is missing; figure 2 skipped")
    plot_f1_bars(ours, vlm, args.figures / "fig3_f1_bars.jpg")
    plot_consistency(ours, vlm, args.figures / "fig4_consistency.jpg")
    table, integral = summary_table(ours, vlm)
    table.to_csv(args.results / "table2.csv", index=False, encoding="utf-8")
    integral.to_csv(args.results / "table2_integral.csv", index=False, encoding="utf-8")
    log.info("\n%s", table.round(3).to_string(index=False))
    log.info("\n%s", integral.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
