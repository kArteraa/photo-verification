"""Calibrate requirement thresholds τ_i by ROC analysis on the calibration split.

For every violation class with a dedicated requirement the raw measurement is
turned into a one-sided score (larger means more violating), a threshold is
chosen on the calibration images by maximum F1 or by a bounded false
rejection rate, and the value is written back into the YAML specification.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve

from app.analyzers import REGISTRY
from app.core.engine import Engine
from app.core.spec import Spec, load_spec
from experiments.common import (
    CLASS_TO_REQUIREMENT,
    CLEAN,
    DATA_GENERATED,
    DOCUMENT_SPEC,
    RESULTS,
    SPLIT_CALIB,
    configure_logging,
    load_labels,
    make_backend,
)
from experiments.evaluate import evaluate_dataset, measure_column

log = logging.getLogger("calibrate")

MODE_F1 = "f1"
MODE_FRR = "frr"
SIDE_LOW = "lo"
SIDE_HIGH = "hi"
CALIBRATION_SIDES = {"dark": SIDE_LOW, "bright": SIDE_HIGH, "face_small": SIDE_LOW}
SELF_SELECTED = ("non_frontal",)
PERCENTILES = (95, 99)
DECIMALS = 4


@dataclass(frozen=True)
class ThresholdChoice:
    """Chosen score cut and its quality on the calibration split."""

    score: float
    f1: float
    frr: float
    tpr: float


def score_sign(op: str, side: str | None) -> float:
    """Sign that turns a raw value into a score growing with violation."""
    if op == "ge" or (op == "range" and side == SIDE_LOW):
        return -1.0
    return 1.0


def choose_threshold(
    positives: np.ndarray, negatives: np.ndarray, mode: str, max_frr: float
) -> ThresholdChoice:
    """Pick the score cut: an image fails when its score exceeds the cut."""
    values = np.sort(np.unique(np.concatenate([positives, negatives])))
    candidates = np.concatenate(
        [[values[0] - 1.0], (values[:-1] + values[1:]) / 2.0, [values[-1] + 1.0]]
    )
    best: ThresholdChoice | None = None
    for cut in candidates:
        tp = float((positives > cut).sum())
        fp = float((negatives > cut).sum())
        fn = float(len(positives) - tp)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        frr = fp / len(negatives) if len(negatives) else 0.0
        tpr = tp / len(positives) if len(positives) else 0.0
        choice = ThresholdChoice(float(cut), f1, frr, tpr)
        if mode == MODE_FRR and frr > max_frr:
            continue
        if best is None or _better(choice, best, mode):
            best = choice
    if best is None:
        cut = float(negatives.max() + 1e-6)
        best = ThresholdChoice(cut, 0.0, 0.0, float((positives > cut).mean()))
    return best


def _better(candidate: ThresholdChoice, best: ThresholdChoice, mode: str) -> bool:
    if mode == MODE_F1:
        return candidate.f1 > best.f1
    return candidate.tpr > best.tpr


def calibrate_requirement(
    frame: pd.DataFrame, spec: Spec, cls: str, mode: str, max_frr: float
) -> dict:
    """Threshold for the requirement targeted by a violation class."""
    requirement = spec.by_id(CLASS_TO_REQUIREMENT[cls])
    predicate = requirement.predicate
    side = CALIBRATION_SIDES.get(cls)
    column = measure_column(requirement.measurer.name, predicate.key)
    sign = score_sign(predicate.op, side)
    subset = frame[frame["cls"].isin((cls, CLEAN))].dropna(subset=[column])
    scores = subset[column].astype(float).to_numpy() * sign
    is_positive = (subset["cls"] == cls).to_numpy()
    choice = choose_threshold(scores[is_positive], scores[~is_positive], mode, max_frr)
    fpr, tpr, _ = roc_curve(is_positive.astype(int), scores)
    old_value = predicate.values[0 if side != SIDE_HIGH else 1]
    return {
        "requirement": requirement.id,
        "cls": cls,
        "key": predicate.key,
        "op": predicate.op,
        "side": side,
        "old_value": old_value,
        "new_value": round(choice.score * sign, DECIMALS),
        "f1": choice.f1,
        "frr": choice.frr,
        "tpr": choice.tpr,
        "n_pos": int(is_positive.sum()),
        "n_neg": int((~is_positive).sum()),
        "roc": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
        "chosen_point": {"fpr": choice.frr, "tpr": choice.tpr},
    }


def clean_percentiles(frame: pd.DataFrame, spec: Spec) -> dict:
    """Reference percentiles of every predicate key on clean images."""
    clean = frame[frame["cls"] == CLEAN]
    result: dict[str, dict[str, float]] = {}
    for requirement in spec.requirements:
        for key in requirement.predicate.keys:
            column = measure_column(requirement.measurer.name, key)
            if column not in clean:
                continue
            values = clean[column].dropna().astype(float).abs()
            if len(values):
                result[f"{requirement.id}.{key}"] = {
                    f"p{p}": round(float(np.percentile(values, p)), DECIMALS) for p in PERCENTILES
                }
    return result


def write_thresholds(spec_path: Path, updates: list[dict]) -> str:
    """Rewrite only the predicate lines of calibrated requirements, keeping the file layout."""
    text = spec_path.read_text(encoding="utf-8")
    for update in updates:
        text = _replace_predicate(text, update)
    return text


def _replace_predicate(text: str, update: dict) -> str:
    pattern = re.compile(
        rf"(- id: {re.escape(update['requirement'])}\n(?:(?!- id:).*\n)*?\s*predicate: )(\{{.*\}})"
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"predicate line not found for {update['requirement']}")
    predicate = yaml.safe_load(match.group(2))
    if update["op"] == "range":
        index = 0 if update["side"] == SIDE_LOW else 1
        predicate["value"][index] = update["new_value"]
    else:
        predicate["value"] = update["new_value"]
    rendered = yaml.safe_dump(
        predicate, default_flow_style=True, sort_keys=False, width=1000
    ).strip()
    return text[: match.start(2)] + rendered + text[match.end(2) :]


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DATA_GENERATED / "labels.csv")
    parser.add_argument("--spec", type=Path, default=DOCUMENT_SPEC)
    parser.add_argument("--mode", choices=(MODE_F1, MODE_FRR), default=MODE_F1)
    parser.add_argument("--max-frr", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None, help="spec to write (default: --spec)")
    parser.add_argument("--results", type=Path, default=RESULTS / "calibration.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    labels = load_labels(args.labels)
    labels = labels[labels["split"] == SPLIT_CALIB]
    if args.limit:
        labels = labels.head(args.limit)
    spec = load_spec(args.spec, REGISTRY)
    log.info("calibrating on %d images", len(labels))
    frame = evaluate_dataset(labels, args.labels.parent, Engine(spec), make_backend())
    present = set(frame["cls"])
    updates = [
        calibrate_requirement(frame, spec, cls, args.mode, args.max_frr)
        for cls in CLASS_TO_REQUIREMENT
        if cls in present and cls not in SELF_SELECTED and CLASS_TO_REQUIREMENT[cls] in spec.ids
    ]
    for update in updates:
        log.info(
            "%s (%s): %s -> %s, F1=%.3f FRR=%.3f",
            update["requirement"],
            update["cls"],
            update["old_value"],
            update["new_value"],
            update["f1"],
            update["frr"],
        )
    out = args.out or args.spec
    out.write_text(write_thresholds(args.spec, updates), encoding="utf-8")
    load_spec(out, REGISTRY)
    payload = {
        "mode": args.mode,
        "max_frr": args.max_frr,
        "n_images": len(frame),
        "self_selected_unchanged": list(SELF_SELECTED),
        "updates": updates,
        "clean_percentiles": clean_percentiles(frame, spec),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("spec written to %s, details in %s", out, args.results)


if __name__ == "__main__":
    main()
