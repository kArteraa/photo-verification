"""Evaluate the system on the generated dataset and write predictions and metrics."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from app.analyzers import REGISTRY
from app.core.engine import Engine
from app.core.report import Kind
from app.core.spec import Spec, load_spec
from experiments.claims import ClaimRow, score_claims
from experiments.common import (
    DATA_GENERATED,
    DOCUMENT_SPEC,
    RESULTS,
    SPLIT_TEST,
    configure_logging,
    load_labels,
    make_backend,
)
from experiments.evaluate import evaluate_dataset, requirement_ids, split_ids
from experiments.metrics import (
    all_requirement_metrics,
    cross_trigger,
    error_rates,
    metrics_table,
    timing,
)

log = logging.getLogger("run_eval")


def hard_ids(spec: Spec) -> frozenset[str]:
    """Ids of hard requirements, the only ones ground-truth labels refer to."""
    return frozenset(r.id for r in spec.requirements if r.kind is Kind.HARD)


def claim_rows(frame: pd.DataFrame, hard: frozenset[str]) -> list[ClaimRow]:
    """Hard-violation claims of the conclusions against labels and own verdicts."""
    return [
        ClaimRow(
            file=row["file"],
            claimed=frozenset(split_ids(row["claimed"]) & hard),
            truth=frozenset({row["expected_fail_id"]} if row["expected_fail_id"] else set()),
            own_fails=frozenset(split_ids(row["failed"]) & hard),
        )
        for _, row in frame.iterrows()
    ]


def summarize(frame: pd.DataFrame, spec: Spec) -> dict:
    """All metrics of one prediction table."""
    per_requirement = all_requirement_metrics(frame, spec)
    rates = error_rates(frame)
    return {
        "n_images": len(frame),
        "per_requirement": [asdict(item) for item in per_requirement],
        "far": rates.far,
        "frr": rates.frr,
        "per_class_far": rates.per_class_far,
        "timing": timing(frame),
        "conclusion_verified_rate": float(frame["conclusion_ok"].mean()),
        "claims": asdict(score_claims(claim_rows(frame, hard_ids(spec)))),
        "cross_trigger": cross_trigger(frame, requirement_ids(frame.columns)).to_dict(),
    }


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DATA_GENERATED / "labels.csv")
    parser.add_argument("--spec", type=Path, default=DOCUMENT_SPEC)
    parser.add_argument("--split", default=SPLIT_TEST)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    labels = load_labels(args.labels)
    labels = labels[labels["split"] == args.split]
    if args.limit:
        labels = labels.head(args.limit)
    spec = load_spec(args.spec, REGISTRY)
    engine = Engine(spec)
    log.info("evaluating %d images of split %s with spec %s", len(labels), args.split, spec.name)
    frame = evaluate_dataset(labels, args.labels.parent, engine, make_backend())
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "raw_predictions.csv", index=False, encoding="utf-8")
    summary = summarize(frame, spec)
    (args.out / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    table = metrics_table(all_requirement_metrics(frame, spec))
    table.to_csv(args.out / "table2.csv", index=False, encoding="utf-8")
    log.info(
        "FAR=%.3f FRR=%.3f median=%.0f ms verified=%.3f",
        summary["far"],
        summary["frr"],
        summary["timing"]["median_ms"],
        summary["conclusion_verified_rate"],
    )
    log.info("\n%s", table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
