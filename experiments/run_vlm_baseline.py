"""Zero-shot VLM baseline on a stratified sample of the test split."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.analyzers import REGISTRY
from app.core.spec import Spec, load_spec
from app.imageio import read_bgr
from app.llm.client import ChatClient
from app.llm.factory import build_client
from app.llm.mock_client import MockChatClient
from experiments.claims import score_claims
from experiments.common import (
    DATA_GENERATED,
    DOCUMENT_SPEC,
    LLM_CACHE,
    LLM_MODEL,
    RESULTS,
    RESULTS_MOCK,
    SEED,
    SPLIT_TEST,
    configure_logging,
    load_labels,
)
from experiments.evaluate import CLAIMED_SEPARATOR, status_column
from experiments.metrics import all_requirement_metrics, error_rates
from experiments.run_eval import claim_rows, hard_ids
from experiments.vlm import encode_image, image_digest, mock_oracle, parse_answer, vlm_request

log = logging.getLogger("run_vlm_baseline")

VLM_STATUS_PREFIX = "vlm_"


def stratified_sample(labels: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Proportional sample over classes with at least one image per class."""
    if limit <= 0 or limit >= len(labels):
        return labels
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in labels.groupby("cls", sort=True):
        count = max(1, round(limit * len(group) / len(labels)))
        picks = rng.choice(len(group), size=min(count, len(group)), replace=False)
        parts.append(group.iloc[sorted(picks)])
    return pd.concat(parts).head(limit)


def evaluate_rows(
    labels: pd.DataFrame, images_dir: Path, spec: Spec, client: ChatClient
) -> pd.DataFrame:
    """Query the model for every sampled image."""
    rows = []
    for index, (_, label) in enumerate(labels.iterrows(), start=1):
        image_jpeg = encode_image(read_bgr(images_dir / label["file"]))
        response = client.complete(vlm_request(spec, image_jpeg))
        answer = parse_answer(response, spec)
        row = {
            "file": label["file"],
            "cls": label["cls"],
            "expected_fail_id": label["expected_fail_id"],
            "backend": response.backend,
            "accepted": answer["accepted"],
            "failed": CLAIMED_SEPARATOR.join(sorted(answer["failed"])),
            "claimed": CLAIMED_SEPARATOR.join(sorted(answer["claimed"])),
        }
        for requirement_id, status in answer["statuses"].items():
            row[status_column(requirement_id)] = status
        rows.append(row)
        if index % 20 == 0:
            log.info("queried %d/%d", index, len(labels))
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, spec: Spec) -> dict:
    """Metrics comparable with the main system."""
    rows = claim_rows(frame, hard_ids(spec))
    rates = error_rates(frame)
    backends = sorted(set(frame["backend"]))
    return {
        "n_images": len(frame),
        "backends": backends,
        "is_mock": backends == ["mock"] or "mock" in backends,
        "per_requirement": [
            {k: v for k, v in asdict(item).items() if k != "auc"}
            for item in all_requirement_metrics(frame, spec)
        ],
        "far": rates.far,
        "frr": rates.frr,
        "per_class_far": rates.per_class_far,
        "claims": asdict(score_claims(rows)),
    }


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DATA_GENERATED / "labels.csv")
    parser.add_argument("--spec", type=Path, default=DOCUMENT_SPEC)
    parser.add_argument("--split", default=SPLIT_TEST)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--cache", type=Path, default=LLM_CACHE)
    parser.add_argument("--out", type=Path, default=None, help="default: results or results/mock")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    spec = load_spec(args.spec, REGISTRY)
    labels = load_labels(args.labels)
    labels = stratified_sample(labels[labels["split"] == args.split], args.limit, args.seed)
    images_dir = args.labels.parent
    truth = {
        image_digest(encode_image(read_bgr(images_dir / row["file"]))): row["expected_fail_id"]
        for _, row in labels.iterrows()
    }
    client = build_client(args.mock, LLM_MODEL, args.cache, mock_oracle(spec, truth, args.seed))
    is_mock = isinstance(client, MockChatClient)
    out = args.out or (RESULTS_MOCK if is_mock else RESULTS)
    if is_mock:
        log.warning("mock client in use; outputs go to %s and are not publishable", out)
    log.info("querying %d images", len(labels))
    frame = evaluate_rows(labels, images_dir, spec, client)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "vlm_predictions.csv", index=False, encoding="utf-8")
    summary = summarize(frame, spec)
    (out / "vlm_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(
        "backends=%s FAR=%.3f FRR=%.3f consistency=%.3f",
        summary["backends"],
        summary["far"],
        summary["frr"],
        summary["claims"]["consistency"],
    )


if __name__ == "__main__":
    main()
