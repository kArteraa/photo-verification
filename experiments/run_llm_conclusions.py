"""Measure the language-model verbalization path on a sample of the test split.

For every sampled image the report is built by the rules engine and handed
to :func:`generate_conclusion`; the ``source`` of the returned conclusion
tells whether the model passed verification at the first attempt, after one
regeneration, or was replaced by the template.  Without an API key the mock
client answers and the outputs go to the mock results directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from app.analyzers import REGISTRY
from app.analyzers.context import FaceBackend
from app.conclusion.llm_gen import SOURCE_TEMPLATE, generate_conclusion, template_synthesizer
from app.conclusion.template_gen import RECOMMENDATIONS_LABEL, VIOLATIONS_LABEL
from app.conclusion.verify import extract_footer
from app.core.engine import Engine
from app.core.report import Status
from app.core.spec import Spec, load_spec
from app.llm.client import BACKEND_MOCK, ChatClient
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
    make_backend,
)
from experiments.evaluate import CLAIMED_SEPARATOR, evaluate_image
from experiments.run_eval import claim_rows, hard_ids
from experiments.run_vlm_baseline import stratified_sample

log = logging.getLogger("run_llm_conclusions")

RETRY_SUFFIX = "_retry"
TEXTS_DIR = "llm_conclusions"


def conclusion_row(
    label: pd.Series, images_dir: Path, engine: Engine, backend: FaceBackend, client: ChatClient
) -> tuple[dict, str]:
    """Generate one conclusion and flatten its outcome into a table row."""
    report, _, _ = evaluate_image(images_dir / label["file"], engine, backend)
    conclusion = generate_conclusion(report, client)
    claimed: set[str] = set()
    for footer in (VIOLATIONS_LABEL, RECOMMENDATIONS_LABEL):
        claimed |= extract_footer(conclusion.text, footer) or set()
    row = {
        "file": label["file"],
        "cls": label["cls"],
        "expected_fail_id": label["expected_fail_id"],
        "source": conclusion.source,
        "backend": conclusion.source.replace(RETRY_SUFFIX, ""),
        "verified_ok": conclusion.verification.ok,
        "claimed": CLAIMED_SEPARATOR.join(sorted(claimed)),
        "failed": CLAIMED_SEPARATOR.join(
            verdict.id for verdict in report.verdicts if verdict.status is Status.FAIL
        ),
    }
    return row, conclusion.text


def summarize_conclusions(frame: pd.DataFrame, spec: Spec) -> dict:
    """Shares of the verification paths and claim quality of the generated texts."""
    sources = frame["source"].astype(str)
    fallback = sources == SOURCE_TEMPLATE
    retry = sources.str.endswith(RETRY_SUFFIX)
    first_try = ~fallback & ~retry
    backends = sorted(set(frame["backend"].astype(str)))
    return {
        "n": len(frame),
        "backends": backends,
        "is_mock": BACKEND_MOCK in backends,
        "first_try_rate": float(first_try.mean()),
        "retry_rate": float(retry.mean()),
        "fallback_rate": float(fallback.mean()),
        "verified_rate": float(frame["verified_ok"].astype(bool).mean()),
        "claims": asdict(score_claims(claim_rows(frame, hard_ids(spec)))),
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
    client = build_client(args.mock, LLM_MODEL, args.cache, template_synthesizer)
    is_mock = isinstance(client, MockChatClient)
    out = args.out or (RESULTS_MOCK if is_mock else RESULTS)
    if is_mock:
        log.warning("mock client in use; outputs go to %s and are not publishable", out)
    engine, backend = Engine(spec), make_backend()
    texts_dir = out / TEXTS_DIR
    texts_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (_, label) in enumerate(labels.iterrows(), start=1):
        row, text = conclusion_row(label, args.labels.parent, engine, backend, client)
        rows.append(row)
        (texts_dir / f"{Path(label['file']).stem}.txt").write_text(text, encoding="utf-8")
        if index % 20 == 0:
            log.info("generated %d/%d", index, len(labels))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "llm_conclusions.csv", index=False, encoding="utf-8")
    summary = summarize_conclusions(frame, spec)
    (out / "llm_conclusions_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(
        "backends=%s first_try=%.3f retry=%.3f fallback=%.3f verified=%.3f",
        summary["backends"],
        summary["first_try_rate"],
        summary["retry_rate"],
        summary["fallback_rate"],
        summary["verified_rate"],
    )


if __name__ == "__main__":
    main()
