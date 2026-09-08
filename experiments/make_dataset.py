"""Generate a labelled violation dataset from a folder of valid portraits.

Every frontal source yields one file per augmentation class plus a clean
copy; sources whose measured yaw exceeds the threshold become the
``non_frontal`` class as they are.  Backgrounds of all synthetic variants are
normalized to a flat canvas first, so the clean class is genuinely clean and
the busy background class differs from it by the background alone.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.analyzers import REGISTRY
from app.analyzers.context import AnalysisContext, FaceBackend
from app.analyzers.geometry import face_area_ratio
from app.imageio import read_bgr, write_bgr
from experiments import augment
from experiments.common import (
    CLASS_TO_REQUIREMENT,
    CLEAN,
    DATA_GENERATED,
    DATA_VALID,
    LABEL_COLUMNS,
    RESULTS,
    SEED,
    SPLIT_CALIB,
    SPLIT_TEST,
    configure_logging,
    image_files,
    make_backend,
)
from experiments.textures import random_texture

log = logging.getLogger("make_dataset")

JPEG_QUALITY = 95
MAX_SMALL_FACE_SCALE = 0.95
MANUAL_CHECK_SAMPLE = 30
MANUAL_EXCLUDE = "exclude"
MANUAL_CLASSES = ("non_frontal", "eyes_closed", MANUAL_EXCLUDE)


@dataclass(frozen=True)
class Source:
    """A usable source portrait with its base measurements."""

    path: Path
    image: np.ndarray
    person: np.ndarray
    yaw: float
    face_ratio: float


def scan_sources(paths: list[Path], backend: FaceBackend) -> tuple[list[Source], list[str]]:
    """Measure every candidate and keep single-face portraits with a solved pose."""
    sources, skipped = [], []
    for index, path in enumerate(paths, start=1):
        ctx = AnalysisContext(read_bgr(path), backend)
        pose = REGISTRY["pose.yaw_pitch"].fn(ctx)
        if len(ctx.faces) != 1 or not pose.applicable:
            skipped.append(path.name)
            continue
        ratio = face_area_ratio(ctx.primary_face.bbox, ctx.width, ctx.height)
        sources.append(Source(path, ctx.bgr, ctx.person_confidence, pose.values["yaw"], ratio))
        if index % 100 == 0:
            log.info("scanned %d/%d", index, len(paths))
    return sources, skipped


def load_manual_labels(path: Path | None) -> dict[str, str]:
    """Optional manual overrides: source file name to class or exclude."""
    if path is None:
        return {}
    table = pd.read_csv(path, dtype=str)
    labels = dict(zip(table["source"], table["cls"], strict=True))
    unknown = {cls for cls in labels.values() if cls not in MANUAL_CLASSES}
    if unknown:
        raise ValueError(f"unknown manual classes: {sorted(unknown)}")
    return labels


def assign_split(
    names: list[str], calib_fraction: float, rng: np.random.Generator
) -> dict[str, str]:
    """Split source names into calibration and test sets."""
    order = list(names)
    rng.shuffle(order)
    calib_count = round(len(order) * calib_fraction)
    return {name: SPLIT_CALIB if i < calib_count else SPLIT_TEST for i, name in enumerate(order)}


def variants(
    source: Source, partner: Source, rng: np.random.Generator
) -> list[tuple[str, str, np.ndarray]]:
    """Every synthetic variant of a frontal source as ``(class, parameter, image)``."""
    base = augment.normalize_background(source.image, source.person)
    height, width = base.shape[:2]
    sigma = float(rng.choice(augment.BLUR_SIGMAS))
    dark_gain = float(rng.choice(augment.DARK_GAINS))
    bright_gain = float(rng.choice(augment.BRIGHT_GAINS))
    texture, texture_name = random_texture(height, width, rng)
    target_ratio = float(rng.choice(augment.SMALL_FACE_RATIOS))
    scale = augment.small_face_scale(source.face_ratio, target_ratio)
    magnitude = float(rng.uniform(*augment.SHIFT_RANGE))
    angle = float(rng.uniform(0, 2 * np.pi))
    shift = (magnitude * np.cos(angle), magnitude * np.sin(angle))
    partner_base = augment.normalize_background(partner.image, partner.person)
    items = [
        (CLEAN, "", base),
        ("blur_face", f"sigma={sigma:g}", augment.blur(base, sigma)),
        ("dark", f"gain={dark_gain:g}", augment.gain(base, dark_gain)),
        ("bright", f"gain={bright_gain:g}", augment.gain(base, bright_gain)),
        (
            "busy_background",
            texture_name,
            augment.replace_background(source.image, source.person, texture),
        ),
        (
            "face_shifted",
            f"shift={magnitude:.2f}",
            augment.compose_on_canvas(source.image, source.person, 1.0, shift),
        ),
        ("multi_face", partner.path.stem, augment.collage(base, partner_base)),
    ]
    if scale < MAX_SMALL_FACE_SCALE:
        small = augment.compose_on_canvas(source.image, source.person, scale, (0.0, 0.0))
        items.insert(5, ("face_small", f"ratio={target_ratio:g}", small))
    else:
        log.debug(
            "%s: face too small already (ratio %.2f), skipping face_small",
            source.path.name,
            source.face_ratio,
        )
    return items


def label_row(file_name: str, source: str, cls: str, param: str, split: str) -> dict:
    """One labels.csv row."""
    values = (file_name, source, cls, param, split, CLASS_TO_REQUIREMENT.get(cls, ""))
    return dict(zip(LABEL_COLUMNS, values, strict=True))


def build_dataset(args: argparse.Namespace) -> None:
    """Scan sources, write variants and labels."""
    rng = np.random.default_rng(args.seed)
    paths = image_files(args.valid)[: args.limit] if args.limit else image_files(args.valid)
    manual = load_manual_labels(args.manual_labels)
    backend = make_backend()
    sources, skipped = scan_sources(paths, backend)
    log.info("usable sources: %d, skipped: %d", len(sources), len(skipped))
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    frontal: list[Source] = []
    as_is: list[tuple[Source, str]] = []
    for source in sources:
        override = manual.get(source.path.name)
        if override == MANUAL_EXCLUDE:
            skipped.append(source.path.name)
        elif override in ("non_frontal", "eyes_closed"):
            as_is.append((source, override))
        elif abs(source.yaw) > args.yaw_threshold:
            as_is.append((source, "non_frontal"))
        else:
            frontal.append(source)
    split = assign_split(
        [s.path.name for s in frontal] + [s.path.name for s, _ in as_is], args.calib_fraction, rng
    )
    for source, cls in as_is:
        file_name = f"{source.path.stem}__{cls}.jpg"
        write_bgr(args.out / file_name, source.image, JPEG_QUALITY)
        rows.append(
            label_row(
                file_name, source.path.name, cls, f"yaw={source.yaw:.1f}", split[source.path.name]
            )
        )
    partners = list(frontal)
    rng.shuffle(partners)
    for index, source in enumerate(frontal, start=1):
        partner = partners[index % len(partners)]
        if partner is source:
            partner = partners[(index + 1) % len(partners)]
        for cls, param, image in variants(source, partner, rng):
            file_name = f"{source.path.stem}__{cls}.jpg"
            write_bgr(args.out / file_name, image, JPEG_QUALITY)
            rows.append(label_row(file_name, source.path.name, cls, param, split[source.path.name]))
        if index % 50 == 0:
            log.info("generated variants for %d/%d sources", index, len(frontal))
    labels = pd.DataFrame(rows, columns=list(LABEL_COLUMNS))
    labels.to_csv(args.out / "labels.csv", index=False, encoding="utf-8")
    write_manifest(args, labels, skipped)
    write_manual_check(as_is, rng)


def write_manifest(args: argparse.Namespace, labels: pd.DataFrame, skipped: list[str]) -> None:
    """Counts per class and split plus the generation parameters."""
    manifest = {
        "seed": args.seed,
        "yaw_threshold": args.yaw_threshold,
        "calib_fraction": args.calib_fraction,
        "valid_dir": str(args.valid),
        "skipped_sources": skipped,
        "counts": {
            split: labels[labels["split"] == split]["cls"].value_counts().to_dict()
            for split in (SPLIT_CALIB, SPLIT_TEST)
        },
        "total": len(labels),
    }
    (args.out / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("labels: %d files, manifest written", len(labels))


def write_manual_check(as_is: list[tuple[Source, str]], rng: np.random.Generator) -> None:
    """Sample of automatically selected non-frontal sources for manual confirmation."""
    candidates = [(s.path.name, s.yaw) for s, cls in as_is if cls == "non_frontal"]
    if not candidates:
        return
    picks = rng.choice(
        len(candidates), size=min(MANUAL_CHECK_SAMPLE, len(candidates)), replace=False
    )
    table = pd.DataFrame(
        [{"source": candidates[i][0], "yaw": round(candidates[i][1], 1), "ok": ""} for i in picks]
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "non_frontal_check.csv", index=False, encoding="utf-8")
    log.info("manual check sample: %d rows in %s", len(table), RESULTS / "non_frontal_check.csv")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid", type=Path, default=DATA_VALID)
    parser.add_argument("--out", type=Path, default=DATA_GENERATED)
    parser.add_argument("--yaw-threshold", type=float, default=15.0)
    parser.add_argument("--calib-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--manual-labels", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N sources")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    build_dataset(args)


if __name__ == "__main__":
    main()
