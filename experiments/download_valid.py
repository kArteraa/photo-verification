"""Download a small FFHQ subset (~500 images) via HF streaming into data/valid/.

Usage:
    pip install datasets pillow
    python experiments/download_valid.py --n 500 --out data/valid

Streaming mode fetches only the rows actually consumed, so this transfers
~50-150 MB instead of the full 27+ GB archive.
FFHQ license: CC BY-NC-SA 4.0 (research use is fine); see NVlabs/ffhq-dataset.
"""

import argparse
import logging
from pathlib import Path

from datasets import load_dataset

log = logging.getLogger("download_valid")

# Mirrors to try in order: (repo_id, image_column). 512px preferred for our analyzers.
MIRRORS = [
    ("Ryan-sjtu/ffhq512-caption", "image"),
    ("merkol/ffhq-256", "image"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="number of images")
    parser.add_argument("--out", type=Path, default=Path("data/valid"))
    parser.add_argument(
        "--skip", type=int, default=0, help="skip first K rows (to get a different subset)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    ds = None
    for repo, col in MIRRORS:
        try:
            log.info("trying mirror %s ...", repo)
            ds = load_dataset(repo, split="train", streaming=True)
            image_col = col
            log.info("using mirror %s", repo)
            break
        except Exception as exc:
            log.warning("mirror %s failed: %s", repo, exc)
    if ds is None:
        raise SystemExit("no FFHQ mirror reachable; check network/HF availability")

    saved = 0
    for i, row in enumerate(ds):
        if i < args.skip:
            continue
        img = row[image_col]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(args.out / f"ffhq_{i:05d}.jpg", quality=95)
        saved += 1
        if saved % 50 == 0:
            log.info("saved %d/%d", saved, args.n)
        if saved >= args.n:
            break

    log.info("done: %d images in %s", saved, args.out)


if __name__ == "__main__":
    main()
