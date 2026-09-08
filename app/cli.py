"""Command line interface: ``photocheck check`` and ``photocheck models download``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from app.analyzers import REGISTRY
from app.analyzers.context import FaceBackend
from app.analyzers.models import download_models
from app.conclusion.llm_gen import generate_conclusion, template_synthesizer
from app.conclusion.template_gen import render_conclusion
from app.core.engine import Engine
from app.core.errors import PhotoCheckError, SpecError
from app.core.pipeline import check_image
from app.core.report import Report
from app.core.spec import load_spec
from app.llm.factory import build_client

log = logging.getLogger("photocheck")

DEFAULT_MODELS_DIR = Path("models")
LLM_MODEL = "claude-sonnet-4-6"
LLM_CACHE_DIR = Path("results") / "llm_cache"
NUM_FACES = 5
MIN_DETECTION_CONFIDENCE = 0.5
EXIT_ACCEPTED = 0
EXIT_REJECTED = 1
EXIT_USAGE = 2
EXIT_RUNTIME = 3

BackendFactory = Callable[[Path], FaceBackend]


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with the ``check`` and ``models`` subcommands."""
    parser = argparse.ArgumentParser(prog="photocheck")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subcommands = parser.add_subparsers(dest="command", required=True)

    check = subcommands.add_parser("check", help="check an image against a specification")
    check.add_argument("image", type=Path)
    check.add_argument("--spec", type=Path, required=True, help="YAML specification")
    check.add_argument("--json", type=Path, help="write the JSON report to this path")
    check.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    check.add_argument(
        "--llm", action="store_true", help="verbalize the report with a language model"
    )
    check.add_argument(
        "--mock", action="store_true", help="with --llm: use the offline mock client"
    )
    check.add_argument("--llm-cache", type=Path, default=LLM_CACHE_DIR, help="recorded responses")

    models = subcommands.add_parser("models", help="manage model files")
    models_commands = models.add_subparsers(dest="models_command", required=True)
    download = models_commands.add_parser("download", help="fetch MediaPipe model files")
    download.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    download.add_argument("--force", action="store_true", help="re-download existing files")
    return parser


def default_backend(models_dir: Path) -> FaceBackend:
    """MediaPipe backend on the models directory."""
    from app.analyzers.mediapipe_backend import MediaPipeBackend

    return MediaPipeBackend(models_dir, NUM_FACES, MIN_DETECTION_CONFIDENCE)


def run_check(args: argparse.Namespace, backend_factory: BackendFactory) -> int:
    """Execute ``photocheck check`` and return the exit code."""
    try:
        engine = Engine(load_spec(args.spec, REGISTRY))
    except (SpecError, OSError) as exc:
        log.error("cannot load specification %s: %s", args.spec, exc)
        return EXIT_USAGE
    try:
        result = check_image(args.image, engine, backend_factory(args.models_dir))
    except PhotoCheckError as exc:
        log.error("%s", exc)
        return EXIT_RUNTIME
    report = result.report
    log.info("processed %s in %.0f ms", args.image.name, result.elapsed_ms)
    print(format_verdicts(report, engine))
    print()
    print(conclusion_text(report, args))
    if args.json is not None:
        report.save(args.json)
        print()
        print(f"JSON: {args.json}")
    return EXIT_ACCEPTED if report.accepted else EXIT_REJECTED


def conclusion_text(report: Report, args: argparse.Namespace) -> str:
    """Conclusion from the template, or from the language model when requested."""
    if not args.llm:
        return render_conclusion(report)
    client = build_client(args.mock, LLM_MODEL, args.llm_cache, template_synthesizer)
    conclusion = generate_conclusion(report, client)
    log.info("conclusion source: %s", conclusion.source)
    return conclusion.text


def run_models_download(args: argparse.Namespace) -> int:
    """Execute ``photocheck models download`` and return the exit code."""
    try:
        paths = download_models(args.models_dir, args.force)
    except (PhotoCheckError, OSError) as exc:
        log.error("download failed: %s", exc)
        return EXIT_RUNTIME
    for path in paths:
        print(path)
    return EXIT_ACCEPTED


def format_verdicts(report: Report, engine: Engine) -> str:
    """Plain-text table of verdicts."""
    lines = [
        f"Спецификация: {engine.spec.name} — {engine.spec.description}",
        f"Изображение: {report.image}",
        "",
        f"{'требование':<22}{'статус':<12}{'тип':<6}измерено",
    ]
    for verdict in report.verdicts:
        measured = ", ".join(
            f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in (verdict.measured or {}).items()
        )
        if verdict.blocked_by is not None:
            measured = f"заблокировано: {verdict.blocked_by}"
        elif verdict.measured is None and verdict.reason:
            measured = verdict.reason
        lines.append(f"{verdict.id:<22}{verdict.status.value:<12}{verdict.kind.value:<6}{measured}")
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None, backend_factory: BackendFactory = default_backend
) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check" and args.mock and not args.llm:
        parser.error("--mock requires --llm")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "check":
        return run_check(args, backend_factory)
    return run_models_download(args)


if __name__ == "__main__":
    sys.exit(main())
