from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .llm import DeepSeekAnalyzer, DemoAnalyzer
from .models import Paper
from .pipeline import run_pipeline


def _read_prompt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing prompt file: {path}") from exc


def _demo_papers() -> list[Paper]:
    now = datetime.now(timezone.utc)
    return [
        Paper(
            arxiv_id="2607.00001",
            title="Calibrated conditional density estimation for scientific inference",
            abstract=(
                "We introduce a calibrated conditional density estimator and "
                "study coverage under distribution shift. A collider toy example "
                "illustrates simulation-based inference with nuisance parameters."
            ),
            authors=["Ada Example", "Chen Example"],
            link="https://arxiv.org/abs/2607.00001",
            published=now,
            updated=now,
            categories=["stat.ML", "hep-ex"],
        ),
        Paper(
            arxiv_id="2607.00002",
            title="Equivariant generative models on irregular sensor geometries",
            abstract=(
                "We propose an equivariant generative model for irregular sensor "
                "geometries with uncertainty-aware reconstruction."
            ),
            authors=["Lin Example"],
            link="https://arxiv.org/abs/2607.00002",
            published=now,
            updated=now,
            categories=["cs.LG"],
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a daily HEP-ML arXiv digest."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run offline with deterministic sample papers and no API key.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override OUTPUT_DIR.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Override STATE_FILE.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings.from_env()
    if args.demo and args.output_dir is None:
        settings.output_dir = Path("build/demo-public")
    if args.demo and args.state_file is None:
        settings.state_file = Path("build/demo-state.json")
    if args.output_dir:
        settings.output_dir = args.output_dir
    if args.state_file:
        settings.state_file = args.state_file

    if args.demo:
        analyzer = DemoAnalyzer()
        papers = _demo_papers()
    else:
        analyzer = DeepSeekAnalyzer(
            settings.screening_model,
            settings.review_model,
            _read_prompt(settings.prompt_dir / "screen.txt"),
            _read_prompt(settings.prompt_dir / "review.txt"),
            settings.request_timeout_seconds,
        )
        papers = None

    result = run_pipeline(settings, analyzer, papers=papers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
