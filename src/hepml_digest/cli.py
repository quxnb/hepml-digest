from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .evaluation import EvaluationSet, evaluate, export_evaluation
from .llm import TERMINAL_API_FAILURES, DeepSeekAnalyzer, DemoAnalyzer
from .models import Paper
from .pipeline import run_pipeline
from .store import load_state, state_lock


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--demo",
        action="store_true",
        help="Run offline with deterministic sample papers and no API key.",
    )
    mode.add_argument(
        "--export-evaluation", type=Path, metavar="PATH",
        help="Export a local annotation sample without network or model calls.",
    )
    mode.add_argument(
        "--evaluate", type=Path, metavar="PATH",
        help="Evaluate local human labels against the current state without API calls.",
    )
    parser.add_argument(
        "--evaluation-limit", type=int, default=100,
        help="Maximum annotation sample size (default: 100).",
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
    parser = build_parser()
    args = parser.parse_args()
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

    if args.export_evaluation or args.evaluate:
        if not settings.state_file.is_file():
            parser.error("Evaluation requires an existing state file")
        with state_lock(settings.state_file):
            state = load_state(settings.state_file)
        if args.export_evaluation:
            if args.evaluation_limit <= 0:
                parser.error("--evaluation-limit must be positive")
            dataset = export_evaluation(
                state, args.export_evaluation, settings.publish_threshold,
                args.evaluation_limit,
            )
            result = {"exported": len(dataset.items)}
        else:
            dataset = EvaluationSet.model_validate_json(
                args.evaluate.read_text(encoding="utf-8")
            )
            result = evaluate(state, dataset, settings.publish_threshold)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

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
            review_max_tokens=settings.review_max_tokens,
            review_reasoning_effort=settings.review_reasoning_effort,
        )
        papers = None

    result = run_pipeline(settings, analyzer, papers=papers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    fatal = result["stop_reason"] in TERMINAL_API_FAILURES
    for alert in result["alerts"]:
        logging.warning("%s", alert)
        if os.getenv("GITHUB_ACTIONS") == "true":
            print(f"::{'error' if fatal else 'warning'}::{alert}")
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        summary = (
            "## Daily digest metrics\n\n"
            "| Metric | Value |\n"
            "|---|---:|\n"
            + "".join(f"| {key} | {value} |\n" for key, value in result.items() if key != "alerts")
            + "".join(f"\n- {alert}" for alert in result["alerts"]) + "\n\n"
        )
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(summary)
    if fatal:
        raise SystemExit(1)
