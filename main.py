"""CLI entry point: python main.py --input ./resumes --output ./output/results.json"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.ai.adapter import build_adapter
from src.config import Config
from src.pipeline import run_pipeline
from src.reporting.json_report import build_report, write_json
from src.reporting.terminal import ProgressReporter, render_header, render_report


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screen and rank PDF resumes for an AI/Python SDE internship.")
    p.add_argument("--input", required=True, type=Path, help="folder containing resume PDFs")
    p.add_argument("--output", required=True, type=Path, help="path of the JSON result file")
    p.add_argument("--top", type=int, default=10, help="candidates shown in the terminal summary")
    p.add_argument("--no-github", action="store_true", help="skip GitHub enrichment (GitHub score = 0)")
    p.add_argument("--no-llm", action="store_true", help="ignore LLM_PROVIDER and run deterministic only")
    p.add_argument("--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    config = Config.from_env()
    adapter = None if args.no_llm else build_adapter(config)

    print(render_header(str(args.input)))
    try:
        batch = run_pipeline(args.input, config, adapter=adapter, enable_github=not args.no_github,
                             on_progress=ProgressReporter())
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(render_report(batch, top=args.top))
    write_json(build_report(batch), args.output)
    print(f"Full results: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
