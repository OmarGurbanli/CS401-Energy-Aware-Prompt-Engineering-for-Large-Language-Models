#!/usr/bin/env python3
"""
TROVE Pipeline — CLI Runner (no GUI, pure argparse)

Usage examples:
  python run.py --task security_detection --dataset primevul_test.jsonl \
                --model qwen2.5-coder:0.5b --n_combos 10 --sample 500

  python run.py --config my_config.yaml

  python run.py --task text_classification \
                --dataset sentiment_test.csv --dataset-format csv \
                --prompts-dir prompts/text_classification \
                --model llama3.2:3b --n_combos 5 --no-emissions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import PipelineConfig, PipelineRunner, list_available_tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="TROVE Generic Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config shortcut ──────────────────────────────────────────
    parser.add_argument(
        "--config", type=str,
        help="Path to a YAML config file. All other flags override YAML values.",
    )

    # ── Core ────────────────────────────────────────────────────
    parser.add_argument("--model",           type=str, default=None, help="Ollama model name")
    parser.add_argument("--task",            type=str, default=None,
                        help=f"Task module name. Available: {list_available_tasks()}")
    parser.add_argument("--dataset",         type=str, default=None, help="Path to dataset file")
    parser.add_argument("--dataset-format",  type=str, default=None,
                        choices=["jsonl", "csv", "tsv", "json"],
                        help="Dataset format")
    parser.add_argument("--sample",          type=int, default=None,
                        help="Number of samples (0 = full dataset)")
    parser.add_argument("--seed",            type=int, default=None)

    # ── Prompts ─────────────────────────────────────────────────
    parser.add_argument("--prompts-dir",     type=str, default=None,
                        help="Directory containing prompt section CSV files")
    parser.add_argument("--sections",        type=str, default=None,
                        help="Comma-separated section names to use, e.g. T,R,O")

    # ── Combinations ─────────────────────────────────────────────
    parser.add_argument("--combo-mode",      type=str, default=None,
                        choices=["random", "exhaustive"])
    parser.add_argument("--n-combos",        type=int, default=None,
                        help="Number of random combinations per sample")
    parser.add_argument("--min-sections",    type=int, default=None,
                        help="Minimum sections per combination")

    # ── Inference ────────────────────────────────────────────────
    parser.add_argument("--temperature",     type=float, default=None)
    parser.add_argument("--max-tokens",      type=int, default=None)

    # ── Output ──────────────────────────────────────────────────
    parser.add_argument("--output-dir",      type=str, default=None)
    parser.add_argument("--output-file",     type=str, default=None)
    parser.add_argument("--no-emissions",    action="store_true",
                        help="Disable CodeCarbon emissions tracking")

    # ── Utility ─────────────────────────────────────────────────
    parser.add_argument("--list-tasks",      action="store_true",
                        help="List available task modules and exit")
    parser.add_argument("--validate-only",   action="store_true",
                        help="Validate config and exit without running")
    parser.add_argument("--save-config",     type=str, default=None,
                        help="Save merged config to this YAML path and exit")

    return parser.parse_args()


def build_config(args) -> PipelineConfig:
    # Start from YAML if provided
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()
    # Note: PipelineConfig.from_yaml / from_dict use dataclass field names,
    # not hasattr, so default_factory fields (lists) are handled correctly.

    # Override with CLI flags
    if args.model:         config.model_name     = args.model
    if args.task:          config.task_name      = args.task
    if args.dataset:       config.dataset_path   = args.dataset
    if args.dataset_format: config.dataset_format = args.dataset_format
    if args.sample is not None: config.sample_size = args.sample
    if args.seed is not None:   config.seed        = args.seed

    if args.prompts_dir:   config.prompts_dir    = args.prompts_dir
    if args.sections:
        config.prompt_sections = [s.strip().upper() for s in args.sections.split(",")]

    if args.combo_mode:    config.combo_mode     = args.combo_mode
    if args.n_combos:      config.n_combos       = args.n_combos
    if args.min_sections:  config.min_sections   = args.min_sections

    if args.temperature is not None: config.temperature = args.temperature
    if args.max_tokens:    config.max_tokens     = args.max_tokens

    if args.output_dir:    config.output_dir     = args.output_dir
    if args.output_file:   config.output_file    = args.output_file
    if args.no_emissions:  config.track_emissions = False

    # Auto-detect prompts dir if not set
    if not args.prompts_dir and not args.config:
        auto = Path("prompts") / config.task_name
        if auto.exists():
            config.prompts_dir = str(auto)

    # Auto-detect dataset format from extension
    if not args.dataset_format and config.dataset_path:
        ext = Path(config.dataset_path).suffix.lstrip(".").lower()
        if ext in ("jsonl", "csv", "tsv", "json"):
            config.dataset_format = ext

    return config


def main() -> None:
    args = parse_args()

    if args.list_tasks:
        tasks = list_available_tasks()
        print("\nAvailable tasks:")
        for t in tasks:
            print(f"  {t}")
        print()
        return

    config = build_config(args)

    if args.save_config:
        config.to_yaml(args.save_config)
        print(f"Config saved to: {args.save_config}")
        return

    errors = config.validate()
    if errors:
        print("\n[ERROR] Config validation failed:")
        for e in errors:
            print(f"  · {e}")
        sys.exit(1)

    if args.validate_only:
        print("\n[OK] Config is valid.")
        return

    # Run
    runner = PipelineRunner(config)
    result = runner.run()

    print(f"\nDone. Results at: {result['output_path']}")
    if result.get("errors"):
        print(f"Inference errors: {result['errors']}")


if __name__ == "__main__":
    main()
