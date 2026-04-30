#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          TROVE PIPELINE  ·  Interactive Terminal GUI            ║
╚══════════════════════════════════════════════════════════════════╝

Launch:  python gui.py
         python gui.py --config my_config.yaml   (to skip wizard)
         python gui.py --run my_config.yaml       (to run immediately)
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Optional

# ── Rich ──────────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.columns import Columns
from rich.rule import Rule
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TaskProgressColumn, TimeRemainingColumn, TextColumn,
)
from rich.syntax import Syntax
from rich.tree import Tree
from rich import box
import rich.traceback
rich.traceback.install()

# ── Pipeline ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import PipelineConfig, PipelineRunner, list_available_tasks, PromptLibrary

console = Console()

# ═══════════════════════════════════════════════════════════════════
# Theme / Helpers
# ═══════════════════════════════════════════════════════════════════

BRAND   = "[bold cyan]TROVE[/bold cyan]"
ACCENT  = "cyan"
OK      = "[bold green]✓[/bold green]"
WARN    = "[bold yellow]⚠[/bold yellow]"
ERR     = "[bold red]✗[/bold red]"
BULLET  = "[dim]·[/dim]"

def _h(text: str) -> str:
    return f"[bold {ACCENT}]{text}[/bold {ACCENT}]"

def _dim(text: str) -> str:
    return f"[dim]{text}[/dim]"

def _sep(title: str = "") -> None:
    console.print(Rule(title, style=ACCENT))

def _ask(label: str, default: str = "", choices: Optional[list] = None) -> str:
    return Prompt.ask(
        f"  [bold]{label}[/bold]",
        default=default,
        choices=choices,
    )

def _ask_int(label: str, default: int) -> int:
    return IntPrompt.ask(f"  [bold]{label}[/bold]", default=default)

def _confirm(label: str, default: bool = True) -> bool:
    return Confirm.ask(f"  [bold]{label}[/bold]", default=default)


# ═══════════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════════

def _banner() -> None:
    console.print()
    console.print(Panel.fit(
        Text.assemble(
            ("  TROVE  ", f"bold {ACCENT}"),
            ("Generic Evaluation Pipeline\n", "bold white"),
            ("  Task-Agnostic · Emission-Tracked · CSV-Driven  ", "dim"),
        ),
        border_style=ACCENT,
        padding=(1, 4),
    ))
    console.print()


# ═══════════════════════════════════════════════════════════════════
# Config Wizard
# ═══════════════════════════════════════════════════════════════════

def run_wizard() -> PipelineConfig:
    """Interactive step-by-step configuration wizard."""
    _banner()
    console.print(_h("  Configuration Wizard"), "\n")

    cfg: dict = {}

    # ── Step 1: Model ────────────────────────────────────────────
    _sep("Step 1 · Model")
    console.print(_dim("  The Ollama model to use for inference."))
    console.print(_dim("  Run [bold]ollama list[/bold] to see available models.\n"))
    cfg["model_name"] = _ask("Model name", "qwen2.5-coder:0.5b")

    # ── Step 2: Task ─────────────────────────────────────────────
    _sep("Step 2 · Task")
    tasks = list_available_tasks()
    _show_tasks_table(tasks)
    cfg["task_name"] = _ask("Task name", tasks[0] if tasks else "security_detection")

    # ── Step 3: Dataset ──────────────────────────────────────────
    _sep("Step 3 · Dataset")
    console.print(_dim("  Supported formats: jsonl, csv, tsv, json\n"))
    cfg["dataset_path"] = _ask("Dataset path", "datasets\quixbugs_test.csv")

    fmt = Path(cfg["dataset_path"]).suffix.lstrip(".") or "jsonl"
    cfg["dataset_format"] = _ask("Dataset format", fmt, ["jsonl", "csv", "tsv", "json"])

    cfg["sample_size"] = _ask_int(
        "Sample size (0 = full dataset)", 1000
    )

    # ── Step 4: Prompts ──────────────────────────────────────────
    _sep("Step 4 · Prompt Library")
    default_prompts = str(Path("prompts") / cfg["task_name"])
    cfg["prompts_dir"] = _ask("Prompts directory", default_prompts)

    _show_prompts_info(cfg["prompts_dir"])

    sections_raw = _ask(
        "Sections to use (comma-separated, blank = all)",
        "",
    )
    cfg["prompt_sections"] = (
        [s.strip().upper() for s in sections_raw.split(",") if s.strip()]
        if sections_raw.strip() else []
    )

    # ── Step 5: Combinations ─────────────────────────────────────
    _sep("Step 5 · Prompt Combinations")
    cfg["combo_mode"] = _ask(
        "Combination mode",
        "random",
        ["random", "exhaustive"],
    )
    if cfg["combo_mode"] == "random":
        cfg["n_combos"] = _ask_int("Number of random combos per sample", 10)
        cfg["min_sections"] = _ask_int("Minimum sections per combo", 2)
    cfg["seed"] = _ask_int("Random seed", 42)

    # ── Step 6: Inference ────────────────────────────────────────
    _sep("Step 6 · Inference Settings")
    cfg["temperature"] = float(_ask("Temperature (0.0 = deterministic)", "0.0"))
    cfg["max_tokens"]  = _ask_int("Max tokens per response", 16)

    # ── Step 7: Output ───────────────────────────────────────────
    _sep("Step 7 · Output")
    cfg["output_dir"]        = _ask("Output directory", "outputs")
    cfg["track_emissions"]   = _confirm("Track CO₂ emissions with CodeCarbon?", True)
    if cfg["track_emissions"]:
        cfg["carbon_output_dir"] = _ask("Carbon reports directory", "carbon_reports")

    # ── Review ───────────────────────────────────────────────────
    _sep()
    _show_config_review(cfg)

    if not _confirm("\n  Proceed with this configuration?", True):
        console.print("\n  [yellow]Wizard cancelled.[/yellow]\n")
        sys.exit(0)

    # Optionally save config
    if _confirm("  Save configuration to YAML?", True):
        yaml_path = _ask("  YAML save path", f"config_{cfg['task_name']}.yaml")
        config = PipelineConfig.from_dict(cfg)
        config.to_yaml(yaml_path)
        console.print(f"\n  {OK} Config saved → [cyan]{yaml_path}[/cyan]\n")

    return PipelineConfig.from_dict(cfg)


def _show_tasks_table(tasks: list[str]) -> None:
    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style=f"bold {ACCENT}")
    table.add_column("Task Name", style="cyan")
    table.add_column("Description", style="dim")

    descriptions = {
        "security_detection": "Binary vulnerability detection (0=safe, 1=vulnerable)",
        "bug_detection":     "Binary bug detection (0=no bug, 1=bug)",
        "technical_debt":  "Multiclass code smell classification (god_class, long_method, feature_envy, data_class, none)",
    }
    for t in tasks:
        table.add_row(t, descriptions.get(t, "Custom task module"))
    console.print(table)
    console.print()


def _show_prompts_info(prompts_dir: str) -> None:
    p = Path(prompts_dir)
    if not p.exists():
        console.print(f"  {WARN} Directory [yellow]{prompts_dir}[/yellow] not found — will error at runtime.\n")
        return
    try:
        lib = PromptLibrary(prompts_dir)
        summary = lib.summary()
        if summary:
            parts = [f"[cyan]{s}[/cyan]=[bold]{n}[/bold]" for s, n in summary.items()]
            console.print(f"  {OK} Found sections: {' '.join(parts)}\n")
        else:
            console.print(f"  {WARN} No CSV files found in {prompts_dir}\n")
    except Exception as e:
        console.print(f"  {WARN} Could not load prompts: {e}\n")


def _show_config_review(cfg: dict) -> None:
    console.print(f"\n  {_h('Configuration Review')}\n")
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Key", style=f"bold {ACCENT}", width=28)
    table.add_column("Value", style="white")

    for k, v in cfg.items():
        if isinstance(v, list):
            v = ", ".join(v) if v else "[dim](all)[/dim]"
        table.add_row(k.replace("_", " ").title(), str(v))

    console.print(table)


# ═══════════════════════════════════════════════════════════════════
# Run Pipeline (with live progress)
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(config: PipelineConfig) -> None:
    """Validate config, then run the pipeline with rich output."""

    # Validate
    errors = config.validate()
    if errors:
        console.print(f"\n  {ERR} [bold red]Config validation failed:[/bold red]")
        for e in errors:
            console.print(f"     {BULLET} {e}")
        console.print()
        sys.exit(1)

    # Pre-run summary panel
    _sep()
    _show_run_summary(config)
    _sep()
    console.print()

    # Run
    t0 = time.perf_counter()
    try:
        runner = PipelineRunner(config)

        # Patch tqdm to suppress in favour of rich output
        # (PipelineRunner uses tqdm internally; we let it run naturally)
        result = runner.run()

    except KeyboardInterrupt:
        console.print(f"\n\n  {WARN} [yellow]Interrupted by user.[/yellow]\n")
        sys.exit(130)
    except Exception:
        console.print_exception()
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    _show_results(result, elapsed)


def _show_run_summary(config: PipelineConfig) -> None:
    console.print(f"\n  {_h('Run Plan')}\n")

    grid = Table.grid(padding=(0, 4))
    grid.add_column(style=f"bold {ACCENT}", width=22)
    grid.add_column(style="white")

    grid.add_row("Model",        config.model_name)
    grid.add_row("Task",         config.task_name)
    grid.add_row("Dataset",      config.dataset_path)
    grid.add_row("Format",       config.dataset_format)
    grid.add_row("Sample size",  str(config.sample_size) if config.sample_size else "full")
    grid.add_row("Combo mode",   config.combo_mode)
    grid.add_row("Num combos",   str(config.n_combos))
    grid.add_row("Prompts dir",  config.prompts_dir)
    grid.add_row("Track CO₂",   "yes" if config.track_emissions else "no")
    console.print(grid)
    console.print()


def _show_results(result: dict, elapsed: float) -> None:
    summary = result.get("summary", {})
    output_path = result.get("output_path", "?")
    emissions = result.get("emissions")
    errors = result.get("errors", 0)

    _sep()
    console.print(f"\n  {_h('Results')}\n")

    # Metrics table
    table = Table(box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}", padding=(0, 2))
    table.add_column("Metric", style=f"bold {ACCENT}")
    table.add_column("Value", justify="right", style="bold white")

    for k, v in summary.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            # Highlight accuracy-like metrics
            if any(w in k for w in ("accuracy", "f1", "precision", "recall", "match")):
                color = "green" if v >= 0.7 else ("yellow" if v >= 0.5 else "red")
                table.add_row(label, f"[{color}]{v:.4f}[/{color}]")
            else:
                table.add_row(label, f"{v:.6f}")
        else:
            table.add_row(label, str(v))

    table.add_row("[dim]Elapsed time[/dim]", f"[dim]{elapsed:.1f}s[/dim]")
    if emissions is not None:
        table.add_row("[dim]CO₂ emitted (kg)[/dim]", f"[dim]{emissions:.8f}[/dim]")
    if errors:
        table.add_row(f"[red]Errors[/red]", f"[red]{errors}[/red]")

    console.print(table)

    console.print()
    console.print(f"  {OK} Results saved  →  [cyan]{output_path}[/cyan]")
    summary_path = Path(output_path).with_suffix(".summary.json")
    if summary_path.exists():
        console.print(f"  {OK} Summary JSON   →  [cyan]{summary_path}[/cyan]")
    console.print()
    _sep()
    console.print()


# ═══════════════════════════════════════════════════════════════════
# Result Analysis Mode
# ═══════════════════════════════════════════════════════════════════

def run_analysis(csv_path: str) -> None:
    """Load an existing results CSV and display analysis."""
    import pandas as pd

    console.print(f"\n  Loading results from: [cyan]{csv_path}[/cyan]\n")
    df = pd.read_csv(csv_path)
    console.print(f"  {OK} {len(df)} rows loaded.\n")

    _sep("Per-Combo Accuracy Breakdown")
    if "prompt_combo" in df.columns and "correct" in df.columns:
        breakdown = (
            df.groupby("prompt_combo")["correct"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "accuracy", "count": "n"})
            .sort_values("accuracy", ascending=False)
        )

        table = Table(box=box.SIMPLE_HEAD, header_style=f"bold {ACCENT}")
        table.add_column("Prompt Combo", style="cyan", overflow="fold")
        table.add_column("Accuracy", justify="right", style="bold")
        table.add_column("N", justify="right", style="dim")

        for _, row in breakdown.head(20).iterrows():
            acc = row["accuracy"]
            color = "green" if acc >= 0.7 else ("yellow" if acc >= 0.5 else "red")
            table.add_row(
                str(row["prompt_combo"])[:80],
                f"[{color}]{acc:.3f}[/{color}]",
                str(int(row["n"])),
            )
        console.print(table)

    _sep("Summary Stats")
    if "correct" in df.columns:
        overall = df["correct"].mean()
        color = "green" if overall >= 0.7 else ("yellow" if overall >= 0.5 else "red")
        console.print(f"  Overall accuracy: [{color}]{overall:.4f}[/{color}]  ({df['correct'].sum()} / {len(df)})\n")

    if "latency_sec" in df.columns:
        console.print(f"  Avg latency:  {df['latency_sec'].mean():.3f}s")
        console.print(f"  Max latency:  {df['latency_sec'].max():.3f}s\n")

    # Load and show JSON summary if available
    summary_path = Path(csv_path).with_suffix(".summary.json")
    if summary_path.exists():
        _sep("Saved Summary")
        with open(summary_path) as f:
            data = json.load(f)
        syntax = Syntax(json.dumps(data, indent=2), "json", theme="monokai", line_numbers=False)
        console.print(syntax)

    console.print()


# ═══════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="TROVE Pipeline — Interactive GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  python gui.py                          → interactive wizard + run
  python gui.py --config cfg.yaml       → wizard pre-filled, then run
  python gui.py --run cfg.yaml          → skip wizard, run immediately
  python gui.py --analyze results.csv   → analyze existing results CSV
  python gui.py --list-tasks            → list available task modules
        """,
    )
    parser.add_argument("--config",       type=str, help="Load config from YAML (still shows wizard)")
    parser.add_argument("--run",          type=str, help="Load config and run immediately (no wizard)")
    parser.add_argument("--analyze",      type=str, help="Analyze an existing results CSV")
    parser.add_argument("--list-tasks",   action="store_true", help="List available task modules")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_tasks:
        _banner()
        _sep("Available Tasks")
        tasks = list_available_tasks()
        _show_tasks_table(tasks)
        return

    if args.analyze:
        _banner()
        run_analysis(args.analyze)
        return

    if args.run:
        _banner()
        console.print(f"  {_h('Loading config')} from [cyan]{args.run}[/cyan]\n")
        config = PipelineConfig.from_yaml(args.run)
        run_pipeline(config)
        return

    # Full wizard flow
    if args.config:
        console.print(f"  {OK} Pre-loading config from [cyan]{args.config}[/cyan]\n")
        config = PipelineConfig.from_yaml(args.config)
        _banner()
        _show_config_review(config.__dict__)
        if _confirm("\n  Use this config and run?", True):
            run_pipeline(config)
            return
        # Otherwise fall through to wizard
        config = run_wizard()
    else:
        config = run_wizard()

    run_pipeline(config)


if __name__ == "__main__":
    main()
