"""
╔══════════════════════════════════════════════════════════════════════╗
║              TROVE GENERIC EVALUATION PIPELINE                      ║
║  Configurable • Task-Agnostic • Emission-Tracked • CSV-Driven       ║
╚══════════════════════════════════════════════════════════════════════╝

Architecture:
  - TaskModule   : pluggable per-task logic (dataset loader + evaluator)
  - PromptLibrary: loads prompt sections from CSV files
  - PipelineConfig: validated configuration (YAML / dict / GUI)
  - Runner       : orchestrates inference, tracking, result saving
"""

from __future__ import annotations

import csv
import importlib
import itertools
import json
import os
import random
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import pandas as pd
import yaml
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """All pipeline parameters in one validated dataclass."""

    # Model
    model_name: str = "qwen2.5-coder:0.5b"

    # Task
    task_name: str = "security_detection"   # maps to tasks/<task_name>.py

    # Dataset
    dataset_path: str = ""
    dataset_format: str = "jsonl"           # jsonl | csv | tsv | json
    sample_size: int = 0                    # 0 = full dataset
    stratify_column: Optional[str] = None  # column to stratify on (optional)
    seed: int = 42

    # Prompts
    prompts_dir: str = "prompts"            # folder with section CSVs
    prompt_sections: list[str] = field(default_factory=list)
    # e.g. ["T", "R", "O", "V", "E"]
    # if empty, ALL CSVs in prompts_dir are used as sections

    # Combination strategy
    combo_mode: str = "random"              # random | exhaustive
    n_combos: int = 10
    min_sections: int = 2                   # minimum sections per combo

    # Output
    output_dir: str = "outputs"
    output_file: str = ""                   # auto-generated if empty

    # Carbon tracking
    track_emissions: bool = True
    carbon_output_dir: str = "carbon_reports"

    # Inference
    temperature: float = 0.0
    max_tokens: int = 16
    request_timeout: int = 120

    @classmethod
    def _field_names(cls) -> set:
        import dataclasses
        return {f.name for f in dataclasses.fields(cls)}

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        known = cls._field_names()
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        known = cls._field_names()
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def validate(self) -> list[str]:
        """Return list of validation error strings (empty = OK)."""
        errors = []
        if not self.model_name:
            errors.append("model_name must not be empty")
        if not self.dataset_path:
            errors.append("dataset_path must not be empty")
        if not Path(self.dataset_path).exists():
            errors.append(f"dataset_path does not exist: {self.dataset_path}")
        if self.combo_mode not in ("random", "exhaustive"):
            errors.append(f"combo_mode must be 'random' or 'exhaustive', got: {self.combo_mode}")
        if self.n_combos < 1:
            errors.append("n_combos must be >= 1")
        return errors


# ─────────────────────────────────────────────────────────────────────
# Prompt Library  (loads from CSV files)
# ─────────────────────────────────────────────────────────────────────

class PromptLibrary:
    """
    Loads prompt sections from CSV files in a directory.

    Expected CSV structure (one file per section, e.g. T.csv):
        key,text
        T0,"You are a security expert..."
        T1,"You are a static analysis tool..."

    OR for few-shot examples (E.csv):
        key,input,output
        E0_shot1,"<code>","1"
        E0_shot2,"<code>","0"
        E1_shot1,"<code>","1"
    """

    def __init__(self, prompts_dir: str, section_names: Optional[list[str]] = None):
        self.prompts_dir = Path(prompts_dir)
        self.sections: dict[str, dict[str, Any]] = {}
        self._load(section_names)

    def _load(self, section_names: Optional[list[str]]) -> None:
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

        csv_files = sorted(self.prompts_dir.glob("*.csv"))
        if section_names:
            csv_files = [self.prompts_dir / f"{s}.csv" for s in section_names]

        for csv_file in csv_files:
            section_key = csv_file.stem.upper()
            if not csv_file.exists():
                raise FileNotFoundError(f"Prompt CSV not found: {csv_file}")

            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

                if "input" in fieldnames and "output" in fieldnames:
                    # Few-shot examples format
                    self.sections[section_key] = self._load_examples(reader)
                elif "text" in fieldnames:
                    # Simple text format
                    self.sections[section_key] = self._load_text(reader)
                else:
                    raise ValueError(
                        f"CSV {csv_file} must have either (key, text) "
                        f"or (key, input, output) columns. Found: {fieldnames}"
                    )

    @staticmethod
    def _load_text(reader) -> dict[str, str]:
        return {row["key"]: row["text"] for row in reader if row.get("key") and row.get("text")}

    @staticmethod
    def _load_examples(reader) -> dict[str, list[dict]]:
        """Group example shots by their prefix (e.g. E0_shot1 → E0)."""
        grouped: dict[str, list[dict]] = {}
        for row in reader:
            key = row.get("key", "")
            # Support both "E0" (single shot per key) and "E0_shot1" grouping
            group = key.split("_")[0] if "_" in key else key
            if group not in grouped:
                grouped[group] = []
            grouped[group].append({
                "input": row.get("input", ""),
                "output": row.get("output", ""),
                "label": row.get("label", row.get("output", "")),
            })
        return grouped

    def get_section_keys(self, section: str) -> list[str]:
        return list(self.sections.get(section, {}).keys())

    def render_section(self, section: str, key: str, input_label: str = "Input") -> str:
        """Render a section entry to a prompt string."""
        data = self.sections.get(section, {})
        val = data.get(key)
        if val is None:
            return ""

        if isinstance(val, str):
            return val

        # Few-shot examples: val is a list of dicts
        if isinstance(val, list):
            lines = ["***EXAMPLE INPUT, OUTPUT FROM PREVIOUS INTERACTIONS***"]
            for i, shot in enumerate(val, 1):
                lines.append(f"\nExample {i}:")
                lines.append(f"{input_label}:\n{shot['input']}")
                lines.append(f"Output:\n{shot['output']}")
            return "\n".join(lines)

        return str(val)

    def summary(self) -> dict[str, int]:
        return {s: len(keys) for s, keys in self.sections.items()}


# ─────────────────────────────────────────────────────────────────────
# Combination Generator
# ─────────────────────────────────────────────────────────────────────

class CombinationGenerator:
    """Generates prompt section combinations (random or exhaustive)."""

    def __init__(self, library: PromptLibrary, config: PipelineConfig):
        self.library = library
        self.config = config
        self.section_keys = list(library.sections.keys())

    def generate(self) -> list[dict[str, Optional[str]]]:
        if self.config.combo_mode == "exhaustive":
            return self._exhaustive()
        return self._random()

    def _random(self) -> list[dict[str, Optional[str]]]:
        random.seed(self.config.seed)
        combos = []
        for _ in range(self.config.n_combos):
            n_secs = random.randint(
                min(self.config.min_sections, len(self.section_keys)),
                len(self.section_keys),
            )
            chosen = random.sample(self.section_keys, n_secs)
            combo: dict[str, Optional[str]] = {s: None for s in self.section_keys}
            for s in chosen:
                keys = self.library.get_section_keys(s)
                if keys:
                    combo[s] = random.choice(keys)
            combos.append(combo)
        return combos

    def _exhaustive(self) -> list[dict[str, Optional[str]]]:
        pools = {
            s: [None] + self.library.get_section_keys(s)
            for s in self.section_keys
        }
        combos = []
        for combo_vals in itertools.product(*[pools[s] for s in self.section_keys]):
            config = dict(zip(self.section_keys, combo_vals))
            if all(v is None for v in config.values()):
                continue
            non_none = sum(1 for v in config.values() if v is not None)
            if non_none >= self.config.min_sections:
                combos.append(config)
        return combos


# ─────────────────────────────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """Assembles a final prompt string from sections + a code/text input."""

    def __init__(self, library: PromptLibrary, input_label: str = "Input"):
        self.library = library
        self.input_label = input_label

    def build(self, sample_input: str, combo: dict[str, Optional[str]]) -> str:
        parts = []
        for section, key in combo.items():
            if key is not None:
                rendered = self.library.render_section(section, key, self.input_label)
                if rendered:
                    parts.append(rendered)
        parts.append(f"\n{self.input_label}:\n{sample_input}")
        return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────
# Ollama Inference Client
# ─────────────────────────────────────────────────────────────────────

class OllamaClient:
    """Thin wrapper around ollama.chat with retry logic."""

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 16):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._ollama = None

    def _get_ollama(self):
        if self._ollama is None:
            import ollama
            self._ollama = ollama
        return self._ollama

    def query(self, prompt: str, retries: int = 3) -> tuple[str, float]:
        """
        Returns (raw_response_text, latency_seconds).
        Raises RuntimeError after all retries exhausted.
        """
        ollama = self._get_ollama()
        for attempt in range(retries):
            try:
                t0 = time.perf_counter()
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                )
                latency = time.perf_counter() - t0
                text = response["message"]["content"].strip()
                return text, latency
            except Exception as e:
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"Ollama query failed after {retries} attempts: {e}"
                    ) from e
                time.sleep(2 ** attempt)

        raise RuntimeError("Unreachable")


# ─────────────────────────────────────────────────────────────────────
# Base Task Module  (subclass this for each task)
# ─────────────────────────────────────────────────────────────────────

class BaseTaskModule:
    """
    Override these methods in your task-specific module:

      load_dataset(path, fmt)  → list of sample dicts
      stratified_sample(...)   → subset of samples
      parse_response(text)     → normalized prediction
      evaluate(pred, sample)   → dict of metrics for this prediction
      aggregate_results(rows)  → dict of summary metrics
      input_label              → e.g. "Input (C/C++ code)"
    """

    # Override in subclass
    TASK_NAME: str = "base"
    INPUT_LABEL: str = "Input"

    def load_dataset(self, path: str, fmt: str) -> list[dict]:
        raise NotImplementedError

    def stratified_sample(
        self,
        dataset: list[dict],
        n: int,
        seed: int = 42,
    ) -> list[dict]:
        """Default: random sample (no stratification)."""
        random.seed(seed)
        if n <= 0 or n >= len(dataset):
            return dataset
        return random.sample(dataset, n)

    def parse_response(self, raw_text: str) -> Any:
        """Convert raw model output to a prediction value."""
        return raw_text.strip()

    def evaluate(self, prediction: Any, sample: dict) -> dict[str, Any]:
        """
        Return a dict of per-sample metrics.
        Must include at least: {"correct": 0|1} for compatibility.
        """
        raise NotImplementedError

    def aggregate_results(self, rows: list[dict]) -> dict[str, Any]:
        """
        Compute summary statistics from all result rows.
        Default: accuracy.
        """
        if not rows:
            return {"accuracy": 0.0, "total": 0}
        total = len(rows)
        correct = sum(r.get("correct", 0) for r in rows)
        return {"accuracy": correct / total, "total": total, "correct": correct}

    def get_input_for_sample(self, sample: dict) -> str:
        """Extract the text to be inserted into the prompt."""
        # Common defaults — override if your schema differs
        for key in ("code", "text", "input", "content", "sentence", "document"):
            if key in sample:
                return str(sample[key])
        raise KeyError(
            f"Cannot auto-detect input field in sample. Keys: {list(sample.keys())}. "
            "Override get_input_for_sample() in your task module."
        )


# ─────────────────────────────────────────────────────────────────────
# Task Registry
# ─────────────────────────────────────────────────────────────────────

_TASK_REGISTRY: dict[str, type[BaseTaskModule]] = {}


def register_task(name: str) -> Callable:
    def decorator(cls: type[BaseTaskModule]) -> type[BaseTaskModule]:
        _TASK_REGISTRY[name] = cls
        return cls
    return decorator


def load_task(task_name: str) -> BaseTaskModule:
    """Load a task module by name (from registry or tasks/ directory)."""
    if task_name in _TASK_REGISTRY:
        return _TASK_REGISTRY[task_name]()

    # Try to import from tasks/<task_name>.py
    tasks_dir = Path(__file__).parent / "tasks"
    module_path = tasks_dir / f"{task_name}.py"

    if not module_path.exists():
        available = [f.stem for f in tasks_dir.glob("*.py") if not f.stem.startswith("_")]
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"Available: {available}. "
            f"Create tasks/{task_name}.py to add a new task."
        )

    spec = importlib.util.spec_from_file_location(f"tasks.{task_name}", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Find the subclass
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseTaskModule)
            and obj is not BaseTaskModule
        ):
            return obj()

    raise ValueError(f"No BaseTaskModule subclass found in tasks/{task_name}.py")


# ─────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────

class PipelineRunner:
    """Orchestrates the full evaluation pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.task = load_task(config.task_name)
        self.library = PromptLibrary(
            config.prompts_dir,
            config.prompt_sections or None,
        )
        self.combo_gen = CombinationGenerator(self.library, config)
        self.builder = PromptBuilder(self.library, self.task.INPUT_LABEL)
        self.client = OllamaClient(
            config.model_name,
            config.temperature,
            config.max_tokens,
        )

        os.makedirs(config.output_dir, exist_ok=True)
        if config.track_emissions:
            os.makedirs(config.carbon_output_dir, exist_ok=True)

    # ── Dataset Loading ──────────────────────────────────────────────

    def load_data(self) -> list[dict]:
        print(f"\n[Dataset] Loading from: {self.config.dataset_path}")
        dataset = self.task.load_dataset(
            self.config.dataset_path,
            self.config.dataset_format,
        )
        print(f"[Dataset] Loaded {len(dataset)} samples.")

        if self.config.sample_size and self.config.sample_size < len(dataset):
            dataset = self.task.stratified_sample(
                dataset,
                self.config.sample_size,
                self.config.seed,
            )
            print(f"[Dataset] Sampled down to {len(dataset)} samples.")
        return dataset

    # ── Combo Generation ─────────────────────────────────────────────

    def get_combos(self) -> list[dict]:
        combos = self.combo_gen.generate()
        print(f"\n[Combos] Mode: {self.config.combo_mode} | Count: {len(combos)}")
        print(f"[Prompt Library] Sections: {self.library.summary()}")
        return combos

    # ── Core Evaluation Loop ─────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """
        Execute the full pipeline.
        Returns a result dict with: rows, summary, emissions, output_path.
        """
        errors_config = self.config.validate()
        if errors_config:
            raise ValueError("Config validation failed:\n" + "\n".join(errors_config))

        dataset = self.load_data()
        combos = self.get_combos()

        total_calls = len(dataset) * len(combos)
        print(f"\n[Pipeline] Starting: {len(dataset)} samples × {len(combos)} combos = {total_calls} calls")
        print(f"[Model] {self.config.model_name}\n")

        tracker = None
        if self.config.track_emissions:
            try:
                from codecarbon import EmissionsTracker
                tracker = EmissionsTracker(
                    project_name=f"trove_{self.config.task_name}",
                    output_dir=self.config.carbon_output_dir,
                    log_level="error",
                )
                tracker.start()
            except Exception as e:
                print(f"[Warning] CodeCarbon unavailable: {e}. Skipping emissions tracking.")

        rows = []
        errors = 0

        with tqdm(total=total_calls, desc="Inference", unit="call") as pbar:
            for sample in dataset:
                sample_input = self.task.get_input_for_sample(sample)

                for combo in combos:
                    prompt = self.builder.build(sample_input, combo)

                    try:
                        raw_text, latency = self.client.query(prompt)
                        prediction = self.task.parse_response(raw_text)
                        metrics = self.task.evaluate(prediction, sample)
                    except Exception as e:
                        raw_text = f"ERROR: {e}"
                        prediction = None
                        metrics = {"correct": 0, "error": str(e)}
                        errors += 1

                    row = {
                        "model": self.config.model_name,
                        "task": self.config.task_name,
                        "prompt_combo": _combo_label(combo),
                        "prompt_text": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                        "raw_response": raw_text,
                        "prediction": str(prediction),
                        "latency_sec": round(latency if "latency" in dir() else 0, 4),
                        **metrics,
                    }
                    # Add sample ground-truth fields (exclude raw input to keep CSV lean)
                    for k, v in sample.items():
                        if k not in ("code", "text", "input", "content"):
                            row[f"gt_{k}"] = v

                    rows.append(row)
                    pbar.update(1)

        emissions = None
        if tracker:
            try:
                emissions = tracker.stop()
            except Exception:
                pass

        summary = self.task.aggregate_results(rows)
        if emissions is not None:
            summary["emissions_kgCO2"] = emissions

        output_path = self._save_results(rows, summary, emissions)

        return {
            "rows": rows,
            "summary": summary,
            "emissions": emissions,
            "output_path": output_path,
            "errors": errors,
        }

    # ── Saving ───────────────────────────────────────────────────────

    def _save_results(
        self,
        rows: list[dict],
        summary: dict[str, Any],
        emissions: Optional[float],
    ) -> str:
        output_file = self.config.output_file or (
            f"{self.config.task_name}_{self.config.model_name.replace(':', '_')}_{_ts()}.csv"
        )
        output_path = Path(self.config.output_dir) / output_file

        df = pd.DataFrame(rows)
        for k, v in summary.items():
            df[f"summary_{k}"] = v
        df.to_csv(output_path, index=False)

        # Also save summary JSON
        summary_path = output_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self._print_summary(summary, output_path, rows)
        return str(output_path)

    @staticmethod
    def _print_summary(
        summary: dict[str, Any],
        output_path: Path,
        rows: list[dict],
    ) -> None:
        print("\n" + "═" * 55)
        print("  EVALUATION SUMMARY")
        print("═" * 55)
        for k, v in summary.items():
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                print(f"  {label:<28} {v:.4f}")
            else:
                print(f"  {label:<28} {v}")

        # Per-combo breakdown
        df = pd.DataFrame(rows)
        if "correct" in df.columns and "prompt_combo" in df.columns:
            breakdown = (
                df.groupby("prompt_combo")["correct"]
                .mean()
                .reset_index()
                .rename(columns={"correct": "accuracy"})
                .sort_values("accuracy", ascending=False)
            )
            print("\n  Top 5 Combos by Accuracy:")
            print("  " + "─" * 45)
            for _, row in breakdown.head(5).iterrows():
                print(f"  {str(row['prompt_combo']):<35} {row['accuracy']:.3f}")

        print("═" * 55)
        print(f"  Results saved → {output_path}")
        print("═" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────

def _combo_label(combo: dict[str, Optional[str]]) -> str:
    parts = [f"{s}={v}" for s, v in combo.items() if v is not None]
    return "|".join(parts) if parts else "empty"


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def list_available_tasks() -> list[str]:
    tasks_dir = Path(__file__).parent / "tasks"
    return [f.stem for f in tasks_dir.glob("*.py") if not f.stem.startswith("_")]


def list_available_prompts(prompts_dir: str) -> dict[str, int]:
    lib = PromptLibrary(prompts_dir)
    return lib.summary()
