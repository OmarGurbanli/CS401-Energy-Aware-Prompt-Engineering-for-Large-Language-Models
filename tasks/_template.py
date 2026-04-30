"""
TROVE Pipeline — Custom Task Template
══════════════════════════════════════
Copy this file to tasks/your_task_name.py and fill in each section.

Steps:
  1. Rename the class and set TASK_NAME / INPUT_LABEL
  2. Implement load_dataset() for your data format
  3. Implement parse_response() to extract the prediction from raw model output
  4. Implement evaluate() to score each prediction against ground truth
  5. Optionally override aggregate_results() for custom summary metrics
  6. Create prompts/your_task_name/ with CSV section files (T.csv, R.csv, etc.)
  7. Create configs/your_task_name.yaml
  8. Run: python run.py --task your_task_name --dataset yourdata.csv
"""

from __future__ import annotations

import json
import random
from typing import Any

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import BaseTaskModule, register_task


@register_task("custom_task")          # ← change "custom_task" to your task name
class CustomTask(BaseTaskModule):      # ← rename this class

    # ── Identity ─────────────────────────────────────────────────
    TASK_NAME   = "custom_task"        # ← must match register_task name
    INPUT_LABEL = "Input"              # ← label shown before input in prompt
                                       #   e.g. "Question", "Code", "Sentence"

    # ── Dataset Loading ──────────────────────────────────────────

    def load_dataset(self, path: str, fmt: str) -> list[dict]:
        """
        Load your dataset and return a list of sample dicts.

        Each dict must have at minimum:
          - whatever field you return from get_input_for_sample()
          - whatever fields you read in evaluate()

        Example for a CSV with columns "text", "label":
        """
        fmt = fmt.lower().strip(".")

        if fmt == "jsonl":
            dataset = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        dataset.append({
                            "text":  obj["text"],    # ← adapt field names
                            "label": obj["label"],
                        })
                    except (KeyError, json.JSONDecodeError):
                        continue
            print(f"  Loaded {len(dataset)} samples.")
            return dataset

        elif fmt in ("csv", "tsv"):
            sep = "\t" if fmt == "tsv" else ","
            df = pd.read_csv(path, sep=sep)
            dataset = [
                {"text": str(row["text"]), "label": str(row["label"])}
                for _, row in df.iterrows()
            ]
            print(f"  Loaded {len(dataset)} samples.")
            return dataset

        else:
            raise ValueError(f"Unsupported format '{fmt}' for custom_task.")

    # ── Stratified Sampling (optional override) ───────────────────

    def stratified_sample(self, dataset: list[dict], n: int, seed: int = 42) -> list[dict]:
        """
        Optional: override if you want class-balanced sampling.
        Default (from BaseTaskModule) is plain random sample.
        """
        # Example: use parent default (random sample)
        return super().stratified_sample(dataset, n, seed)

    # ── Input Extraction ─────────────────────────────────────────

    def get_input_for_sample(self, sample: dict) -> str:
        """Return the text that gets inserted into the prompt."""
        return sample["text"]   # ← adapt to your sample dict

    # ── Response Parsing ─────────────────────────────────────────

    def parse_response(self, raw_text: str) -> Any:
        """
        Convert raw model output string to a prediction value.

        Examples:
          Binary:     return "1" if "1" in raw_text else "0"
          Label:      return raw_text.strip().lower().split()[0]
          Number:     import re; m = re.search(r'\d+', raw_text); return int(m.group()) if m else 0
          Free-text:  return raw_text.strip()
        """
        return raw_text.strip()   # ← replace with your logic

    # ── Per-Sample Evaluation ─────────────────────────────────────

    def evaluate(self, prediction: Any, sample: dict) -> dict[str, Any]:
        """
        Score the prediction against the ground truth in sample.

        Must return a dict that includes at least {"correct": 0|1}.
        Add any per-sample metrics you need; they will appear in the CSV.
        """
        true_label = str(sample["label"]).strip()
        pred       = str(prediction).strip()
        correct    = int(pred == true_label)

        return {
            "true_label": true_label,
            "prediction": pred,
            "correct":    correct,
            # add more metrics here:
            # "char_overlap": ...
        }

    # ── Aggregate Metrics ─────────────────────────────────────────

    def aggregate_results(self, rows: list[dict]) -> dict[str, Any]:
        """
        Compute summary statistics from all result rows.
        Override to add task-specific aggregate metrics.
        Default: accuracy.
        """
        if not rows:
            return {}
        n       = len(rows)
        correct = sum(r.get("correct", 0) for r in rows)
        return {
            "accuracy":          round(correct / n, 4),
            "total_predictions": n,
            # add more summary metrics here
        }
