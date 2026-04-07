"""
Task: Security / Vulnerability Detection
─────────────────────────────────────────
Detects binary vulnerability label (0 = safe, 1 = vulnerable).
Supports: PrimeVul (JSONL), and any CSV/TSV with configurable columns.

Dataset schemas supported:
  JSONL: {"func": "<code>", "target": 0|1, ...}
  CSV:   columns configurable via LABEL_COLUMN / CODE_COLUMN env vars
         or by subclassing and overriding.

Metrics computed:
  - Accuracy, Precision, Recall, F1 (macro + per-class)
  - False Positive Rate, False Negative Rate
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any

import pandas as pd

# Import base from parent package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import BaseTaskModule, register_task


@register_task("security_detection")
class SecurityDetectionTask(BaseTaskModule):
    """
    Binary vulnerability classification task.

    Sample dict keys expected:
      "code"  → the function/code snippet (str)
      "label" → 0 (non-vulnerable) or 1 (vulnerable) (int)
    """

    TASK_NAME = "security_detection"
    INPUT_LABEL = "Input (C/C++ code)"

    # ── Dataset columns (override or pass via config) ──────────────
    CODE_COLUMN = "func"      # JSONL/CSV column for code
    LABEL_COLUMN = "target"   # JSONL/CSV column for label
    # For CSV datasets, you may have different column names:
    ALT_CODE_COLUMNS = ["code", "function", "snippet", "source"]
    ALT_LABEL_COLUMNS = ["label", "vulnerable", "is_vulnerable", "vul"]

    # ── Dataset Loader ─────────────────────────────────────────────

    def load_dataset(self, path: str, fmt: str) -> list[dict]:
        fmt = fmt.lower().strip(".")
        if fmt == "jsonl":
            return self._load_jsonl(path)
        elif fmt in ("csv", "tsv"):
            sep = "\t" if fmt == "tsv" else ","
            return self._load_csv(path, sep)
        elif fmt == "json":
            return self._load_json(path)
        else:
            raise ValueError(f"Unsupported format '{fmt}' for security_detection. Use: jsonl, csv, tsv, json")

    def _load_jsonl(self, path: str) -> list[dict]:
        dataset = []
        skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  [WARN] Line {line_num}: JSON parse error — {e}")
                    skipped += 1
                    continue

                code = obj.get(self.CODE_COLUMN)
                label = obj.get(self.LABEL_COLUMN)

                if code is None or label is None:
                    skipped += 1
                    continue

                dataset.append({"code": str(code), "label": int(label)})

        print(f"  Loaded {len(dataset)} samples ({skipped} skipped) from JSONL.")
        return dataset

    def _load_csv(self, path: str, sep: str = ",") -> list[dict]:
        df = pd.read_csv(path, sep=sep)
        code_col = self._find_column(df.columns, [self.CODE_COLUMN] + self.ALT_CODE_COLUMNS)
        label_col = self._find_column(df.columns, [self.LABEL_COLUMN] + self.ALT_LABEL_COLUMNS)
        dataset = [
            {"code": str(row[code_col]), "label": int(row[label_col])}
            for _, row in df.iterrows()
            if pd.notna(row[code_col]) and pd.notna(row[label_col])
        ]
        print(f"  Loaded {len(dataset)} samples from CSV (code='{code_col}', label='{label_col}').")
        return dataset

    def _load_json(self, path: str) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return self._normalize_list(data)
        raise ValueError("JSON file must contain a top-level list of objects.")

    def _normalize_list(self, data: list[dict]) -> list[dict]:
        result = []
        for obj in data:
            code = None
            for c in [self.CODE_COLUMN] + self.ALT_CODE_COLUMNS:
                if c in obj:
                    code = obj[c]
                    break
            label = None
            for l in [self.LABEL_COLUMN] + self.ALT_LABEL_COLUMNS:
                if l in obj:
                    label = obj[l]
                    break
            if code is not None and label is not None:
                result.append({"code": str(code), "label": int(label)})
        return result

    @staticmethod
    def _find_column(columns, candidates: list[str]) -> str:
        for c in candidates:
            if c in columns:
                return c
        raise KeyError(
            f"No matching column found. Tried: {candidates}. "
            f"Available: {list(columns)}"
        )

    # ── Stratified Sampling ────────────────────────────────────────

    def stratified_sample(self, dataset: list[dict], n: int, seed: int = 42) -> list[dict]:
        random.seed(seed)
        vuln = [s for s in dataset if s["label"] == 1]
        safe = [s for s in dataset if s["label"] == 0]

        half = n // 2
        n_vuln = min(half, len(vuln))
        n_safe = min(n - n_vuln, len(safe))
        n_vuln = min(n - n_safe, len(vuln))  # compensate if one class short

        sampled = random.sample(vuln, n_vuln) + random.sample(safe, n_safe)
        random.shuffle(sampled)
        print(f"  Stratified sample: {n_vuln} vulnerable + {n_safe} non-vulnerable = {len(sampled)} total")
        return sampled

    # ── Inference ─────────────────────────────────────────────────

    def get_input_for_sample(self, sample: dict) -> str:
        return sample["code"]

    def parse_response(self, raw_text: str) -> str:
        """Extract 0 or 1 from model output."""
        cleaned = raw_text.strip()
        if "1" in cleaned:
            return "1"
        return "0"

    # ── Per-sample Evaluation ─────────────────────────────────────

    def evaluate(self, prediction: Any, sample: dict) -> dict[str, Any]:
        true_label = str(sample["label"])
        pred_str = str(prediction)
        correct = int(pred_str == true_label)
        return {
            "true_label": true_label,
            "prediction": pred_str,
            "correct": correct,
            "true_positive":  int(pred_str == "1" and true_label == "1"),
            "false_positive": int(pred_str == "1" and true_label == "0"),
            "true_negative":  int(pred_str == "0" and true_label == "0"),
            "false_negative": int(pred_str == "0" and true_label == "1"),
        }

    # ── Aggregate Metrics ─────────────────────────────────────────

    def aggregate_results(self, rows: list[dict]) -> dict[str, Any]:
        if not rows:
            return {}

        total = len(rows)
        correct = sum(r.get("correct", 0) for r in rows)
        tp = sum(r.get("true_positive", 0) for r in rows)
        fp = sum(r.get("false_positive", 0) for r in rows)
        tn = sum(r.get("true_negative", 0) for r in rows)
        fn = sum(r.get("false_negative", 0) for r in rows)

        accuracy   = correct / total
        precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = TPR
        f1         = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
        fpr        = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr        = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return {
            "accuracy":   round(accuracy, 4),
            "precision":  round(precision, 4),
            "recall":     round(recall, 4),
            "f1_score":   round(f1, 4),
            "fpr":        round(fpr, 4),
            "fnr":        round(fnr, 4),
            "true_positives":  tp,
            "false_positives": fp,
            "true_negatives":  tn,
            "false_negatives": fn,
            "total_predictions": total,
        }
