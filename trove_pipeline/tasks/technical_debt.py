"""
Task: Technical Debt Detection (Code Smell Classification)
────────────────────────────────────────────────────────────
Detects code smells from the MLCQ dataset.

Labels: none | god_class | long_method | data_class | feature_envy
Output: multiclass classification (5 classes)

Metrics: accuracy, F1 macro, F1 per class
"""

from __future__ import annotations
import re
from typing import Any
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import BaseTaskModule, register_task


VALID_LABELS = {"none", "god_class", "long_method", "data_class", "feature_envy"}


@register_task("technical_debt")
class TechnicalDebtTask(BaseTaskModule):

    TASK_NAME   = "technical_debt"
    INPUT_LABEL = "Input (Java code)"

    # ── Dataset Loading ───────────────────────────────────────────

    def load_dataset(self, path: str, fmt: str) -> list[dict]:
        df = pd.read_csv(path, sep=';', quotechar='"')

        # Auto-detect code and label columns
        code_col  = self._find_column(df.columns, ["code_snippet", "code", "snippet", "source"])
        label_col = self._find_column(df.columns, ["smell", "label", "code_smell", "type"])

        dataset = []
        for _, row in df.iterrows():
            if pd.isna(row[code_col]) or pd.isna(row[label_col]):
                continue
            # Normalize label to lowercase with underscores
            raw_label = str(row[label_col]).strip().lower().replace(" ", "_").replace("-", "_")
            dataset.append({"code": str(row[code_col]), "label": raw_label})

        print(f"  Loaded {len(dataset)} samples from MLCQ dataset.")
        label_counts = {}
        for r in dataset:
            label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        print(f"  Label distribution: {label_counts}")
        return dataset

    @staticmethod
    def _find_column(columns, candidates):
        for c in candidates:
            if c in columns:
                return c
        raise KeyError(f"No matching column. Tried: {candidates}. Available: {list(columns)}")

    # ── Stratified Sampling ───────────────────────────────────────

    def stratified_sample(self, dataset: list[dict], n: int, seed: int = 42) -> list[dict]:
        import random
        random.seed(seed)
        groups = {}
        for row in dataset:
            groups.setdefault(row["label"], []).append(row)
        per_class = n // len(groups)
        sampled = []
        for cls, rows in groups.items():
            take = min(per_class, len(rows))
            sampled.extend(random.sample(rows, take))
        # Fill remainder
        remainder = n - len(sampled)
        if remainder > 0:
            pool = [r for r in dataset if r not in sampled]
            sampled.extend(random.sample(pool, min(remainder, len(pool))))
        random.shuffle(sampled)
        return sampled

    # ── Input & Response ──────────────────────────────────────────

    def get_input_for_sample(self, sample: dict) -> str:
        return sample["code"]

    def parse_response(self, raw_text: str) -> str:
        """Extract the smell label from model output."""
        text = raw_text.strip().lower()

        # Check for each label (order matters — check specific ones first)
        if "god_class" in text or "god class" in text or "blob" in text:
            return "god_class"
        if "long_method" in text or "long method" in text:
            return "long_method"
        if "data_class" in text or "data class" in text:
            return "data_class"
        if "feature_envy" in text or "feature envy" in text:
            return "feature_envy"
        if "none" in text or "no smell" in text or "clean" in text:
            return "none"
        # Default: none
        return "none"

    # ── Per-sample Evaluation ─────────────────────────────────────

    def evaluate(self, prediction: Any, sample: dict) -> dict[str, Any]:
        true_label = str(sample["label"]).strip().lower()
        pred       = str(prediction).strip().lower()
        correct    = int(pred == true_label)
        return {
            "true_label": true_label,
            "prediction": pred,
            "correct":    correct,
        }

    # ── Aggregate Metrics ─────────────────────────────────────────

    def aggregate_results(self, rows: list[dict]) -> dict[str, Any]:
        if not rows:
            return {}

        total   = len(rows)
        correct = sum(r.get("correct", 0) for r in rows)
        accuracy = correct / total

        # F1 per class
        classes = list(VALID_LABELS)
        f1_scores = {}
        for cls in classes:
            tp = sum(1 for r in rows if r["prediction"] == cls and r["true_label"] == cls)
            fp = sum(1 for r in rows if r["prediction"] == cls and r["true_label"] != cls)
            fn = sum(1 for r in rows if r["prediction"] != cls and r["true_label"] == cls)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            f1_scores[cls] = round(f1, 4)

        f1_macro = round(sum(f1_scores.values()) / len(f1_scores), 4)

        return {
            "accuracy":             round(accuracy, 4),
            "f1_macro":             f1_macro,
            "f1_none":              f1_scores["none"],
            "f1_god_class":         f1_scores["god_class"],
            "f1_long_method":       f1_scores["long_method"],
            "f1_data_class":        f1_scores["data_class"],
            "f1_feature_envy":      f1_scores["feature_envy"],
            "total_predictions":    total,
        }