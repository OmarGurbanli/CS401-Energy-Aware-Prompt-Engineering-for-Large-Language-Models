"""
Task: Technical Debt Detection (Code Smell Classification)
────────────────────────────────────────────────────────────
Detects code smells from the MLCQ dataset.

Dataset labels (smell column):  blob | long method | feature envy | data class
Mapped internal labels:         god_class | long_method | feature_envy | data_class

NOTE: This dataset has NO "none" samples — every snippet has a smell.
The model may still output "none"; this just counts as a wrong prediction.

Metrics: accuracy, F1 macro, F1 per class
"""

from __future__ import annotations

import random
from typing import Any

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import BaseTaskModule, register_task


# Canonical internal label set
VALID_LABELS = {"god_class", "long_method", "data_class", "feature_envy", "none"}

# Maps raw CSV values → internal labels
LABEL_MAP = {
    "blob":         "god_class",
    "long method":  "long_method",
    "long_method":  "long_method",
    "feature envy": "feature_envy",
    "feature_envy": "feature_envy",
    "data class":   "data_class",
    "data_class":   "data_class",
    "none":         "none",
}


@register_task("technical_debt")
class TechnicalDebtTask(BaseTaskModule):

    TASK_NAME   = "technical_debt"
    INPUT_LABEL = "Input (Java code)"

    # ── Dataset Loading ───────────────────────────────────────────────

    def load_dataset(self, path: str, fmt: str) -> list[dict]:
        df = pd.read_csv(
            path,
            sep=";",
            quotechar='"',
            on_bad_lines="skip",
            engine="python",
        )

        code_col  = self._find_column(df.columns, ["code_snippet", "code", "snippet", "source"])
        label_col = self._find_column(df.columns, ["smell", "label", "code_smell"])

        dataset = []
        skipped = 0
        for _, row in df.iterrows():
            if pd.isna(row[code_col]) or pd.isna(row[label_col]):
                skipped += 1
                continue

            raw = str(row[label_col]).strip().lower()
            mapped = LABEL_MAP.get(raw)
            if mapped is None:
                # Unknown label — skip rather than silently mislabel
                skipped += 1
                continue

            dataset.append({
                "code":  str(row[code_col]),
                "label": mapped,
            })

        counts: dict[str, int] = {}
        for r in dataset:
            counts[r["label"]] = counts.get(r["label"], 0) + 1

        print(f"  Loaded {len(dataset)} samples ({skipped} skipped).")
        print(f"  Label distribution: {counts}")
        return dataset

    @staticmethod
    def _find_column(columns, candidates: list[str]) -> str:
        for c in candidates:
            if c in columns:
                return c
        raise KeyError(
            f"No matching column found. Tried: {candidates}. "
            f"Available columns: {list(columns)}"
        )

    # ── Stratified Sampling ───────────────────────────────────────────

    def stratified_sample(self, dataset: list[dict], n: int, seed: int = 42) -> list[dict]:
        if n <= 0 or n >= len(dataset):
            return dataset

        random.seed(seed)
        groups: dict[str, list[dict]] = {}
        for row in dataset:
            groups.setdefault(row["label"], []).append(row)

        per_class = n // len(groups)
        sampled: list[dict] = []

        for rows in groups.values():
            take = min(per_class, len(rows))
            sampled.extend(random.sample(rows, take))

        # Top up to exactly n if rounding left us short
        remainder = n - len(sampled)
        if remainder > 0:
            pool = [r for r in dataset if r not in sampled]
            sampled.extend(random.sample(pool, min(remainder, len(pool))))

        random.shuffle(sampled)
        return sampled

    # ── Input Extraction ─────────────────────────────────────────────

    def get_input_for_sample(self, sample: dict) -> str:
        return sample["code"]

    # ── Response Parsing ─────────────────────────────────────────────

    def parse_response(self, raw_text: str) -> str:
        """Map model free-text output to one of the 5 internal labels."""
        text = raw_text.strip().lower()

        # Check most specific patterns first to avoid false matches
        if any(x in text for x in ["god_class", "god class", "blob", "god-class"]):
            return "god_class"
        if any(x in text for x in ["long_method", "long method", "long-method"]):
            return "long_method"
        if any(x in text for x in ["data_class", "data class", "data-class"]):
            return "data_class"
        if any(x in text for x in ["feature_envy", "feature envy", "feature-envy"]):
            return "feature_envy"
        if any(x in text for x in ["none", "no smell", "clean", "no code smell"]):
            return "none"

        # Fallback — treat unrecognized output as "none"
        return "none"

    # ── Per-sample Evaluation ────────────────────────────────────────

    def evaluate(self, prediction: Any, sample: dict) -> dict[str, Any]:
        true_label = str(sample["label"]).strip().lower()
        pred       = str(prediction).strip().lower()
        correct    = int(pred == true_label)
        return {
            "true_label": true_label,
            "prediction": pred,
            "correct":    correct,
        }

    # ── Aggregate Metrics ────────────────────────────────────────────

    def aggregate_results(self, rows: list[dict]) -> dict[str, Any]:
        if not rows:
            return {}

        total    = len(rows)
        correct  = sum(r.get("correct", 0) for r in rows)
        accuracy = correct / total

        # F1 per class (macro average)
        f1_scores: dict[str, float] = {}
        for cls in sorted(VALID_LABELS):
            tp = sum(1 for r in rows if r["prediction"] == cls and r["true_label"] == cls)
            fp = sum(1 for r in rows if r["prediction"] == cls and r["true_label"] != cls)
            fn = sum(1 for r in rows if r["prediction"] != cls and r["true_label"] == cls)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            f1_scores[cls] = round(f1, 4)

        # Macro F1 only over classes that appear in ground truth
        active_classes = {r["true_label"] for r in rows}
        active_f1s = [f1_scores[c] for c in active_classes if c in f1_scores]
        f1_macro = round(sum(active_f1s) / len(active_f1s), 4) if active_f1s else 0.0

        return {
            "accuracy":          round(accuracy, 4),
            "f1_macro":          f1_macro,
            "f1_god_class":      f1_scores.get("god_class", 0.0),
            "f1_long_method":    f1_scores.get("long_method", 0.0),
            "f1_data_class":     f1_scores.get("data_class", 0.0),
            "f1_feature_envy":   f1_scores.get("feature_envy", 0.0),
            "f1_none":           f1_scores.get("none", 0.0),
            "total_predictions": total,
            "correct":           correct,
        }