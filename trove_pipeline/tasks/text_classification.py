"""
Task: Text Classification
─────────────────────────
Generic multi-class (or binary) text classification.
Suitable for: sentiment analysis, spam detection, topic classification,
              intent recognition, toxicity detection, etc.

Dataset schemas supported:
  CSV/TSV: configurable text + label columns
  JSONL: {"text": "...", "label": "positive"|0|...}

Metrics:
  - Accuracy (overall)
  - Per-class precision, recall, F1
  - Macro-averaged F1
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from typing import Any

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import BaseTaskModule, register_task


@register_task("text_classification")
class TextClassificationTask(BaseTaskModule):

    TASK_NAME = "text_classification"
    INPUT_LABEL = "Input (Text)"

    # Column name candidates (auto-detected)
    TEXT_COLUMNS  = ["text", "sentence", "content", "review", "document", "body", "message"]
    LABEL_COLUMNS = ["label", "class", "category", "sentiment", "target", "output"]

    def load_dataset(self, path: str, fmt: str) -> list[dict]:
        fmt = fmt.lower().strip(".")
        if fmt == "jsonl":
            return self._load_jsonl(path)
        elif fmt in ("csv", "tsv"):
            sep = "\t" if fmt == "tsv" else ","
            return self._load_csv(path, sep)
        elif fmt == "json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return self._normalize(data)
            raise ValueError("JSON must contain a list.")
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def _load_jsonl(self, path: str) -> list[dict]:
        dataset = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text, label = self._extract(obj)
                    dataset.append({"text": text, "label": label})
                except Exception:
                    continue
        print(f"  Loaded {len(dataset)} samples from JSONL.")
        return dataset

    def _load_csv(self, path: str, sep: str = ",") -> list[dict]:
        df = pd.read_csv(path, sep=sep)
        text_col  = self._find_col(df.columns, self.TEXT_COLUMNS)
        label_col = self._find_col(df.columns, self.LABEL_COLUMNS)
        dataset = [
            {"text": str(r[text_col]), "label": str(r[label_col])}
            for _, r in df.iterrows()
            if pd.notna(r[text_col]) and pd.notna(r[label_col])
        ]
        print(f"  Loaded {len(dataset)} samples from CSV.")
        return dataset

    def _normalize(self, data: list[dict]) -> list[dict]:
        result = []
        for obj in data:
            try:
                text, label = self._extract(obj)
                result.append({"text": text, "label": label})
            except Exception:
                continue
        return result

    def _extract(self, obj: dict) -> tuple[str, str]:
        text = next((str(obj[c]) for c in self.TEXT_COLUMNS if c in obj), None)
        label = next((str(obj[c]) for c in self.LABEL_COLUMNS if c in obj), None)
        if text is None or label is None:
            raise KeyError(f"Cannot find text/label columns in {list(obj.keys())}")
        return text, label

    @staticmethod
    def _find_col(columns, candidates: list[str]) -> str:
        for c in candidates:
            if c in columns:
                return c
        raise KeyError(f"No matching column. Tried: {candidates}. Found: {list(columns)}")

    def stratified_sample(self, dataset: list[dict], n: int, seed: int = 42) -> list[dict]:
        random.seed(seed)
        if n <= 0 or n >= len(dataset):
            return dataset

        # Group by label
        by_label: dict[str, list] = defaultdict(list)
        for s in dataset:
            by_label[str(s["label"])].append(s)

        classes = list(by_label.keys())
        per_class = n // len(classes)
        sampled = []
        for cls in classes:
            pool = by_label[cls]
            k = min(per_class, len(pool))
            sampled.extend(random.sample(pool, k))

        # Fill remainder
        remaining = n - len(sampled)
        all_remaining = [s for s in dataset if s not in sampled]
        sampled.extend(random.sample(all_remaining, min(remaining, len(all_remaining))))
        random.shuffle(sampled)

        label_counts = defaultdict(int)
        for s in sampled:
            label_counts[s["label"]] += 1
        print(f"  Stratified sample: {dict(label_counts)}")
        return sampled

    def get_input_for_sample(self, sample: dict) -> str:
        return sample["text"]

    def parse_response(self, raw_text: str) -> str:
        return raw_text.strip().lower().split()[0] if raw_text.strip() else ""

    def evaluate(self, prediction: Any, sample: dict) -> dict[str, Any]:
        true_label = str(sample["label"]).lower().strip()
        pred = str(prediction).lower().strip()
        correct = int(pred == true_label)
        return {
            "true_label": true_label,
            "prediction": pred,
            "correct": correct,
        }

    def aggregate_results(self, rows: list[dict]) -> dict[str, Any]:
        if not rows:
            return {}

        total = len(rows)
        correct = sum(r.get("correct", 0) for r in rows)

        # Per-class metrics
        classes = set(r["true_label"] for r in rows if "true_label" in r)
        per_class_metrics = {}
        f1_scores = []

        for cls in sorted(classes):
            tp = sum(1 for r in rows if r.get("true_label") == cls and r.get("prediction") == cls)
            fp = sum(1 for r in rows if r.get("true_label") != cls and r.get("prediction") == cls)
            fn = sum(1 for r in rows if r.get("true_label") == cls and r.get("prediction") != cls)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_scores.append(f1)
            per_class_metrics[f"class_{cls}_precision"] = round(prec, 4)
            per_class_metrics[f"class_{cls}_recall"]    = round(rec, 4)
            per_class_metrics[f"class_{cls}_f1"]        = round(f1, 4)

        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return {
            "accuracy":     round(correct / total, 4),
            "macro_f1":     round(macro_f1, 4),
            "total_predictions": total,
            **per_class_metrics,
        }
