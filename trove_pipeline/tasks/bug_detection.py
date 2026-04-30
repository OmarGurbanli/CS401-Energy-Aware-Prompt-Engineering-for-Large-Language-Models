from __future__ import annotations

import json
import random
from typing import Any

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import BaseTaskModule, register_task


@register_task("bug_detection")
class BugDetectionTask(BaseTaskModule):
    """
    Binary bug detection task.

    Labels:
      0 → buggy / incorrect
      1 → correct / fixed

    Expected sample format:
      "code"  → code snippet (str)
      "label" → 0 or 1
    """

    TASK_NAME = "bug_detection"
    INPUT_LABEL = "Input (Code)"

    # Column detection (compatible with QuixBugs-style CSV)
    CODE_COLUMN = "code"
    LABEL_COLUMN = "label"

    ALT_CODE_COLUMNS = ["func", "function", "snippet", "source"]
    ALT_LABEL_COLUMNS = ["target", "class", "output"]

    # ── Dataset Loading ──────────────────────────────────────────

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
            raise ValueError(f"Unsupported format: {fmt}")

    def _load_jsonl(self, path: str) -> list[dict]:
        dataset = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                code = self._get(obj, [self.CODE_COLUMN] + self.ALT_CODE_COLUMNS)
                label = self._get(obj, [self.LABEL_COLUMN] + self.ALT_LABEL_COLUMNS)
                if code is not None and label is not None:
                    dataset.append({"code": str(code), "label": int(label)})
        print(f"  Loaded {len(dataset)} samples from JSONL.")
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
        return [
            {"code": str(self._get(obj, [self.CODE_COLUMN] + self.ALT_CODE_COLUMNS)),
             "label": int(self._get(obj, [self.LABEL_COLUMN] + self.ALT_LABEL_COLUMNS))}
            for obj in data
            if self._get(obj, [self.CODE_COLUMN] + self.ALT_CODE_COLUMNS) is not None
        ]

    @staticmethod
    def _get(obj: dict, keys: list[str]):
        for k in keys:
            if k in obj:
                return obj[k]
        return None

    @staticmethod
    def _find_column(columns, candidates: list[str]) -> str:
        for c in candidates:
            if c in columns:
                return c
        raise KeyError(f"No matching column found. Tried: {candidates}, got: {list(columns)}")

    # ── Sampling ────────────────────────────────────────────────

    def stratified_sample(self, dataset: list[dict], n: int, seed: int = 42) -> list[dict]:
        random.seed(seed)
        buggy = [s for s in dataset if s["label"] == 0]
        correct = [s for s in dataset if s["label"] == 1]

        half = n // 2
        n_buggy = min(half, len(buggy))
        n_correct = min(n - n_buggy, len(correct))
        n_buggy = min(n - n_correct, len(buggy))

        sampled = random.sample(buggy, n_buggy) + random.sample(correct, n_correct)
        random.shuffle(sampled)

        print(f"  Stratified sample: {n_buggy} buggy + {n_correct} correct = {len(sampled)}")
        return sampled

    # ── Inference ───────────────────────────────────────────────

    def get_input_for_sample(self, sample: dict) -> str:
        return sample["code"]

    def parse_response(self, raw_text: str) -> str:
        """
        Expect model to output '0' or '1'.
        More strict than security_detection to avoid false positives.
        """
        cleaned = raw_text.strip().lower()

        # direct match
        if cleaned in ("0", "1"):
            return cleaned

        # fallback: first token
        token = cleaned.split()[0] if cleaned else ""
        if token in ("0", "1"):
            return token

        return ""  # invalid / unparseable

    # ── Evaluation ──────────────────────────────────────────────

    def evaluate(self, prediction: Any, sample: dict) -> dict[str, Any]:
        true_label = str(sample["label"])
        pred = str(prediction)

        correct = int(pred == true_label)

        return {
            "true_label": true_label,
            "prediction": pred,
            "correct": correct,
            "true_positive":  int(pred == "1" and true_label == "1"),
            "false_positive": int(pred == "1" and true_label == "0"),
            "true_negative":  int(pred == "0" and true_label == "0"),
            "false_negative": int(pred == "0" and true_label == "1"),
        }

    # ── Metrics ─────────────────────────────────────────────────

    def aggregate_results(self, rows: list[dict]) -> dict[str, Any]:
        if not rows:
            return {}

        total = len(rows)
        correct = sum(r.get("correct", 0) for r in rows)

        tp = sum(r.get("true_positive", 0) for r in rows)
        fp = sum(r.get("false_positive", 0) for r in rows)
        tn = sum(r.get("true_negative", 0) for r in rows)
        fn = sum(r.get("false_negative", 0) for r in rows)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0.0)

        return {
            "accuracy": round(correct / total, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "total_predictions": total,
        }