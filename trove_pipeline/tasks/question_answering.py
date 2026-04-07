"""
Task: Question Answering
────────────────────────
Evaluates open-domain or closed-domain QA.

Dataset schema:
  "question"  → the question to ask
  "answer"    → ground truth answer (string or list)
  "context"   → optional context passage

Metrics:
  - exact_match (EM)
  - token_overlap_f1 (F1 over answer tokens)
  - contains_answer (soft match)
"""

from __future__ import annotations

import json
import re
import random
from typing import Any

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import BaseTaskModule, register_task


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^a-z0-9 ]', '', s)
    return ' '.join(s.split())


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens) if pred_tokens else 0.0
    rec  = len(common) / len(gold_tokens) if gold_tokens else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


@register_task("question_answering")
class QuestionAnsweringTask(BaseTaskModule):

    TASK_NAME = "question_answering"
    INPUT_LABEL = "Question"

    Q_COLUMNS       = ["question", "query", "input", "prompt"]
    ANSWER_COLUMNS  = ["answer", "answers", "label", "target", "gold"]
    CONTEXT_COLUMNS = ["context", "passage", "paragraph", "document"]

    def load_dataset(self, path: str, fmt: str) -> list[dict]:
        fmt = fmt.lower().strip(".")
        if fmt == "jsonl":
            return self._load_jsonl(path)
        elif fmt in ("csv", "tsv"):
            return self._load_csv(path, "\t" if fmt == "tsv" else ",")
        elif fmt == "json":
            with open(path) as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("data", [])
            return [self._extract(i) for i in items if isinstance(i, dict)]
        raise ValueError(f"Unsupported format: {fmt}")

    def _load_jsonl(self, path: str) -> list[dict]:
        dataset = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    dataset.append(self._extract(json.loads(line)))
                except Exception:
                    continue
        print(f"  Loaded {len(dataset)} QA samples.")
        return dataset

    def _load_csv(self, path: str, sep: str = ",") -> list[dict]:
        df = pd.read_csv(path, sep=sep)
        dataset = [self._extract(dict(row)) for _, row in df.iterrows()]
        print(f"  Loaded {len(dataset)} QA samples.")
        return dataset

    def _extract(self, obj: dict) -> dict:
        question = next((str(obj[c]) for c in self.Q_COLUMNS if c in obj), "")
        answer_raw = next((obj[c] for c in self.ANSWER_COLUMNS if c in obj), "")
        # Handle list-style answers (SQuAD format)
        if isinstance(answer_raw, list):
            answer = answer_raw[0] if answer_raw else ""
        elif isinstance(answer_raw, dict):
            answer = answer_raw.get("text", str(answer_raw))
        else:
            answer = str(answer_raw) if answer_raw else ""
        context = next((str(obj[c]) for c in self.CONTEXT_COLUMNS if c in obj), "")
        return {"question": question, "answer": answer, "context": context}

    def get_input_for_sample(self, sample: dict) -> str:
        if sample.get("context"):
            return f"Context: {sample['context']}\n\nQuestion: {sample['question']}"
        return sample["question"]

    def parse_response(self, raw_text: str) -> str:
        return raw_text.strip()

    def evaluate(self, prediction: Any, sample: dict) -> dict[str, Any]:
        pred = str(prediction).strip()
        gold = str(sample.get("answer", "")).strip()

        em = int(_normalize_answer(pred) == _normalize_answer(gold))
        f1 = _token_f1(pred, gold)
        contains = int(_normalize_answer(gold) in _normalize_answer(pred)) if gold else 0

        return {
            "true_label": gold,
            "prediction": pred[:300],
            "correct": em,
            "exact_match": em,
            "token_f1": round(f1, 4),
            "contains_answer": contains,
        }

    def aggregate_results(self, rows: list[dict]) -> dict[str, Any]:
        if not rows:
            return {}
        n = len(rows)
        return {
            "exact_match":        round(sum(r.get("exact_match", 0) for r in rows) / n, 4),
            "avg_token_f1":       round(sum(r.get("token_f1", 0) for r in rows) / n, 4),
            "contains_answer_rate": round(sum(r.get("contains_answer", 0) for r in rows) / n, 4),
            "total_predictions":  n,
        }
