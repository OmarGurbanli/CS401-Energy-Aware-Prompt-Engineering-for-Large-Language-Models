"""
Task: Code Generation / Quality Assessment
──────────────────────────────────────────
Evaluates code generation quality using LLM-as-judge or
keyword/pattern matching against expected outputs.

Suitable for: code completion, code summarization,
              docstring generation, bug fixing quality.

Dataset schema (CSV/JSONL):
  "prompt"     → the coding task description
  "reference"  → expected/reference output (optional)
  "language"   → programming language (optional)

Metrics:
  - exact_match     : exact string match with reference
  - contains_code   : whether response contains code-like content
  - keyword_match   : % of expected keywords found in response
  - response_length : average token length of responses
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import BaseTaskModule, register_task


@register_task("code_generation")
class CodeGenerationTask(BaseTaskModule):

    TASK_NAME = "code_generation"
    INPUT_LABEL = "Task"

    PROMPT_COLUMNS    = ["prompt", "instruction", "task", "question", "input"]
    REFERENCE_COLUMNS = ["reference", "expected", "answer", "output", "solution", "canonical_solution"]
    LANG_COLUMNS      = ["language", "lang", "programming_language"]

    CODE_PATTERNS = [
        r"def\s+\w+",          # Python functions
        r"function\s+\w+",     # JS functions
        r"void\s+\w+\s*\(",    # C/Java void
        r"int\s+\w+\s*\(",     # C/Java int
        r"class\s+\w+",        # class definitions
        r"return\s+",          # return statements
        r"for\s*\(", r"while\s*\(",  # loops
        r"```",                # markdown code fences
    ]

    def load_dataset(self, path: str, fmt: str) -> list[dict]:
        fmt = fmt.lower().strip(".")
        if fmt == "jsonl":
            return self._load_jsonl(path)
        elif fmt in ("csv", "tsv"):
            return self._load_csv(path, "\t" if fmt == "tsv" else ",")
        elif fmt == "json":
            with open(path) as f:
                data = json.load(f)
            return self._normalize(data if isinstance(data, list) else [data])
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
                    dataset.append(self._extract(obj))
                except Exception:
                    continue
        print(f"  Loaded {len(dataset)} samples from JSONL.")
        return dataset

    def _load_csv(self, path: str, sep: str = ",") -> list[dict]:
        df = pd.read_csv(path, sep=sep)
        dataset = [self._extract(dict(row)) for _, row in df.iterrows()]
        print(f"  Loaded {len(dataset)} samples from CSV.")
        return dataset

    def _normalize(self, data: list) -> list[dict]:
        return [self._extract(obj) for obj in data if isinstance(obj, dict)]

    def _extract(self, obj: dict) -> dict:
        prompt = next((str(obj[c]) for c in self.PROMPT_COLUMNS if c in obj and pd.notna(obj[c])), "")
        reference = next((str(obj[c]) for c in self.REFERENCE_COLUMNS if c in obj and pd.notna(obj.get(c))), None)
        language = next((str(obj[c]) for c in self.LANG_COLUMNS if c in obj and pd.notna(obj.get(c))), "unknown")
        return {"prompt": prompt, "reference": reference, "language": language}

    def get_input_for_sample(self, sample: dict) -> str:
        return sample["prompt"]

    def parse_response(self, raw_text: str) -> str:
        return raw_text.strip()

    def evaluate(self, prediction: Any, sample: dict) -> dict[str, Any]:
        pred = str(prediction).strip()
        reference = sample.get("reference") or ""

        exact_match = int(pred.lower() == reference.lower()) if reference else 0

        contains_code = int(
            any(re.search(p, pred, re.IGNORECASE) for p in self.CODE_PATTERNS)
        )

        # Keyword overlap with reference
        if reference:
            ref_tokens = set(re.findall(r"\b\w+\b", reference.lower()))
            pred_tokens = set(re.findall(r"\b\w+\b", pred.lower()))
            keyword_match = len(ref_tokens & pred_tokens) / len(ref_tokens) if ref_tokens else 0.0
        else:
            keyword_match = 0.0

        response_length = len(pred.split())

        return {
            "true_label": reference or "",
            "prediction": pred[:200],  # truncate for CSV
            "correct": exact_match,
            "exact_match": exact_match,
            "contains_code": contains_code,
            "keyword_match": round(keyword_match, 4),
            "response_length_tokens": response_length,
        }

    def aggregate_results(self, rows: list[dict]) -> dict[str, Any]:
        if not rows:
            return {}
        n = len(rows)
        return {
            "exact_match_rate":      round(sum(r.get("exact_match", 0) for r in rows) / n, 4),
            "code_presence_rate":    round(sum(r.get("contains_code", 0) for r in rows) / n, 4),
            "avg_keyword_match":     round(sum(r.get("keyword_match", 0) for r in rows) / n, 4),
            "avg_response_length":   round(sum(r.get("response_length_tokens", 0) for r in rows) / n, 1),
            "total_predictions":     n,
        }
