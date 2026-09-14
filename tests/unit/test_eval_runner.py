"""H3: EvalRunner 与 Golden Test Set 测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.types import RetrievalResult
from libs.evaluator.base_evaluator import BaseEvaluator
from observability.evaluation.eval_runner import EvalRunner


def _result(chunk_id: str, source: str = "docs/a.md") -> RetrievalResult:
    return RetrievalResult(chunk_id, 1.0, f"text for {chunk_id}", {"source_path": source})


class _Hybrid:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        return [_result("chunk-a"), _result("chunk-b")]


class _Evaluator(BaseEvaluator):
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], Any]] = []

    def evaluate(self, query, retrieved_ids, golden_ids, trace=None):
        self.calls.append((query, retrieved_ids, trace))
        return {"quality": 0.8}


def _write_set(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {"query": "first", "expected_chunk_ids": ["chunk-b"], "top_k": 2},
                    {"query": "second", "expected_chunk_ids": ["missing"]},
                ]
            }
        ),
        encoding="utf-8",
    )


def test_runner_returns_aggregate_and_case_metrics(tmp_path: Path) -> None:
    test_set = tmp_path / "golden.json"
    _write_set(test_set)
    hybrid = _Hybrid()
    evaluator = _Evaluator()
    runner = EvalRunner(SimpleNamespace(retrieval=SimpleNamespace(top_k=5)), hybrid, evaluator)

    report = runner.run(test_set)

    assert report.metrics == {"hit_rate": 0.5, "mrr": 0.25, "quality": 0.8}
    assert report.hit_rate == 0.5
    assert report.mrr == 0.25
    assert report.case_results[0].retrieved_chunk_ids == ["chunk-a", "chunk-b"]
    assert report.case_results[0].hit is True
    assert hybrid.calls == [("first", 2), ("second", 5)]
    assert evaluator.calls[0][2]["contexts"] == ["text for chunk-a", "text for chunk-b"]


def test_runner_validates_golden_set(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"test_cases": [{"query": ""}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="query"):
        EvalRunner._load_test_cases(path)


def test_runner_reports_missing_test_set(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="不存在"):
        EvalRunner._load_test_cases(tmp_path / "missing.json")
