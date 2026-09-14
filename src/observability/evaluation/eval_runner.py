"""Run retrieval evaluation against a small, versioned golden test set."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator


@dataclass(frozen=True)
class EvalCaseResult:
    """The retrieval and evaluator output for one golden test case."""

    query: str
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    expected_sources: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def hit(self) -> bool:
        return self.metrics.get("hit_rate", 0.0) > 0.0

    @property
    def mrr(self) -> float:
        return self.metrics.get("mrr", 0.0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hit"] = self.hit
        data["mrr"] = self.mrr
        return data


@dataclass(frozen=True)
class EvalReport:
    """Aggregated evaluation metrics and per-query details."""

    metrics: dict[str, float]
    case_results: list[EvalCaseResult]

    @property
    def hit_rate(self) -> float:
        return self.metrics.get("hit_rate", 0.0)

    @property
    def mrr(self) -> float:
        return self.metrics.get("mrr", 0.0)

    @property
    def results(self) -> list[EvalCaseResult]:
        """Backward-friendly alias for consumers that call them query results."""
        return self.case_results

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "case_results": [case.to_dict() for case in self.case_results],
        }


class EvalRunner:
    """Connect a HybridSearch instance to one or more evaluator backends."""

    def __init__(self, settings: Any, hybrid_search: Any, evaluator: BaseEvaluator) -> None:
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator
        retrieval = getattr(settings, "retrieval", None)
        self.default_top_k = getattr(retrieval, "top_k", 10)
        if not isinstance(self.default_top_k, int) or self.default_top_k <= 0:
            self.default_top_k = 10

    def run(self, test_set_path: str | Path) -> EvalReport:
        """Load a golden set, execute each query, and return an aggregate report."""
        cases = self._load_test_cases(test_set_path)
        case_results: list[EvalCaseResult] = []
        metric_values: dict[str, list[float]] = {}

        for case in cases:
            query = case["query"]
            expected_ids = case["expected_chunk_ids"]
            top_k = case.get("top_k", self.default_top_k)
            retrieved = self.hybrid_search.search(query, top_k=top_k)
            retrieved_ids = [self._chunk_id(item) for item in retrieved]
            retrieved_ids = [item for item in retrieved_ids if item]
            retrieved_sources = [self._source(item) for item in retrieved]
            retrieved_sources = [item for item in retrieved_sources if item]

            retrieval_metrics = self._retrieval_metrics(retrieved_ids, expected_ids)
            local_metrics = dict(retrieval_metrics)
            evaluator_trace = {
                "answer": case.get("answer", case.get("expected_answer", "")),
                "reference": case.get("reference", case.get("expected_answer", "")),
                "contexts": [self._text(item) for item in retrieved if self._text(item)],
            }
            evaluator_metrics = self.evaluator.evaluate(
                query, retrieved_ids, expected_ids, trace=evaluator_trace
            )
            # Runner-owned retrieval metrics are always present, even when a
            # generation evaluator (such as Ragas) does not return them.
            local_metrics.update({name: float(value) for name, value in evaluator_metrics.items()})
            local_metrics.update(retrieval_metrics)
            result = EvalCaseResult(
                query=query,
                expected_chunk_ids=expected_ids,
                retrieved_chunk_ids=retrieved_ids,
                expected_sources=case.get("expected_sources", []),
                retrieved_sources=retrieved_sources,
                metrics=local_metrics,
            )
            case_results.append(result)
            for name, value in local_metrics.items():
                metric_values.setdefault(name, []).append(float(value))

        aggregate = {
            name: sum(values) / len(values) for name, values in metric_values.items() if values
        }
        return EvalReport(metrics=aggregate, case_results=case_results)

    @classmethod
    def _load_test_cases(cls, test_set_path: str | Path) -> list[dict[str, Any]]:
        path = Path(test_set_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        if not path.exists():
            raise FileNotFoundError(f"Golden test set 不存在: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Golden test set 不是有效 JSON: {path}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("test_cases"), list):
            raise ValueError("Golden test set 必须包含 test_cases 数组")

        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(payload["test_cases"]):
            if not isinstance(raw, dict) or not isinstance(raw.get("query"), str) or not raw["query"].strip():
                raise ValueError(f"Golden test case #{index + 1} 缺少非空 query")
            expected = raw.get("expected_chunk_ids", [])
            if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
                raise ValueError(f"Golden test case #{index + 1} 的 expected_chunk_ids 必须是字符串数组")
            sources = raw.get("expected_sources", [])
            if not isinstance(sources, list) or not all(isinstance(item, str) for item in sources):
                raise ValueError(f"Golden test case #{index + 1} 的 expected_sources 必须是字符串数组")
            top_k = raw.get("top_k")
            if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
                raise ValueError(f"Golden test case #{index + 1} 的 top_k 必须是正整数")
            normalized.append({**raw, "query": raw["query"].strip(), "expected_chunk_ids": expected, "expected_sources": sources})
        return normalized

    @staticmethod
    def _retrieval_metrics(retrieved_ids: list[str], expected_ids: list[str]) -> dict[str, float]:
        expected = set(expected_ids)
        rank = next((index for index, item in enumerate(retrieved_ids) if item in expected), None)
        return {
            "hit_rate": 1.0 if expected and rank is not None else 0.0,
            "mrr": 1.0 / (rank + 1) if rank is not None else 0.0,
        }

    @staticmethod
    def _chunk_id(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return str(item.get("chunk_id", item.get("id", "")) or "")
        return str(getattr(item, "chunk_id", getattr(item, "id", "")) or "")

    @staticmethod
    def _text(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("text", "") or "")
        return str(getattr(item, "text", "") or "")

    @staticmethod
    def _source(item: Any) -> str:
        metadata = item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})
        if not isinstance(metadata, dict):
            return ""
        return str(metadata.get("source_path", metadata.get("source", "")) or "")
