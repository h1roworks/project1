"""H2: CompositeEvaluator 组合与工厂路由测试。"""

from __future__ import annotations

from typing import Any

from core.settings import EvaluationSettings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory
from observability.evaluation.composite_evaluator import CompositeEvaluator


class _FirstEvaluator(BaseEvaluator):
    name = "first"

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any = None,
    ) -> dict[str, float]:
        return {"first_score": 0.25}


class _SecondEvaluator(BaseEvaluator):
    name = "second"

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any = None,
    ) -> dict[str, float]:
        return {"second_score": 0.75}


def test_composite_merges_metrics_from_all_evaluators() -> None:
    evaluator = CompositeEvaluator([_FirstEvaluator(), _SecondEvaluator()])
    assert evaluator.evaluate("q", ["chunk"], ["chunk"]) == {
        "first_score": 0.25,
        "second_score": 0.75,
    }


def test_composite_namespaces_duplicate_metric_names() -> None:
    evaluator = CompositeEvaluator([_FirstEvaluator(), _FirstEvaluator()])
    assert evaluator.evaluate("q", [], []) == {
        "first_score": 0.25,
        "first.first_score": 0.25,
    }


def test_empty_composite_returns_empty_metrics() -> None:
    assert CompositeEvaluator([]).evaluate("q", [], []) == {}


def test_factory_builds_configured_composite() -> None:
    evaluator = EvaluatorFactory.create(
        EvaluationSettings(provider="composite", backends=["custom", "custom"])
    )
    assert isinstance(evaluator, CompositeEvaluator)
    assert evaluator.evaluate("q", ["hit"], ["hit"]) == {
        "hit_rate": 1.0,
        "mrr": 1.0,
        "custom.hit_rate": 1.0,
        "custom.mrr": 1.0,
    }
