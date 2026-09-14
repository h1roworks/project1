"""H1: RagasEvaluator 的无网络单元测试。"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from core.settings import EvaluationSettings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory
from observability.evaluation.ragas_evaluator import RagasEvaluator


def _install_fake_ragas(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}
    ragas = ModuleType("ragas")
    metrics = ModuleType("ragas.metrics")
    metrics.faithfulness = object()
    metrics.answer_relevancy = object()
    metrics.context_precision = object()

    def evaluate(*, dataset, metrics):
        captured["dataset"] = dataset
        captured["metrics"] = metrics
        return {
            "faithfulness": 0.91,
            "answer_relevancy": 0.82,
            "context_precision": 0.76,
        }

    ragas.evaluate = evaluate
    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics)
    return captured


def test_evaluate_returns_normalised_ragas_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_ragas(monkeypatch)

    metrics = RagasEvaluator().evaluate(
        "RRF 是什么？",
        ["chunk-1", "chunk-2"],
        ["chunk-1"],
        trace={
            "answer": "RRF 会融合多个检索结果的排序。",
            "contexts": ["RRF 使用倒数排名融合。"],
            "reference": "RRF 是倒数排名融合算法。",
        },
    )

    assert metrics == {
        "faithfulness": 0.91,
        "answer_relevancy": 0.82,
        "context_precision": 0.76,
    }
    row = captured["dataset"][0]
    assert row["question"] == "RRF 是什么？"
    assert row["contexts"] == ["RRF 使用倒数排名融合。"]


def test_evaluate_uses_ids_as_safe_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_ragas(monkeypatch)

    RagasEvaluator().evaluate("q", ["retrieved"], ["golden"])

    row = captured["dataset"][0]
    assert row["answer"] == "retrieved"
    assert row["contexts"] == ["retrieved"]
    assert row["ground_truth"] == "golden"


def test_factory_creates_ragas_evaluator() -> None:
    evaluator = EvaluatorFactory.create(EvaluationSettings(provider="ragas"))
    assert isinstance(evaluator, RagasEvaluator)
    assert isinstance(evaluator, BaseEvaluator)


def test_missing_ragas_has_actionable_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__("importlib").import_module

    def missing_ragas(name: str, *args, **kwargs):
        if name == "ragas":
            raise ModuleNotFoundError("No module named 'ragas'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("observability.evaluation.ragas_evaluator.importlib.import_module", missing_ragas)
    with pytest.raises(ImportError, match="pip install ragas"):
        RagasEvaluator().evaluate("q", [], [])
