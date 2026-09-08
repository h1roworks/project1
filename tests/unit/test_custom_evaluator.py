"""B6: CustomEvaluator 与 Evaluator 工厂测试。"""

import pytest

from core.settings import EvaluationSettings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


def evaluate(retrieved, golden) -> dict[str, float]:
    return CustomEvaluator().evaluate("q", retrieved, golden)


# ---------- CustomEvaluator 指标 ----------

def test_hit_rate_hit() -> None:
    assert evaluate(["a", "b"], ["b"])["hit_rate"] == 1.0


def test_hit_rate_miss() -> None:
    assert evaluate(["a", "b"], ["z"])["hit_rate"] == 0.0


def test_mrr_first_rank() -> None:
    assert evaluate(["a", "b"], ["a"])["mrr"] == 1.0


def test_mrr_third_rank() -> None:
    assert evaluate(["a", "b", "c"], ["c"])["mrr"] == 1.0 / 3.0


def test_mrr_miss() -> None:
    assert evaluate(["a", "b"], ["z"])["mrr"] == 0.0


def test_empty_retrieved() -> None:
    assert evaluate([], ["a"]) == {"hit_rate": 0.0, "mrr": 0.0}


def test_empty_golden() -> None:
    assert evaluate(["a"], []) == {"hit_rate": 0.0, "mrr": 0.0}


def test_metrics_stable_types() -> None:
    metrics = evaluate(["a", "b", "c"], ["c"])
    assert set(metrics) == {"hit_rate", "mrr"}
    assert all(isinstance(v, float) for v in metrics.values())


# ---------- Evaluator 工厂 ----------

def test_factory_creates_custom() -> None:
    ev = EvaluatorFactory.create(EvaluationSettings(provider="custom"))
    assert isinstance(ev, CustomEvaluator)
    assert isinstance(ev, BaseEvaluator)


def test_factory_empty_provider_raises() -> None:
    with pytest.raises(ValueError, match="provider 未配置"):
        EvaluatorFactory.create(EvaluationSettings(provider=""))


def test_factory_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="未知的 Evaluator provider"):
        EvaluatorFactory.create(EvaluationSettings(provider="nonexistent"))
