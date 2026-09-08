"""CustomEvaluator：轻量自定义指标（hit_rate / mrr）。

不依赖任何外部框架，用于快速回归测试与上线前 sanity check。
"""

from __future__ import annotations

from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    name = "custom"

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any = None,
    ) -> dict[str, float]:
        golden = set(golden_ids)
        hit = next((i for i, rid in enumerate(retrieved_ids) if rid in golden), None)

        hit_rate = 1.0 if (golden and hit is not None) else 0.0
        mrr = 1.0 / (hit + 1) if hit is not None else 0.0
        return {"hit_rate": hit_rate, "mrr": mrr}
