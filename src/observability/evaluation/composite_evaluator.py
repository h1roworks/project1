"""Composite evaluator that runs several evaluators concurrently."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


class CompositeEvaluator(BaseEvaluator):
    """Combine the metric dictionaries returned by multiple evaluators.

    Evaluators are independent (for example, CustomEvaluator is local while
    RagasEvaluator may call an LLM), so they run in a small thread pool.  The
    output keeps metric names unchanged when possible. If two evaluators
    return the same name, the later one is namespaced as
    ``<evaluator.name>.<metric>`` instead of silently overwriting a score.
    """

    name = "composite"

    def __init__(self, evaluators: list[BaseEvaluator]) -> None:
        self.evaluators = list(evaluators)

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any = None,
    ) -> dict[str, float]:
        """Run all configured evaluators in parallel and merge their scores."""
        if not self.evaluators:
            return {}

        with ThreadPoolExecutor(max_workers=len(self.evaluators)) as executor:
            futures = [
                executor.submit(
                    evaluator.evaluate,
                    query,
                    retrieved_ids,
                    golden_ids,
                    trace,
                )
                for evaluator in self.evaluators
            ]
            results = [future.result() for future in futures]

        merged: dict[str, float] = {}
        for evaluator, metrics in zip(self.evaluators, results):
            for metric_name, value in metrics.items():
                output_name = metric_name
                if output_name in merged:
                    output_name = f"{evaluator.name}.{metric_name}"
                merged[output_name] = float(value)
        return merged


EvaluatorFactory.register("composite", CompositeEvaluator)
