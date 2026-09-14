"""H4: evaluation Dashboard display helpers."""

from __future__ import annotations

from observability.dashboard.pages.evaluation_panel import _case_rows
from observability.evaluation.eval_runner import EvalCaseResult, EvalReport


def test_case_rows_are_dashboard_friendly() -> None:
    report = EvalReport(
        metrics={"hit_rate": 1.0, "mrr": 0.5},
        case_results=[
            EvalCaseResult(
                query="什么是 RRF？",
                expected_chunk_ids=["expected"],
                retrieved_chunk_ids=["other", "expected"],
                retrieved_sources=["docs/rag.md"],
                metrics={"hit_rate": 1.0, "mrr": 0.5},
            )
        ],
    )

    assert _case_rows(report) == [
        {
            "问题": "什么是 RRF？",
            "命中": "是",
            "MRR": 0.5,
            "期望 Chunk": "expected",
            "召回 Chunk": "other, expected",
            "来源": "docs/rag.md",
        }
    ]
