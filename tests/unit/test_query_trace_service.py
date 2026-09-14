"""Tests for G6 query trace parsing and filtering."""

from __future__ import annotations

import json

from observability.dashboard.services.trace_service import TraceService


def test_query_trace_service_extracts_query_and_retrieval_details(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps({
            "trace_type": "query",
            "trace_id": "query-1",
            "started_at": "2026-09-14T02:00:00+00:00",
            "total_elapsed_ms": 12,
            "stages": [
                {"name": "query_processing", "elapsed_ms": 1, "details": {"query": "如何配置 RAG"}},
                {"name": "dense_retrieval", "elapsed_ms": 4, "details": {"result_count": 1, "results": [{"chunk_id": "dense-1", "score": 0.9}]}},
                {"name": "sparse_retrieval", "elapsed_ms": 3, "details": {"result_count": 1, "results": [{"chunk_id": "sparse-1", "score": 2.1}]}},
                {"name": "rerank", "elapsed_ms": 2, "details": {"before_ids": ["dense-1", "sparse-1"], "after_ids": ["sparse-1", "dense-1"]}},
            ],
        }),
        encoding="utf-8",
    )

    service = TraceService(path)
    traces = service.list_query_traces("rag")

    assert len(traces) == 1
    assert traces[0].query == "如何配置 RAG"
    assert traces[0].stages[1].details["results"][0]["chunk_id"] == "dense-1"
    assert traces[0].stages[3].details["after_ids"] == ["sparse-1", "dense-1"]
    assert service.list_query_traces("不存在") == []
