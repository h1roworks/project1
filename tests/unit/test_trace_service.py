"""Tests for G5 ingestion trace parsing and normalization."""

from __future__ import annotations

import json

from observability.dashboard.services.trace_service import TraceService


def test_trace_service_filters_sorts_and_normalizes_records(tmp_path) -> None:
    trace_file = tmp_path / "traces.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                "not-json",
                json.dumps({"trace_type": "query", "trace_id": "query"}),
                json.dumps({
                    "trace_type": "ingestion",
                    "trace_id": "old",
                    "started_at": "2026-09-14T01:00:00+00:00",
                    "total_elapsed_ms": 0,
                    "stages": [{"name": "load", "elapsed_ms": 2, "details": {"source_path": "/old.pdf", "collection": "docs"}}],
                }),
                json.dumps({
                    "trace_type": "ingestion",
                    "trace_id": "new",
                    "started_at": "2026-09-14T02:00:00+00:00",
                    "finished_at": "2026-09-14T02:00:01+00:00",
                    "total_elapsed_ms": 10,
                    "stages": [
                        {"name": "load", "elapsed_ms": 3, "method": "loader", "details": {"source_path": "/new.pdf", "collection": "docs"}},
                        {"name": "upsert", "elapsed_ms": 7, "method": "vector_bm25_upsert", "details": {}},
                    ],
                }),
            ]
        ),
        encoding="utf-8",
    )

    traces = TraceService(trace_file).list_ingestion_traces()

    assert [trace.trace_id for trace in traces] == ["new", "old"]
    assert traces[0].filename == "new.pdf"
    assert traces[0].collection == "docs"
    assert traces[0].status == "success"
    assert [stage.name for stage in traces[0].stages] == ["load", "upsert"]
    assert traces[1].total_elapsed_ms == 2


def test_trace_service_marks_failed_and_skipped_records(tmp_path) -> None:
    trace_file = tmp_path / "traces.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                json.dumps({"trace_type": "ingestion", "trace_id": "failed", "stages": [{"name": "load", "method": "failed", "details": {"error": "bad pdf"}}]}),
                json.dumps({"trace_type": "ingestion", "trace_id": "skipped", "stages": [{"name": "load", "details": {"skipped": True}}]}),
            ]
        ),
        encoding="utf-8",
    )

    traces = {trace.trace_id: trace for trace in TraceService(trace_file).list_ingestion_traces()}

    assert traces["failed"].status == "failed"
    assert traces["failed"].error == "bad pdf"
    assert traces["skipped"].status == "skipped"
