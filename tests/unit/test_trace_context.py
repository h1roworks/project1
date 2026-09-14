"""F1: TraceContext 的生命周期、耗时和 JSON 契约测试。"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from core.trace import TraceCollector, TraceContext


def test_trace_defaults_to_query_and_has_unique_id() -> None:
    first = TraceContext()
    second = TraceContext()

    assert first.trace_type == "query"
    assert first.trace_id != second.trace_id


def test_trace_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="trace_type"):
        TraceContext("evaluation")


def test_record_stage_preserves_measurement_and_details() -> None:
    trace = TraceContext("ingestion")
    trace.record_stage(
        "embed",
        12.5,
        method="dense",
        provider="ollama",
        details={"chunk_count": 2},
    )

    assert trace.elapsed_ms("embed") == 12.5
    assert trace.stages == [{
        "name": "embed",
        "elapsed_ms": 12.5,
        "method": "dense",
        "provider": "ollama",
        "details": {"chunk_count": 2},
    }]


def test_finish_is_idempotent_and_stops_total_duration() -> None:
    trace = TraceContext()
    trace.finish()
    first_finished_at = trace.finished_at
    first_elapsed = trace.elapsed_ms()
    trace.finish()

    assert trace.finished_at == first_finished_at
    assert trace.elapsed_ms() == first_elapsed


def test_to_dict_has_required_json_serializable_fields() -> None:
    trace = TraceContext("ingestion")
    trace.record_stage("split", 3, details={"when": datetime(2026, 1, 1)})
    trace.finish()
    record = trace.to_dict()

    assert set(record) == {
        "trace_id", "trace_type", "started_at", "finished_at",
        "total_elapsed_ms", "stages",
    }
    assert record["trace_type"] == "ingestion"
    assert record["finished_at"] is not None
    assert record["total_elapsed_ms"] >= 0
    assert json.loads(json.dumps(record)) == record


def test_collect_finishes_trace_and_calls_persistence_callback() -> None:
    persisted: list[dict[str, object]] = []
    collector = TraceCollector(persisted.append)
    trace = TraceContext()

    collector.collect(trace)

    assert trace.finished_at is not None
    assert collector.records == persisted
    assert persisted[0]["trace_id"] == trace.trace_id


def test_cannot_record_stage_after_finish() -> None:
    trace = TraceContext()
    trace.finish()

    with pytest.raises(RuntimeError, match="finished"):
        trace.record_stage("late", 1)
