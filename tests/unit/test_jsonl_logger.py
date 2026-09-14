"""F2: Trace JSON Lines 日志测试。"""

from __future__ import annotations

import json
import logging

import pytest

from observability.logger import JSONFormatter, get_trace_logger, write_trace


def test_json_formatter_keeps_trace_fields() -> None:
    formatter = JSONFormatter()
    record = logging.makeLogRecord({
        "name": "trace-test",
        "levelno": logging.INFO,
        "levelname": "INFO",
        "msg": "trace",
        "trace": {"trace_id": "t-1", "trace_type": "query", "text": "中文"},
    })

    assert json.loads(formatter.format(record)) == {
        "trace_id": "t-1", "trace_type": "query", "text": "中文",
    }


def test_write_trace_creates_one_valid_json_line(tmp_path) -> None:
    trace_file = tmp_path / "nested" / "traces.jsonl"
    write_trace({"trace_id": "t-1", "trace_type": "ingestion", "stages": []}, trace_file)

    lines = trace_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trace_type"] == "ingestion"


def test_write_trace_appends_instead_of_overwriting(tmp_path) -> None:
    trace_file = tmp_path / "traces.jsonl"
    write_trace({"trace_id": "t-1", "trace_type": "query"}, trace_file)
    write_trace({"trace_id": "t-2", "trace_type": "ingestion"}, trace_file)

    assert [json.loads(line)["trace_id"] for line in trace_file.read_text(encoding="utf-8").splitlines()] == [
        "t-1", "t-2",
    ]


def test_get_trace_logger_does_not_duplicate_handlers(tmp_path) -> None:
    trace_file = tmp_path / "traces.jsonl"

    first = get_trace_logger(trace_file)
    second = get_trace_logger(trace_file)

    assert first is second
    assert len(first.handlers) == 1


def test_write_trace_rejects_non_dictionary(tmp_path) -> None:
    with pytest.raises(TypeError, match="trace_dict"):
        write_trace(["not", "a", "dict"], tmp_path / "traces.jsonl")  # type: ignore[arg-type]
