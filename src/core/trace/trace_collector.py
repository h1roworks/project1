"""Trace 的收集入口。

持久化策略由调用方以回调函数注入；F2 的 JSON Lines logger 会成为该回调的
实现。这使 TraceContext 保持为纯数据对象，也让单元测试不必读写文件。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.trace.trace_context import TraceContext


class TraceCollector:
    """收集已结束的 trace，并可转交给持久化回调。"""

    def __init__(self, persist: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._persist = persist
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> list[dict[str, Any]]:
        """返回已收集记录的浅拷贝，避免外部直接修改内部列表。"""
        return list(self._records)

    def collect(self, trace: TraceContext) -> None:
        """结束、收集 trace，随后调用可选的持久化函数。"""
        trace.finish()
        record = trace.to_dict()
        self._records.append(record)
        if self._persist is not None:
            self._persist(record)
