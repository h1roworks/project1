"""一次 RAG 请求的轻量级追踪上下文。

``TraceContext`` 不知道日志最终写到哪里；它只负责在内存中收集阶段数据，
并生成可直接交给 JSON Lines logger 的字典。这样业务组件可以选择性地传入
trace，而不需要依赖具体的日志实现。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

_TRACE_TYPES = frozenset({"query", "ingestion"})


class TraceContext:
    """收集一次 query 或 ingestion 请求的阶段信息。

    Args:
        trace_type: 追踪链路类型，只能是 ``"query"`` 或 ``"ingestion"``。
    """

    def __init__(self, trace_type: str = "query") -> None:
        if trace_type not in _TRACE_TYPES:
            allowed = ", ".join(sorted(_TRACE_TYPES))
            raise ValueError(f"trace_type must be one of: {allowed}")

        self.trace_id = str(uuid4())
        self.trace_type = trace_type
        self.started_at = datetime.now(timezone.utc)
        self.finished_at: datetime | None = None
        self.stages: list[dict[str, Any]] = []
        self._started_counter = perf_counter()
        self._finished_counter: float | None = None

    def record_stage(
        self,
        stage_name: str,
        elapsed_ms: float,
        *,
        method: str | None = None,
        provider: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """追加一个已完成阶段的记录。

        ``elapsed_ms`` 由调用该阶段的组件测量；TraceContext 只保存该结果，
        所以它既可记录本地计算，也可记录网络调用。
        """
        if self.finished_at is not None:
            raise RuntimeError("cannot record a stage after the trace is finished")
        if not stage_name or not isinstance(stage_name, str):
            raise ValueError("stage_name must be a non-empty string")
        if elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative")

        stage: dict[str, Any] = {
            "name": stage_name,
            "elapsed_ms": float(elapsed_ms),
        }
        if method is not None:
            stage["method"] = method
        if provider is not None:
            stage["provider"] = provider
        if details is not None:
            stage["details"] = _json_safe(dict(details))
        self.stages.append(stage)

    def finish(self) -> None:
        """标记请求结束。

        该方法是幂等的：重复调用不会覆盖第一次结束时间，便于异常处理的
        ``finally`` 块安全地调用它。
        """
        if self.finished_at is None:
            self.finished_at = datetime.now(timezone.utc)
            self._finished_counter = perf_counter()

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        """返回指定阶段（同名时最后一个）或整个请求的耗时，单位为毫秒。"""
        if stage_name is not None:
            for stage in reversed(self.stages):
                if stage["name"] == stage_name:
                    return float(stage["elapsed_ms"])
            raise KeyError(f"stage not found: {stage_name}")

        end = self._finished_counter if self._finished_counter is not None else perf_counter()
        return max(0.0, (end - self._started_counter) * 1000)

    def to_dict(self) -> dict[str, Any]:
        """生成能被 :func:`json.dumps` 直接序列化的追踪记录。"""
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "total_elapsed_ms": self.elapsed_ms(),
            "stages": _json_safe(self.stages),
        }


def _json_safe(value: Any) -> Any:
    """把阶段 details 中常见的 Python 值转换为 JSON 支持的类型。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)
