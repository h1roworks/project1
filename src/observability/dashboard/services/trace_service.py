"""Read and normalize persisted ingestion traces for Dashboard pages."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.settings import Settings
from observability.dashboard.services.config_service import ConfigService


@dataclass(frozen=True)
class TraceStage:
    """One normalized stage in an ingestion trace."""

    name: str
    elapsed_ms: float
    method: str = ""
    provider: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionTrace:
    """A display-ready ingestion trace record."""

    trace_id: str
    started_at: str
    finished_at: str | None
    total_elapsed_ms: float
    source_path: str
    collection: str
    status: str
    stages: list[TraceStage]

    @property
    def filename(self) -> str:
        return Path(self.source_path).name if self.source_path else "(unknown file)"

    @property
    def error(self) -> str | None:
        for stage in self.stages:
            if stage.details.get("error"):
                return str(stage.details["error"])
        return None


@dataclass(frozen=True)
class QueryTrace:
    """A display-ready query trace, including retrieval comparisons."""

    trace_id: str
    started_at: str
    finished_at: str | None
    total_elapsed_ms: float
    query: str
    stages: list[TraceStage]


class TraceService:
    """Load only ``trace_type == ingestion`` records from JSON Lines storage."""

    def __init__(
        self,
        trace_file: str | Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        if trace_file is None:
            settings = settings or ConfigService().get_settings()
            trace_file = settings.observability.trace_file
        path = Path(trace_file)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[4] / path
        self.trace_file = path

    def list_ingestion_traces(self) -> list[IngestionTrace]:
        """Read valid records, tolerate malformed lines, and sort newest first."""
        if not self.trace_file.exists():
            return []
        traces: list[IngestionTrace] = []
        for line in self.trace_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(raw, dict) and raw.get("trace_type") == "ingestion":
                traces.append(self._parse(raw))
        return sorted(traces, key=lambda item: _sort_key(item.started_at), reverse=True)

    def list_query_traces(self, query_filter: str = "") -> list[QueryTrace]:
        """Return query traces newest first, optionally matching query text."""
        if not self.trace_file.exists():
            return []
        traces: list[QueryTrace] = []
        needle = query_filter.strip().casefold()
        for line in self.trace_file.read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(raw, dict) or raw.get("trace_type") != "query":
                continue
            trace = self._parse_query(raw)
            if not needle or needle in trace.query.casefold():
                traces.append(trace)
        return sorted(traces, key=lambda item: _sort_key(item.started_at), reverse=True)

    @staticmethod
    def _parse(raw: dict[str, Any]) -> IngestionTrace:
        stages: list[TraceStage] = []
        source_path = ""
        collection = ""
        failed = False
        skipped = False
        for stage_raw in raw.get("stages") or []:
            if not isinstance(stage_raw, dict) or not stage_raw.get("name"):
                continue
            details = stage_raw.get("details")
            details = dict(details) if isinstance(details, dict) else {}
            source_path = source_path or str(details.get("source_path") or details.get("source") or "")
            collection = collection or str(details.get("collection") or "")
            method = str(stage_raw.get("method") or "")
            failed = failed or method == "failed" or "error" in details
            skipped = skipped or bool(details.get("skipped"))
            stages.append(
                TraceStage(
                    name=str(stage_raw["name"]),
                    elapsed_ms=_number(stage_raw.get("elapsed_ms", 0)),
                    method=method,
                    provider=str(stage_raw.get("provider") or ""),
                    details=details,
                )
            )
        total = _number(raw.get("total_elapsed_ms", 0))
        if total <= 0:
            total = sum(stage.elapsed_ms for stage in stages)
        return IngestionTrace(
            trace_id=str(raw.get("trace_id") or "unknown"),
            started_at=str(raw.get("started_at") or ""),
            finished_at=str(raw["finished_at"]) if raw.get("finished_at") else None,
            total_elapsed_ms=total,
            source_path=source_path,
            collection=collection,
            status="failed" if failed else "skipped" if skipped else "success",
            stages=stages,
        )

    @staticmethod
    def _parse_query(raw: dict[str, Any]) -> QueryTrace:
        stages: list[TraceStage] = []
        query = ""
        for stage_raw in raw.get("stages") or []:
            if not isinstance(stage_raw, dict) or not stage_raw.get("name"):
                continue
            details = stage_raw.get("details")
            details = dict(details) if isinstance(details, dict) else {}
            query = query or str(details.get("query") or "")
            stages.append(
                TraceStage(
                    name=str(stage_raw["name"]),
                    elapsed_ms=_number(stage_raw.get("elapsed_ms", 0)),
                    method=str(stage_raw.get("method") or ""),
                    provider=str(stage_raw.get("provider") or ""),
                    details=details,
                )
            )
        total = _number(raw.get("total_elapsed_ms", 0))
        if total <= 0:
            total = sum(stage.elapsed_ms for stage in stages)
        return QueryTrace(
            trace_id=str(raw.get("trace_id") or "unknown"),
            started_at=str(raw.get("started_at") or ""),
            finished_at=str(raw["finished_at"]) if raw.get("finished_at") else None,
            total_elapsed_ms=total,
            query=query,
            stages=stages,
        )


def _number(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _sort_key(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.min
