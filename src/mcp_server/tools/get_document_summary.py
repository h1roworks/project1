"""MCP tool for reading a document's metadata and summary from local storage."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.response.response_builder import MCPResponse
from core.settings import Settings, load_settings
from libs.vector_store.base_vector_store import VectorMatch
from libs.vector_store.vector_store_factory import VectorStoreFactory


GET_DOCUMENT_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_id": {
            "type": "string",
            "description": "摄取时生成的文档 ID。",
        }
    },
    "required": ["doc_id"],
}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


class DocumentNotFoundError(LookupError):
    """Raised when no stored chunk belongs to the requested document."""


class DocumentSummaryTool:
    """Build a document-level view from metadata attached to its stored chunks."""

    def __init__(
        self,
        document_reader: Callable[[str], list[VectorMatch]] | None = None,
        reader_factory: Callable[[], Callable[[str], list[VectorMatch]]] | None = None,
    ) -> None:
        if document_reader is None and reader_factory is None:
            raise ValueError("Provide document_reader or reader_factory.")
        self._document_reader = document_reader
        self._reader_factory = reader_factory

    @classmethod
    def from_config(
        cls, config_path: str | Path = _DEFAULT_CONFIG_PATH
    ) -> "DocumentSummaryTool":
        """Create a lazily connected reader for the configured vector store."""
        settings = load_settings(config_path)
        return cls(reader_factory=lambda: _build_document_reader(settings))

    def __call__(self, doc_id: str) -> MCPResponse:
        """Return title, summary and tags for ``doc_id`` or raise not-found."""
        normalized_id = self._validate_doc_id(doc_id)
        chunks = self._get_document_reader()(normalized_id)
        if not chunks:
            raise DocumentNotFoundError(normalized_id)

        first = chunks[0]
        metadata = first.metadata
        title = _text_value(metadata.get("title")) or _fallback_title(metadata, normalized_id)
        summary = _text_value(metadata.get("summary")) or _fallback_summary(first.text)
        tags = _tags_value(metadata.get("tags"))
        source = _text_value(
            metadata.get("source_path") or metadata.get("source") or metadata.get("file_path")
        ) or "未知来源"
        document = {
            "doc_id": normalized_id,
            "title": title,
            "summary": summary,
            "tags": tags,
            "source": source,
            "chunk_count": len(chunks),
        }
        markdown = "\n".join(
            [
                f"## {title}",
                "",
                f"**文档 ID：** `{normalized_id}`",
                f"**来源：** `{source}`",
                f"**片段数：** {len(chunks)}",
                f"**标签：** {', '.join(tags) if tags else '无'}",
                "",
                summary,
            ]
        )
        return {
            "content": [{"type": "text", "text": markdown}],
            "structuredContent": {"document": document},
        }

    def _get_document_reader(self) -> Callable[[str], list[VectorMatch]]:
        if self._document_reader is None:
            assert self._reader_factory is not None
            self._document_reader = self._reader_factory()
        return self._document_reader

    @staticmethod
    def _validate_doc_id(doc_id: str) -> str:
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("doc_id must be a non-empty string.")
        return doc_id.strip()


def _build_document_reader(settings: Settings) -> Callable[[str], list[VectorMatch]]:
    vector_store = VectorStoreFactory.create(settings.vector_store)
    find_by_metadata = getattr(vector_store, "find_by_metadata", None)
    if not callable(find_by_metadata):
        raise RuntimeError(
            "The configured vector store does not support document metadata lookup."
        )
    return lambda doc_id: find_by_metadata("doc_id", doc_id)


def _text_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _tags_value(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [value]
        value = parsed
    if not isinstance(value, list):
        return []
    return [str(tag).strip() for tag in value if str(tag).strip()]


def _fallback_title(metadata: dict[str, Any], doc_id: str) -> str:
    source = _text_value(metadata.get("source_path"))
    return Path(source).stem if source else doc_id


def _fallback_summary(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[:limit - 1]}…"


def get_document_summary(doc_id: str) -> MCPResponse:
    """Convenience entry point for calling the tool outside the MCP server."""
    return DocumentSummaryTool.from_config()(doc_id)
