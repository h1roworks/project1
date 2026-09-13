"""MCP tool for browsing the local document collections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.response.response_builder import MCPResponse


LIST_COLLECTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DOCUMENTS_DIRECTORY = _PROJECT_ROOT / "data" / "documents"


@dataclass(frozen=True)
class CollectionInfo:
    """A collection folder and lightweight file statistics for client display."""

    name: str
    document_count: int
    total_size_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "document_count": self.document_count,
            "total_size_bytes": self.total_size_bytes,
        }


class ListCollectionsTool:
    """List immediate subdirectories of ``data/documents`` as collections."""

    def __init__(self, documents_directory: str | Path = _DEFAULT_DOCUMENTS_DIRECTORY) -> None:
        self.documents_directory = Path(documents_directory)

    def __call__(self) -> MCPResponse:
        """Return Markdown and structured metadata for every collection folder."""
        collections = self.list_collections()
        if not collections:
            answer = "暂未发现知识库集合。请在 data/documents/ 下创建集合目录并摄取文档。"
            return {
                "content": [{"type": "text", "text": answer}],
                "structuredContent": {"collections": []},
            }

        markdown_lines = ["## 可用知识库集合", ""]
        for collection in collections:
            markdown_lines.append(
                f"- `{collection.name}`：{collection.document_count} 个文档"
            )
        return {
            "content": [{"type": "text", "text": "\n".join(markdown_lines)}],
            "structuredContent": {
                "collections": [collection.to_dict() for collection in collections]
            },
        }

    def list_collections(self) -> list[CollectionInfo]:
        """Return sorted collection directories with recursive file statistics."""
        if not self.documents_directory.is_dir():
            return []

        collections: list[CollectionInfo] = []
        for directory in sorted(self.documents_directory.iterdir(), key=lambda item: item.name.casefold()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            files = [path for path in directory.rglob("*") if path.is_file()]
            collections.append(
                CollectionInfo(
                    name=directory.name,
                    document_count=len(files),
                    total_size_bytes=sum(path.stat().st_size for path in files),
                )
            )
        return collections


def list_collections() -> MCPResponse:
    """Convenience entry point for calling the tool outside the MCP server."""
    return ListCollectionsTool()()
