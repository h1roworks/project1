"""Create stable, client-friendly citations from retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.types import RetrievalResult


@dataclass(frozen=True)
class Citation:
    """A source location attached to one retrieved chunk."""

    source: str
    page: int | str | None
    chunk_id: str
    score: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
        }


class CitationGenerator:
    """Extract the citation fields promised by the MCP tool contract."""

    _SOURCE_FIELDS = ("source_path", "source", "file_path")
    _PAGE_FIELDS = ("page", "page_num", "page_number")

    def generate(self, retrieval_results: list[RetrievalResult]) -> list[Citation]:
        """Convert results into citations without mutating their metadata."""
        return [self._from_result(result) for result in retrieval_results]

    @classmethod
    def _from_result(cls, result: RetrievalResult) -> Citation:
        metadata = result.metadata
        source = cls._first_value(metadata, cls._SOURCE_FIELDS, "未知来源")
        page = cls._first_value(metadata, cls._PAGE_FIELDS, None)
        return Citation(
            source=str(source),
            page=page if isinstance(page, (int, str)) else None,
            chunk_id=result.chunk_id,
            score=float(result.score),
            text=result.text,
        )

    @staticmethod
    def _first_value(
        metadata: dict[str, Any], names: tuple[str, ...], default: Any
    ) -> Any:
        for name in names:
            value = metadata.get(name)
            if value not in (None, ""):
                return value
        return default
