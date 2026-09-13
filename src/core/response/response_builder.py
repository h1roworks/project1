"""Format retrieved chunks as an MCP tool result with transparent citations."""

from __future__ import annotations

from typing import Any, TypedDict

from core.response.citation_generator import Citation, CitationGenerator
from core.types import RetrievalResult


class MCPResponse(TypedDict):
    """The JSON-serializable part of an MCP ``tools/call`` result."""

    content: list[dict[str, str]]
    structuredContent: dict[str, Any]


class ResponseBuilder:
    """Build the human-readable and structured parts of a query response."""

    def __init__(self, citation_generator: CitationGenerator | None = None) -> None:
        self.citation_generator = citation_generator or CitationGenerator()

    def build(self, retrieval_results: list[RetrievalResult], query: str) -> MCPResponse:
        """Return Markdown plus citations for the supplied retrieval results."""
        citations = self.citation_generator.generate(retrieval_results)
        if not citations:
            answer = "未找到相关文档，请先运行 ingest.py 摄取数据，或换一个更具体的问题。"
            return {
                "content": [{"type": "text", "text": answer}],
                "structuredContent": {"answer": answer, "citations": []},
            }

        answer = f"知识库找到了 {len(citations)} 条与问题相关的片段。"
        markdown_lines = [
            "## 知识库检索结果",
            "",
            f"**问题：** {query}",
            "",
            answer,
            "",
        ]
        for index, citation in enumerate(citations, start=1):
            markdown_lines.extend(self._markdown_citation(index, citation))

        return {
            "content": [{"type": "text", "text": "\n".join(markdown_lines)}],
            "structuredContent": {
                "answer": answer,
                "citations": [citation.to_dict() for citation in citations],
            },
        }

    @staticmethod
    def _markdown_citation(index: int, citation: Citation) -> list[str]:
        page = f"，第 {citation.page} 页" if citation.page is not None else ""
        quoted_text = "\n".join(
            f"> {line}" if line else ">" for line in citation.text.splitlines()
        )
        return [
            f"### [{index}] `{citation.source}`{page}",
            quoted_text or "> （该片段没有可显示的正文）",
            f"_chunk_id: `{citation.chunk_id}` · score: {citation.score:.4f}_",
            "",
        ]
