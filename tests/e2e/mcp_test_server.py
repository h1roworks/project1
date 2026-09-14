"""Deterministic MCP server used only by the MCP client E2E test.

The production server builds its query engine from the local configuration,
which can require Ollama and a populated Chroma collection.  This small entry
point keeps the stdio server and the real query response formatting intact,
but supplies a fixed retrieval result so the client protocol test is portable.
"""

from __future__ import annotations

import asyncio

from core.types import RetrievalResult
from mcp.server.stdio import stdio_server
from mcp_server.server import create_server
from mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool


class _FixedHybridSearch:
    """Return one known chunk without opening an embedding model or database."""

    def search(self, _query: str, top_k: int, *, filters=None, trace=None):
        del filters, trace
        return [
            RetrievalResult(
                chunk_id="e2e-rag-chunk",
                score=0.99,
                text="RAG combines retrieved context with a language model.",
                metadata={"source_path": "fixtures/rag-introduction.md", "page_num": 1},
            )
        ][:top_k]


class _IdentityReranker:
    """Preserve the fixed candidate's order for this transport-focused test."""

    def rerank(self, _query: str, candidates, *, trace=None):
        del trace
        return candidates


def _empty_tool(**_arguments):
    return {"content": [], "structuredContent": {}}


async def run() -> None:
    """Run the real MCP stdio server with deterministic query dependencies."""
    query_tool = QueryKnowledgeHubTool(
        hybrid_search=_FixedHybridSearch(),
        reranker=_IdentityReranker(),
        default_top_k=1,
        trace_writer=lambda _trace: None,
    )
    server = create_server(
        query_tool=query_tool,
        collections_tool=_empty_tool,
        document_summary_tool=_empty_tool,
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run())
