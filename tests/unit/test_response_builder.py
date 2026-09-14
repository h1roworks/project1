"""Tests for E3 MCP response and query-tool formatting."""

from __future__ import annotations

from core.response.response_builder import ResponseBuilder
from core.types import RetrievalResult
from mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool


def make_result(
    chunk_id: str = "chunk-1", score: float = 0.91, **metadata: object
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text="RAG 会先检索与问题相关的文档片段，再将其作为上下文提供给模型。",
        metadata={"source_path": "docs/rag.md", "page": 3, **metadata},
    )


def test_response_builder_returns_markdown_and_structured_citations() -> None:
    response = ResponseBuilder().build([make_result()], "什么是 RAG？")

    markdown = response["content"][0]["text"]
    citation = response["structuredContent"]["citations"][0]
    assert "## 知识库检索结果" in markdown
    assert "### [1] `docs/rag.md`，第 3 页" in markdown
    assert "RAG 会先检索" in markdown
    assert citation == {
        "source": "docs/rag.md",
        "page": 3,
        "chunk_id": "chunk-1",
        "score": 0.91,
        "text": "RAG 会先检索与问题相关的文档片段，再将其作为上下文提供给模型。",
    }


def test_response_builder_returns_a_friendly_message_when_nothing_matches() -> None:
    response = ResponseBuilder().build([], "不存在的问题")

    assert "未找到相关文档" in response["content"][0]["text"]
    assert response["structuredContent"]["citations"] == []


class FakeHybridSearch:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, dict[str, str] | None]] = []
        self.traces = []

    def search(self, query: str, top_k: int, filters=None, trace=None):
        self.calls.append((query, top_k, filters))
        self.traces.append(trace)
        return self.results


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievalResult]]] = []
        self.traces = []

    def rerank(self, query: str, candidates: list[RetrievalResult], trace=None) -> list[RetrievalResult]:
        self.calls.append((query, candidates))
        self.traces.append(trace)
        return list(reversed(candidates))


def test_query_tool_runs_search_rerank_and_applies_collection_filter() -> None:
    first = make_result("first", 0.4)
    second = make_result("second", 0.7, source_path="docs/bm25.md", page_num=5)
    search = FakeHybridSearch([first, second])
    reranker = FakeReranker()
    tool = QueryKnowledgeHubTool(
        hybrid_search=search,
        reranker=reranker,
        default_top_k=5,
        trace_writer=lambda _trace: None,
    )

    response = tool("  RAG 如何检索？  ", top_k=1, collection="course-notes")

    assert search.calls == [("RAG 如何检索？", 1, {"collection": "course-notes"})]
    assert reranker.calls == [("RAG 如何检索？", [first, second])]
    assert response["structuredContent"]["citations"][0]["chunk_id"] == "second"
    assert len(response["structuredContent"]["citations"]) == 1


def test_query_tool_passes_one_trace_through_the_chain_and_persists_it() -> None:
    search = FakeHybridSearch([make_result()])
    reranker = FakeReranker()
    persisted = []
    tool = QueryKnowledgeHubTool(
        hybrid_search=search,
        reranker=reranker,
        trace_writer=persisted.append,
    )

    tool("什么是 RAG？")

    assert search.traces[0] is reranker.traces[0]
    assert persisted[0]["trace_type"] == "query"
    assert persisted[0]["finished_at"] is not None
