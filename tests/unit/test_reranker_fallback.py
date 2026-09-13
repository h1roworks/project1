"""D6：Core Reranker 的编排、超时与回退测试。"""

import time
from types import SimpleNamespace

import pytest

from core.query_engine.reranker import Reranker
from core.types import RetrievalResult
from libs.reranker.base_reranker import NoneReranker, RerankCandidate


def make_result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"text for {chunk_id}",
        metadata={"source_path": f"{chunk_id}.md"},
    )


class RecordingBackend:
    backend = "fake"

    def __init__(self, returned=None, error: Exception | None = None, delay: float = 0) -> None:
        self.returned = returned
        self.error = error
        self.delay = delay
        self.calls = []

    def rerank(self, query, candidates, trace=None):
        self.calls.append((query, candidates, trace))
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.returned if self.returned is not None else list(candidates)


def settings(top_m: int = 10, timeout_seconds: float = 1) -> SimpleNamespace:
    return SimpleNamespace(
        rerank=SimpleNamespace(top_m=top_m, timeout_seconds=timeout_seconds)
    )


def test_successful_rerank_maps_order_scores_and_metadata() -> None:
    backend = RecordingBackend([
        RerankCandidate(id="b", text="reranked b", score=0.99, metadata={"rank": 1}),
        RerankCandidate(id="a", text="reranked a", score=0.80, metadata={"rank": 2}),
    ])
    reranker = Reranker(settings(), reranker=backend)

    results = reranker.rerank("query", [make_result("a", 0.4), make_result("b", 0.3)])

    assert [item.chunk_id for item in results] == ["b", "a"]
    assert [item.score for item in results] == [0.99, 0.80]
    assert results[0].text == "reranked b"
    assert results[0].metadata["rerank_backend"] == "fake"
    assert results[0].metadata["rerank_fallback"] is False
    assert backend.calls[0][1][0].id == "a"


def test_backend_error_returns_original_fusion_order_with_fallback_marker() -> None:
    originals = [make_result("a", 0.4), make_result("b", 0.3)]
    reranker = Reranker(settings(), reranker=RecordingBackend(error=RuntimeError("offline")))

    results = reranker.rerank("query", originals)

    assert [item.chunk_id for item in results] == ["a", "b"]
    assert [item.score for item in results] == [0.4, 0.3]
    assert all(item.metadata["rerank_fallback"] is True for item in results)
    assert all("offline" in item.metadata["rerank_error"] for item in results)
    assert "rerank_fallback" not in originals[0].metadata


def test_timeout_returns_fallback_without_waiting_for_backend_completion() -> None:
    backend = RecordingBackend(delay=0.2)
    reranker = Reranker(settings(timeout_seconds=0.02), reranker=backend)
    start = time.monotonic()

    results = reranker.rerank("query", [make_result("a", 0.4)])

    assert time.monotonic() - start < 0.15
    assert results[0].metadata["rerank_fallback"] is True
    assert "未完成" in results[0].metadata["rerank_error"]


def test_top_m_limits_backend_pool_and_keeps_remaining_candidates() -> None:
    backend = RecordingBackend([
        RerankCandidate(id="b", text="text for b", score=0.9),
        RerankCandidate(id="a", text="text for a", score=0.8),
    ])
    reranker = Reranker(settings(top_m=2), reranker=backend)

    results = reranker.rerank(
        "query", [make_result("a", 0.5), make_result("b", 0.4), make_result("c", 0.3)]
    )

    assert [candidate.id for candidate in backend.calls[0][1]] == ["a", "b"]
    assert [item.chunk_id for item in results] == ["b", "a", "c"]


def test_backend_omitting_candidate_keeps_it_after_returned_ranking() -> None:
    backend = RecordingBackend([RerankCandidate(id="b", text="text for b", score=0.9)])

    results = Reranker(settings(), reranker=backend).rerank(
        "query", [make_result("a", 0.5), make_result("b", 0.4)]
    )

    assert [item.chunk_id for item in results] == ["b", "a"]


def test_none_backend_keeps_order_without_fallback() -> None:
    results = Reranker(settings(), reranker=NoneReranker()).rerank(
        "query", [make_result("a", 0.5), make_result("b", 0.4)]
    )

    assert [item.chunk_id for item in results] == ["a", "b"]
    assert all(item.metadata["rerank_fallback"] is False for item in results)
    assert all(item.metadata["rerank_backend"] == "none" for item in results)


def test_empty_candidates_returns_without_calling_backend() -> None:
    backend = RecordingBackend()

    assert Reranker(settings(), reranker=backend).rerank("query", []) == []
    assert backend.calls == []


@pytest.mark.parametrize(
    ("query", "candidates", "error"),
    [
        ("", [], "query"),
        (None, [], "query"),
        ("query", None, "candidates"),
    ],
)
def test_invalid_arguments_are_rejected(query, candidates, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        Reranker(settings(), reranker=RecordingBackend()).rerank(query, candidates)


def test_trace_is_forwarded_to_backend() -> None:
    backend = RecordingBackend()
    trace = object()

    Reranker(settings(), reranker=backend).rerank("query", [make_result("a", 0.5)], trace=trace)

    assert backend.calls[0][2] is trace


def test_constructor_uses_factory_when_backend_is_not_injected(monkeypatch) -> None:
    backend = RecordingBackend()
    config = settings()
    monkeypatch.setattr(
        "core.query_engine.reranker.RerankerFactory.create",
        lambda received: backend if received is config.rerank else None,
    )

    reranker = Reranker(config)

    assert reranker.backend is backend
