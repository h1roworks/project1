"""D5：HybridSearch 编排测试，使用替身组件验证跨模块协作。"""

from threading import Event
from types import SimpleNamespace

import pytest

from core.query_engine.fusion import Fusion
from core.query_engine.hybrid_search import HybridSearch
from core.types import ProcessedQuery, RetrievalResult


def result(chunk_id: str, **metadata) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=1.0,
        text=f"text for {chunk_id}",
        metadata=metadata,
    )


class RecordingProcessor:
    def __init__(self, processed: ProcessedQuery) -> None:
        self.processed = processed
        self.calls = []

    def process(self, query, filters=None, trace=None):
        self.calls.append((query, filters, trace))
        return self.processed


class RecordingRetriever:
    def __init__(self, results=None, error: Exception | None = None) -> None:
        self.results = [] if results is None else results
        self.error = error
        self.calls = []

    def retrieve(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.results


class RecordingFusion:
    def __init__(self) -> None:
        self.calls = []
        self._real_fusion = Fusion(k=10)

    def fuse(self, dense_results, sparse_results, top_k=None, trace=None):
        self.calls.append((dense_results, sparse_results, top_k, trace))
        return self._real_fusion.fuse(dense_results, sparse_results, top_k, trace)


def settings(hard_filters=None) -> SimpleNamespace:
    return SimpleNamespace(
        retrieval=SimpleNamespace(
            dense_top_k=4,
            sparse_top_k=5,
            fusion_k=10,
            hard_filters=hard_filters or [],
        )
    )


def make_search(
    processed: ProcessedQuery,
    dense=None,
    sparse=None,
    fusion=None,
    hard_filters=None,
) -> tuple[HybridSearch, RecordingProcessor, RecordingRetriever, RecordingRetriever, RecordingFusion]:
    processor = RecordingProcessor(processed)
    dense = dense or RecordingRetriever()
    sparse = sparse or RecordingRetriever()
    fusion = fusion or RecordingFusion()
    search = HybridSearch(
        settings(hard_filters),
        query_processor=processor,
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=fusion,
    )
    return search, processor, dense, sparse, fusion


def test_search_orchestrates_both_routes_fusion_and_top_k() -> None:
    processed = ProcessedQuery(
        original_query="What is RAG?",
        dense_query="What is RAG?",
        sparse_terms={"rag": 1.0, "retrieval": 0.8},
        filters={"collection": "docs"},
    )
    dense = RecordingRetriever([result("shared", collection="docs")])
    sparse = RecordingRetriever([
        result("shared", collection="docs"), result("sparse", collection="docs")
    ])
    search, processor, dense, sparse, fusion = make_search(
        processed, dense, sparse, hard_filters=["collection"]
    )
    trace = object()

    results = search.search("What is RAG?", top_k=2, filters={"collection": "docs"}, trace=trace)

    assert [item.chunk_id for item in results] == ["shared", "sparse"]
    assert processor.calls == [("What is RAG?", {"collection": "docs"}, trace)]
    assert dense.calls == [(("What is RAG?",), {"top_k": 4, "filters": {"collection": "docs"}, "trace": trace})]
    assert sparse.calls == [(({"rag": 1.0, "retrieval": 0.8},), {"top_k": 5, "trace": trace})]
    assert fusion.calls[0][2:] == (None, trace)


def test_no_hard_filters_skips_pre_filter_but_post_filters_results() -> None:
    processed = ProcessedQuery(
        original_query="RAG",
        dense_query="RAG",
        filters={"doc_type": "pdf"},
    )
    dense = RecordingRetriever([
        result("markdown", doc_type="md"), result("pdf", doc_type="pdf")
    ])
    search, _, dense, _, _ = make_search(processed, dense=dense)

    results = search.search("RAG")

    assert dense.calls[0][1]["filters"] is None
    assert [item.chunk_id for item in results] == ["pdf"]


def test_post_filter_supports_or_values_and_keeps_missing_metadata() -> None:
    candidates = [
        result("pdf", doc_type="pdf", language="zh"),
        result("markdown", doc_type="md", language="en"),
        result("wrong", doc_type="txt", language="zh"),
        result("unknown"),
    ]

    results = HybridSearch._apply_metadata_filters(
        candidates, {"doc_type": ["pdf", "md"], "language": "zh"}
    )

    assert [item.chunk_id for item in results] == ["pdf", "unknown"]


def test_dense_failure_degrades_to_sparse_results() -> None:
    processed = ProcessedQuery(original_query="BM25", dense_query="BM25", sparse_terms={"bm25": 1.0})
    dense = RecordingRetriever(error=RuntimeError("embedding unavailable"))
    sparse = RecordingRetriever([result("sparse")])
    search, _, _, _, fusion = make_search(processed, dense=dense, sparse=sparse)

    results = search.search("BM25")

    assert [item.chunk_id for item in results] == ["sparse"]
    assert fusion.calls[0][0] == []


def test_sparse_failure_degrades_to_dense_results() -> None:
    processed = ProcessedQuery(original_query="RAG", dense_query="RAG", sparse_terms={"rag": 1.0})
    dense = RecordingRetriever([result("dense")])
    sparse = RecordingRetriever(error=RuntimeError("index unavailable"))
    search, _, _, _, fusion = make_search(processed, dense=dense, sparse=sparse)

    results = search.search("RAG")

    assert [item.chunk_id for item in results] == ["dense"]
    assert fusion.calls[0][1] == []


def test_both_route_failures_return_empty_list() -> None:
    processed = ProcessedQuery(original_query="RAG", dense_query="RAG", sparse_terms={"rag": 1.0})
    search, _, _, _, fusion = make_search(
        processed,
        dense=RecordingRetriever(error=RuntimeError()),
        sparse=RecordingRetriever(error=RuntimeError()),
    )

    assert search.search("RAG") == []
    assert fusion.calls == []


def test_empty_dense_query_only_calls_sparse_route() -> None:
    processed = ProcessedQuery(original_query="filter", dense_query="", sparse_terms={"tag": 1.0})
    sparse = RecordingRetriever([result("sparse")])
    search, _, dense, sparse, _ = make_search(processed, sparse=sparse)

    assert [item.chunk_id for item in search.search("filter")] == ["sparse"]
    assert dense.calls == []
    assert len(sparse.calls) == 1


def test_empty_sparse_terms_only_calls_dense_route() -> None:
    processed = ProcessedQuery(original_query="semantic", dense_query="semantic", sparse_terms={})
    dense = RecordingRetriever([result("dense")])
    search, _, dense, sparse, _ = make_search(processed, dense=dense)

    assert [item.chunk_id for item in search.search("semantic")] == ["dense"]
    assert len(dense.calls) == 1
    assert sparse.calls == []


def test_top_k_is_applied_after_post_filtering() -> None:
    processed = ProcessedQuery(original_query="query", dense_query="query", filters={"doc_type": "pdf"})
    dense = RecordingRetriever([
        result("filtered-out", doc_type="md"),
        result("first-pdf", doc_type="pdf"),
        result("second-pdf", doc_type="pdf"),
    ])
    search, _, _, _, fusion = make_search(processed, dense=dense)

    results = search.search("query", top_k=1)

    assert [item.chunk_id for item in results] == ["first-pdf"]
    assert fusion.calls[0][2] is None


@pytest.mark.parametrize(
    ("query", "top_k", "filters", "error"),
    [
        (None, 1, None, "query"),
        ("query", 0, None, "top_k"),
        ("query", True, None, "top_k"),
        ("query", 1, ["collection"], "filters"),
    ],
)
def test_invalid_search_arguments_are_rejected(query, top_k, filters, error: str) -> None:
    processed = ProcessedQuery(original_query="query", dense_query="query")
    search, *_ = make_search(processed)

    with pytest.raises((TypeError, ValueError), match=error):
        search.search(query, top_k=top_k, filters=filters)


def test_routes_run_concurrently() -> None:
    dense_started = Event()
    sparse_started = Event()

    class ConcurrentRetriever(RecordingRetriever):
        def __init__(self, started: Event, other_started: Event, item: RetrievalResult) -> None:
            super().__init__([item])
            self.started = started
            self.other_started = other_started
            self.saw_other = False

        def retrieve(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            self.started.set()
            self.saw_other = self.other_started.wait(timeout=1)
            return self.results

    processed = ProcessedQuery(original_query="query", dense_query="query", sparse_terms={"query": 1.0})
    dense = ConcurrentRetriever(dense_started, sparse_started, result("dense"))
    sparse = ConcurrentRetriever(sparse_started, dense_started, result("sparse"))
    search, *_ = make_search(processed, dense=dense, sparse=sparse)

    search.search("query")

    assert dense.saw_other is True
    assert sparse.saw_other is True


def test_stage_callback_receives_dense_sparse_and_fusion_results() -> None:
    processed = ProcessedQuery(
        original_query="RAG", dense_query="RAG", sparse_terms={"rag": 1.0}
    )
    search, *_ = make_search(
        processed,
        dense=RecordingRetriever([result("dense")]),
        sparse=RecordingRetriever([result("sparse")]),
    )
    stages = {}

    search.search("RAG", on_stage=lambda name, values: stages.setdefault(name, values))

    assert [item.chunk_id for item in stages["dense"]] == ["dense"]
    assert [item.chunk_id for item in stages["sparse"]] == ["sparse"]
    assert [item.chunk_id for item in stages["fusion"]] == ["dense", "sparse"]
