"""D4：RRF 融合的单元测试。"""

import math

import pytest

from core.query_engine.fusion import Fusion
from core.types import RetrievalResult


def make_result(chunk_id: str, score: float = 0.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"text for {chunk_id}",
        metadata={"source_path": f"{chunk_id}.md"},
    )


def test_shared_chunk_accumulates_two_rrf_contributions() -> None:
    fusion = Fusion(k=10)

    results = fusion.fuse(
        [make_result("shared"), make_result("dense-only")],
        [make_result("sparse-only"), make_result("shared")],
    )

    by_id = {result.chunk_id: result for result in results}
    assert math.isclose(by_id["shared"].score, 1 / 11 + 1 / 12)
    assert by_id["shared"].score > by_id["dense-only"].score
    assert by_id["shared"].text == "text for shared"


def test_fusion_uses_rank_not_original_score_scale() -> None:
    fusion = Fusion(k=60)

    results = fusion.fuse(
        [make_result("dense-first", score=0.01), make_result("dense-second", score=0.99)],
        [],
    )

    assert [result.chunk_id for result in results] == ["dense-first", "dense-second"]
    assert results[0].score == 1 / 61


def test_results_from_only_one_route_are_retained() -> None:
    results = Fusion().fuse([make_result("dense")], [make_result("sparse")])

    assert {result.chunk_id for result in results} == {"dense", "sparse"}


def test_empty_routes_return_empty_list() -> None:
    assert Fusion().fuse([], []) == []


def test_top_k_truncates_after_fusion_ranking() -> None:
    results = Fusion(k=10).fuse(
        [make_result("a"), make_result("b"), make_result("c")],
        [make_result("c"), make_result("b"), make_result("a")],
        top_k=2,
    )

    assert len(results) == 2
    assert [result.chunk_id for result in results] == ["a", "c"]


def test_duplicate_ids_in_one_route_are_counted_once() -> None:
    result = Fusion(k=10).fuse(
        [make_result("same"), make_result("same"), make_result("other")],
        [],
    )[0]

    assert result.chunk_id == "same"
    assert result.score == 1 / 11


def test_ties_are_resolved_deterministically_by_chunk_id() -> None:
    results = Fusion(k=10).fuse(
        [make_result("b"), make_result("a")],
        [make_result("a"), make_result("b")],
    )

    assert [result.chunk_id for result in results] == ["a", "b"]


def test_fuse_does_not_mutate_input_result_scores() -> None:
    original = make_result("chunk", score=0.42)

    fused = Fusion().fuse([original], [])

    assert original.score == 0.42
    assert fused[0].score != original.score
    assert fused[0] is not original


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_rrf_k_is_rejected(k) -> None:
    with pytest.raises(ValueError, match="k"):
        Fusion(k=k)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected(top_k) -> None:
    with pytest.raises(ValueError, match="top_k"):
        Fusion().fuse([make_result("chunk")], [], top_k=top_k)
