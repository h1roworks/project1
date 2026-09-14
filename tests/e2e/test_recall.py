"""H5: Golden Set recall regression test.

This test deliberately uses the real settings, query engine, Chroma store and
BM25 index.  A checkout without locally ingested Golden Set chunks cannot
produce a meaningful recall number, so it skips with an actionable message
instead of turning an environment issue into a false regression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import load_settings
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.evaluator.custom_evaluator import CustomEvaluator
from observability.evaluation.eval_runner import EvalRunner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
GOLDEN_SET_PATH = PROJECT_ROOT / "tests" / "fixtures" / "golden_test_set.json"
MIN_HIT_RATE_AT_K = 0.50


def _golden_ids() -> set[str]:
    cases = EvalRunner._load_test_cases(GOLDEN_SET_PATH)
    return {
        chunk_id
        for case in cases
        for chunk_id in case["expected_chunk_ids"]
    }


@pytest.mark.e2e
def test_golden_set_recall_at_k_does_not_regress() -> None:
    """The configured retrieval pipeline must meet the minimum recall baseline."""
    try:
        settings = load_settings(CONFIG_PATH)
    except Exception as exc:  # Configuration is an environment prerequisite.
        pytest.skip(f"无法加载本地配置，跳过 Recall E2E：{exc}")

    collection = settings.vector_store.collection or "default"
    bm25_path = PROJECT_ROOT / "data" / "db" / "bm25" / f"{collection}.pkl"
    chroma_path = Path(settings.vector_store.persist_dir or "data/db/chroma")
    if not chroma_path.is_absolute():
        chroma_path = PROJECT_ROOT / chroma_path
    if not bm25_path.exists() or not chroma_path.exists():
        pytest.skip(
            "未找到当前配置对应的 Chroma/BM25 索引；请先摄取 Golden Set 文档后再运行 Recall E2E。"
        )

    try:
        indexed_ids = set(BM25Indexer.load_from(bm25_path).documents)
    except Exception as exc:
        pytest.skip(f"无法读取 BM25 索引，跳过 Recall E2E：{exc}")
    missing_ids = sorted(_golden_ids() - indexed_ids)
    if missing_ids:
        pytest.skip(
            "当前索引缺少 Golden Set Chunk（例如："
            f"{missing_ids[0]}）；请先摄取匹配文档。"
        )

    # Import lazily so collecting unit tests does not initialize retrieval clients.
    from scripts.query import build_query_engine

    hybrid_search, _ = build_query_engine(settings)
    report = EvalRunner(settings, hybrid_search, CustomEvaluator()).run(GOLDEN_SET_PATH)

    assert report.hit_rate >= MIN_HIT_RATE_AT_K, (
        f"Golden Set hit_rate@K={report.hit_rate:.3f} 低于阈值 "
        f"{MIN_HIT_RATE_AT_K:.3f}；逐 query 结果：{report.to_dict()['case_results']}"
    )
