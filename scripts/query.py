#!/usr/bin/env python3
"""D7：知识库在线查询命令行入口。

示例：

    python scripts/query.py --query "如何配置 Azure" --verbose
    python scripts/query.py --query "RAG 是什么" --collection research --no-rerank

脚本只负责装配与展示：查询逻辑仍由 D1--D6 的 QueryProcessor、
DenseRetriever、SparseRetriever、HybridSearch 和 Reranker 共同完成。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.query_engine.dense_retriever import DenseRetriever  # noqa: E402
from core.query_engine.fusion import Fusion  # noqa: E402
from core.query_engine.hybrid_search import HybridSearch  # noqa: E402
from core.query_engine.query_processor import QueryProcessor  # noqa: E402
from core.query_engine.reranker import Reranker  # noqa: E402
from core.query_engine.sparse_retriever import SparseRetriever  # noqa: E402
from core.settings import Settings, load_settings  # noqa: E402
from core.types import RetrievalResult  # noqa: E402
from ingestion.storage.bm25_indexer import BM25Indexer  # noqa: E402

DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_BM25_DIR = Path("data") / "db" / "bm25"


def _positive_int(value: str) -> int:
    """Argparse 转换器：让 --top-k 的错误在入口处清晰可见。"""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是正整数") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """构造独立、可测试的命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="query",
        description="在本地知识库中执行 HybridSearch，并可选进行 Reranker 精排。",
    )
    parser.add_argument("--query", required=True, help="要检索的问题或关键词")
    parser.add_argument(
        "--top-k", type=_positive_int, default=10, help="返回结果数（默认：10）"
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="要使用的 BM25 集合（默认使用 settings.vector_store.collection）",
    )
    parser.add_argument("--verbose", action="store_true", help="显示各检索阶段的中间结果")
    parser.add_argument("--no-rerank", action="store_true", help="跳过 Reranker 精排阶段")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"settings.yaml 路径（默认：{DEFAULT_CONFIG}）",
    )
    return parser


def bm25_index_path(settings: Settings, collection: str | None) -> Path:
    """返回与 C14 摄取约定一致的 BM25 索引路径。"""
    name = collection or settings.vector_store.collection or "default"
    return DEFAULT_BM25_DIR / f"{name}.pkl"


def build_query_engine(
    settings: Settings,
    collection: str | None = None,
) -> tuple[HybridSearch, Reranker]:
    """显式装配 D1--D6 组件，便于 CLI 和测试复用。

    collection 在 MVP 中选择 BM25 索引文件；向量库集合由 settings 固定，
    与摄取 Pipeline 的既有边界保持一致。
    """
    query_processor = QueryProcessor()
    dense_retriever = DenseRetriever(settings)
    sparse_retriever = SparseRetriever(
        settings,
        bm25_indexer=BM25Indexer(bm25_index_path(settings, collection)),
    )
    fusion = Fusion(k=settings.retrieval.fusion_k)
    hybrid_search = HybridSearch(
        settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion,
    )
    return hybrid_search, Reranker(settings)


def _summary(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[:limit - 1]}…"


def _metadata_value(metadata: dict[str, Any], *names: str, default: str = "-") -> Any:
    for name in names:
        value = metadata.get(name)
        if value not in (None, ""):
            return value
    return default


def print_results(title: str, results: list[RetrievalResult]) -> None:
    """按对初学者友好的格式展示某一检索阶段的结果。"""
    print(f"\n{title}（{len(results)} 条）")
    if not results:
        print("  无结果")
        return

    for index, item in enumerate(results, start=1):
        source = _metadata_value(item.metadata, "source_path", "source", "file_path")
        page = _metadata_value(item.metadata, "page", "page_num", "page_number")
        print(f"{index}. score={item.score:.4f}  chunk_id={item.chunk_id}")
        print(f"   来源：{source}  页码：{page}")
        print(f"   摘要：{_summary(item.text)}")


def main(argv: list[str] | None = None) -> int:
    """运行查询，成功或未命中返回 0；初始化/检索异常返回 1。"""
    args = build_parser().parse_args(argv)

    try:
        settings = load_settings(args.config)
        hybrid_search, reranker = build_query_engine(settings, args.collection)
    except Exception as exc:  # noqa: BLE001 - CLI 需要把初始化问题转成可读提示
        print(f"[query] 初始化失败：{exc}", file=sys.stderr)
        return 1

    stages: dict[str, list[RetrievalResult]] = {}

    def capture_stage(name: str, results: list[RetrievalResult]) -> None:
        stages[name] = results

    try:
        candidates = hybrid_search.search(
            args.query,
            top_k=args.top_k,
            on_stage=capture_stage if args.verbose else None,
        )
        if args.verbose:
            print_results("[verbose] Dense 召回", stages.get("dense", []))
            print_results("[verbose] Sparse 召回", stages.get("sparse", []))
            print_results("[verbose] RRF Fusion", stages.get("fusion", []))

        if args.no_rerank:
            results = candidates
            if args.verbose:
                print("\n[verbose] Rerank：已通过 --no-rerank 跳过")
        else:
            results = reranker.rerank(args.query, candidates)
            if args.verbose:
                print_results("[verbose] Rerank", results)
    except Exception as exc:  # noqa: BLE001 - 对命令行用户呈现而非回溯
        print(f"[query] 查询失败：{exc}", file=sys.stderr)
        return 1

    if not results:
        print("[query] 未找到相关文档，请先运行 ingest.py 摄取数据。")
        return 0

    print_results("[query] Top-K 结果", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
