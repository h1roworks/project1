"""D7：scripts/query.py 的 CLI 装配与输出测试。"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.types import RetrievalResult

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "query.py"

_spec = importlib.util.spec_from_file_location("query_script", SCRIPT_PATH)
query_script = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(query_script)


def result(chunk_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text="这是一个用于验证命令行输出的文档片段。",
        metadata={"source_path": "docs/example.md", "page_num": 3},
    )


class FakeHybridSearch:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls = []

    def search(self, query, top_k, on_stage=None):
        self.calls.append((query, top_k))
        if on_stage:
            on_stage("dense", [result("dense")])
            on_stage("sparse", [result("sparse")])
            on_stage("fusion", self.results)
        return self.results


class FakeReranker:
    def __init__(self) -> None:
        self.calls = []

    def rerank(self, query, candidates):
        self.calls.append((query, candidates))
        return list(reversed(candidates))


def _install_fakes(monkeypatch, results: list[RetrievalResult]):
    hybrid = FakeHybridSearch(results)
    reranker = FakeReranker()
    settings = SimpleNamespace(vector_store=SimpleNamespace(collection="knowledge_hub"))
    monkeypatch.setattr(query_script, "load_settings", lambda path: settings)
    monkeypatch.setattr(
        query_script,
        "build_query_engine",
        lambda loaded_settings, collection: (hybrid, reranker),
    )
    return hybrid, reranker


def test_default_output_runs_hybrid_then_reranker(monkeypatch, capsys) -> None:
    hybrid, reranker = _install_fakes(monkeypatch, [result("first"), result("second")])

    assert query_script.main(["--query", "什么是 RAG？"]) == 0

    captured = capsys.readouterr()
    assert hybrid.calls == [("什么是 RAG？", 10)]
    assert len(reranker.calls) == 1
    assert "[query] Top-K 结果" in captured.out
    assert "来源：docs/example.md" in captured.out
    assert "页码：3" in captured.out


def test_verbose_prints_all_intermediate_stages(monkeypatch, capsys) -> None:
    _install_fakes(monkeypatch, [result("final")])

    assert query_script.main(["--query", "RAG", "--verbose"]) == 0

    output = capsys.readouterr().out
    assert "[verbose] Dense 召回" in output
    assert "[verbose] Sparse 召回" in output
    assert "[verbose] RRF Fusion" in output
    assert "[verbose] Rerank" in output


def test_no_rerank_leaves_hybrid_order_unchanged(monkeypatch, capsys) -> None:
    _, reranker = _install_fakes(monkeypatch, [result("first"), result("second")])

    assert query_script.main(["--query", "RAG", "--no-rerank", "--verbose"]) == 0

    output = capsys.readouterr().out
    assert reranker.calls == []
    assert "已通过 --no-rerank 跳过" in output


def test_empty_results_show_ingestion_hint(monkeypatch, capsys) -> None:
    _install_fakes(monkeypatch, [])

    assert query_script.main(["--query", "不存在的内容"]) == 0

    assert "请先运行 ingest.py 摄取数据" in capsys.readouterr().out


def test_initialization_error_returns_one(monkeypatch, capsys) -> None:
    monkeypatch.setattr(query_script, "load_settings", lambda path: (_ for _ in ()).throw(ValueError("bad config")))

    assert query_script.main(["--query", "RAG"]) == 1

    assert "初始化失败" in capsys.readouterr().err


def test_collection_selects_matching_bm25_index_path() -> None:
    settings = SimpleNamespace(vector_store=SimpleNamespace(collection="knowledge_hub"))

    assert query_script.bm25_index_path(settings, "research") == Path("data/db/bm25/research.pkl")
    assert query_script.bm25_index_path(settings, None) == Path("data/db/bm25/knowledge_hub.pkl")


def test_invalid_top_k_is_rejected(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        query_script.main(["--query", "RAG", "--top-k", "0"])

    assert exc.value.code == 2
    assert "必须是正整数" in capsys.readouterr().err
