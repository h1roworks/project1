"""C15: scripts/ingest.py CLI 脚本测试。

scripts/ 不是包（不参与 setuptools 打包），测试用 ``importlib`` 按文件路径加载模块；
monkeypatch ``build_pipeline`` 返回 FakePipeline，避免触碰真实 Chroma/网络。

覆盖：
- 单文件成功：exit 0，stdout 含 collection/chunks，进度回调写入 stderr
- 增量跳过：skipped=True → exit 0，输出 SKIP
- 失败：error 非空 → exit 1，错误进 stderr
- 缺省 --collection：把 None 透传给 pipeline.run（由 settings 兜底）
- 目录模式：按 Loader 扩展名过滤，多文件逐个摄取
- --force：先删除完整性记录再 run（注入临时 checker 验证）
- 空目录/不支持类型：exit 1 且提示
"""

import importlib.util
from pathlib import Path

import pytest

from ingestion.pipeline import IngestionResult
from libs.loader.file_integrity import SQLiteIntegrityChecker

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "ingest.py"

_spec = importlib.util.spec_from_file_location("ingest_script", SCRIPT_PATH)
ingest_script = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(ingest_script)


# ---------- Fake 组件 ----------


class FakeLoader:
    """仅供 ``_resolve_targets`` 读取扩展名。"""

    supported_extensions = (".md", ".txt")


class FakePipeline:
    """记录调用，返回可注入状态的结果，不触碰真实后端。"""

    def __init__(self, error: str | None = None, skipped: bool = False) -> None:
        self._loaders = [FakeLoader()]
        self._error = error
        self._skipped = skipped
        self.calls: list[tuple[str, str | None]] = []

    def run(self, source_path, collection=None, on_progress=None):
        self.calls.append((source_path, collection))
        if on_progress:
            on_progress("load", 1, 1)
            on_progress("upsert", 1, 1)
        return IngestionResult(
            source_path=source_path,
            collection=collection or "default",
            file_hash="deadbeef",
            skipped=self._skipped,
            doc_id="doc",
            total_chunks=0 if self._skipped else 2,
            error=self._error,
        )


# ---------- 测试工具 ----------


def _write_config(tmp_path) -> str:
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        "llm:\n"
        "  provider: dashscope\n"
        "  model: qwen-plus\n"
        "embedding:\n"
        "  provider: dashscope\n"
        "  model: text-embedding-v3\n"
        "vector_store:\n"
        "  provider: chroma\n"
        "  collection: knowledge_hub\n"
        "  persist_dir: data/db/chroma\n"
        "retrieval:\n"
        "  top_k: 10\n"
        "observability:\n"
        "  log_dir: logs\n"
        "  trace_file: logs/traces.jsonl\n",
        encoding="utf-8",
    )
    return str(cfg)


def _write(src: Path, content: str = "rag bm25 content") -> str:
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(content, encoding="utf-8")
    return str(src)


def _make_pipeline(monkeypatch, error=None, skipped=False) -> FakePipeline:
    pipe = FakePipeline(error=error, skipped=skipped)
    monkeypatch.setattr(ingest_script, "build_pipeline", lambda settings: pipe)
    return pipe


# ---------- 测试用例 ----------


def test_single_file_success(tmp_path, monkeypatch, capsys) -> None:
    pipe = _make_pipeline(monkeypatch)
    src = _write(tmp_path / "doc.md")
    rc = ingest_script.main(["--path", src, "--collection", "test", "--config", _write_config(tmp_path)])

    assert rc == 0
    out = capsys.readouterr()
    assert "[ingest] OK" in out.out
    assert "collection=test" in out.out
    assert "chunks=2" in out.out
    assert "[progress]" in out.err  # 进度回调走 stderr
    assert pipe.calls == [(src, "test")]


def test_incremental_skip_exits_zero(tmp_path, monkeypatch, capsys) -> None:
    _make_pipeline(monkeypatch, skipped=True)
    src = _write(tmp_path / "doc.md")
    rc = ingest_script.main(["--path", src, "--config", _write_config(tmp_path)])

    assert rc == 0
    out = capsys.readouterr()
    assert "SKIP" in out.out
    assert "跳过" in out.out


def test_failure_exits_one(tmp_path, monkeypatch, capsys) -> None:
    _make_pipeline(monkeypatch, error="embed failed")
    src = _write(tmp_path / "doc.md")
    rc = ingest_script.main(["--path", src, "--config", _write_config(tmp_path)])

    assert rc == 1
    captured = capsys.readouterr()
    assert "embed failed" in captured.err
    assert "FAIL" in captured.err


def test_collection_defaults_to_none(tmp_path, monkeypatch) -> None:
    pipe = _make_pipeline(monkeypatch)
    src = _write(tmp_path / "doc.md")
    ingest_script.main(["--path", src, "--config", _write_config(tmp_path)])

    # 未传 --collection → None 透传给 pipeline，由 settings.vector_store.collection 兜底
    assert pipe.calls == [(src, None)]


def test_directory_walks_supported_files(tmp_path, monkeypatch, capsys) -> None:
    pipe = _make_pipeline(monkeypatch)
    _write(tmp_path / "dir" / "a.md")
    _write(tmp_path / "dir" / "sub" / "b.txt")
    _write(tmp_path / "dir" / "unsupported.bin")  # 不受支持 → 被过滤
    rc = ingest_script.main(
        ["--path", str(tmp_path / "dir"), "--config", _write_config(tmp_path)]
    )

    assert rc == 0
    assert len(pipe.calls) == 2
    out = capsys.readouterr()
    assert "2 成功" in out.out
    assert "unsupported.bin" not in out.out


def test_empty_directory_reports_and_exits_one(tmp_path, monkeypatch, capsys) -> None:
    _make_pipeline(monkeypatch)
    (tmp_path / "empty").mkdir()
    rc = ingest_script.main(
        ["--path", str(tmp_path / "empty"), "--config", _write_config(tmp_path)]
    )

    assert rc == 1
    assert "没有受支持" in capsys.readouterr().err


def test_missing_path_exits_one(tmp_path, monkeypatch, capsys) -> None:
    _make_pipeline(monkeypatch)
    rc = ingest_script.main(
        ["--path", str(tmp_path / "nope.pdf"), "--config", _write_config(tmp_path)]
    )

    assert rc == 1
    assert "路径不存在" in capsys.readouterr().err


def test_force_removes_integrity_record_before_run(tmp_path) -> None:
    checker = SQLiteIntegrityChecker(db_path=tmp_path / "db" / "ingestion.db")
    src = _write(tmp_path / "doc.md")
    file_hash = checker.compute_sha256(src)
    checker.mark_success(file_hash, src, chunk_count=3)  # 已有成功记录 → 本应跳过

    pipe = FakePipeline()
    try:
        result = ingest_script.ingest_file(
            pipe, src, collection="test", force=True, integrity=checker
        )
    finally:
        checker.close()

    assert result.skipped is False
    # 记录已被删除：pipeline 内部再跑也不会走 should_skip 分支
    checker = SQLiteIntegrityChecker(db_path=tmp_path / "db" / "ingestion.db")
    try:
        assert checker.should_skip(file_hash) is False
    finally:
        checker.close()
    assert pipe.calls == [(src, "test")]


def test_help_flag_prints_usage(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        ingest_script.main(["--help"])
    assert exc.value.code == 0
    assert "--path" in capsys.readouterr().out
