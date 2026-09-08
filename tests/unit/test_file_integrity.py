"""C2: 文件完整性检查（SHA256）单元测试。

验证 FileIntegrityChecker 抽象接口与 SQLiteIntegrityChecker 默认实现：
- SHA256 计算（确定性 / 已知值 / 文件不存在）
- SQLite 存储初始化（自动建库建表 / 幂等 / WAL 模式）
- 增量去重判定（should_skip 仅在 status == 'success' 时为 True）
- 状态流转（processing → success / failed）
- 生命周期管理（remove_record / list_processed，供后续 DocumentManager 复用）
"""

import sqlite3

import pytest

from libs.loader.file_integrity import (
    DEFAULT_DB_PATH,
    FileIntegrityChecker,
    SQLiteIntegrityChecker,
)

# SHA256("hello") 的已知参考值，用于确定性断言
_HELLO_SHA256 = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


@pytest.fixture
def checker(tmp_path):
    """基于临时目录的 SQLite 检查器，测试结束关闭连接。"""
    c = SQLiteIntegrityChecker(tmp_path / "db" / "ingestion_history.db")
    yield c
    c.close()


@pytest.fixture
def sample_file(tmp_path):
    """写入一段固定内容的临时文件，返回其路径。"""
    path = tmp_path / "sample.pdf"
    path.write_text("hello", encoding="utf-8")
    return str(path)


# ---------- compute_sha256 ----------

def test_compute_sha256_deterministic(checker, sample_file) -> None:
    assert checker.compute_sha256(sample_file) == checker.compute_sha256(sample_file)


def test_compute_sha256_known_value(checker, sample_file) -> None:
    assert checker.compute_sha256(sample_file) == _HELLO_SHA256


def test_compute_sha256_differs_for_different_content(checker, tmp_path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello", encoding="utf-8")
    b.write_text("world", encoding="utf-8")
    assert checker.compute_sha256(str(a)) != checker.compute_sha256(str(b))


def test_compute_sha256_missing_file_raises(checker, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        checker.compute_sha256(str(tmp_path / "not_exist.pdf"))


# ---------- 存储初始化 ----------

def test_db_file_auto_created(tmp_path) -> None:
    db_path = tmp_path / "nested" / "dir" / "ingestion_history.db"
    c = SQLiteIntegrityChecker(db_path)
    try:
        assert db_path.exists()
    finally:
        c.close()


def test_init_is_idempotent(tmp_path) -> None:
    db_path = tmp_path / "db" / "ingestion_history.db"
    SQLiteIntegrityChecker(db_path).close()
    SQLiteIntegrityChecker(db_path).close()  # 重复初始化不报错


def test_table_has_expected_schema(tmp_path) -> None:
    db_path = tmp_path / "ingestion_history.db"
    c = SQLiteIntegrityChecker(db_path)
    try:
        cols = {
            row[1]: row[2] for row in c._conn.execute("PRAGMA table_info(ingestion_history)")
        }
        assert cols == {
            "file_hash": "TEXT",
            "file_path": "TEXT",
            "file_size": "INTEGER",
            "status": "TEXT",
            "processed_at": "TIMESTAMP",
            "error_msg": "TEXT",
            "chunk_count": "INTEGER",
        }
    finally:
        c.close()


def test_wal_mode_enabled(tmp_path) -> None:
    c = SQLiteIntegrityChecker(tmp_path / "ingestion_history.db")
    try:
        mode = c._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
    finally:
        c.close()


def test_default_db_path_constant() -> None:
    assert DEFAULT_DB_PATH == "data/db/ingestion_history.db"


# ---------- should_skip / 增量去重 ----------

def test_should_skip_false_for_unknown_hash(checker) -> None:
    assert checker.should_skip("deadbeef") is False


def test_should_skip_true_after_mark_success(checker, sample_file) -> None:
    file_hash = checker.compute_sha256(sample_file)
    checker.mark_processing(file_hash, sample_file)
    checker.mark_success(file_hash, sample_file, chunk_count=3)
    assert checker.should_skip(file_hash) is True


def test_should_skip_false_while_processing(checker, sample_file) -> None:
    file_hash = checker.compute_sha256(sample_file)
    checker.mark_processing(file_hash, sample_file)
    assert checker.should_skip(file_hash) is False


def test_should_skip_false_after_mark_failed(checker, sample_file) -> None:
    file_hash = checker.compute_sha256(sample_file)
    checker.mark_processing(file_hash, sample_file)
    checker.mark_failed(file_hash, "boom")
    assert checker.should_skip(file_hash) is False


def test_should_skip_is_content_based(checker, tmp_path) -> None:
    """同名同内容、不同路径的文件应共享同一 hash（按内容去重而非按路径）。"""
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_text("same content", encoding="utf-8")
    b.write_text("same content", encoding="utf-8")
    checker.mark_processing(checker.compute_sha256(str(a)), str(a))
    checker.mark_success(checker.compute_sha256(str(a)), str(a))
    assert checker.should_skip(checker.compute_sha256(str(b))) is True


def test_modified_content_no_longer_skipped(checker, tmp_path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_text("v1", encoding="utf-8")
    checker.mark_processing(checker.compute_sha256(str(path)), str(path))
    checker.mark_success(checker.compute_sha256(str(path)), str(path))
    assert checker.should_skip(checker.compute_sha256(str(path))) is True

    path.write_text("v2", encoding="utf-8")  # 内容变更 → hash 变化
    assert checker.should_skip(checker.compute_sha256(str(path))) is False


# ---------- 状态流转 ----------

def test_get_status_none_for_unknown(checker) -> None:
    assert checker.get_status("nope") is None


def test_mark_processing_creates_row(checker, sample_file) -> None:
    file_hash = checker.compute_sha256(sample_file)
    checker.mark_processing(file_hash, sample_file)
    row = checker._conn.execute(
        "SELECT * FROM ingestion_history WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    assert row["status"] == "processing"
    assert row["file_path"] == sample_file
    assert row["file_size"] == 5
    assert checker.get_status(file_hash) == "processing"


def test_mark_processing_is_upsert(checker, sample_file) -> None:
    file_hash = checker.compute_sha256(sample_file)
    checker.mark_processing(file_hash, sample_file)
    checker.mark_processing(file_hash, sample_file)  # 重复调用不产生重复行
    rows = checker._conn.execute(
        "SELECT COUNT(*) AS n FROM ingestion_history WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    assert rows["n"] == 1


def test_mark_success_records_chunk_count(checker, sample_file) -> None:
    file_hash = checker.compute_sha256(sample_file)
    checker.mark_processing(file_hash, sample_file)
    checker.mark_success(file_hash, sample_file, chunk_count=7)
    row = checker._conn.execute(
        "SELECT * FROM ingestion_history WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    assert row["status"] == "success"
    assert row["chunk_count"] == 7
    assert row["error_msg"] is None


def test_mark_failed_records_error_msg(checker, sample_file) -> None:
    file_hash = checker.compute_sha256(sample_file)
    checker.mark_processing(file_hash, sample_file)
    checker.mark_failed(file_hash, "parse error")
    row = checker._conn.execute(
        "SELECT * FROM ingestion_history WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error_msg"] == "parse error"


def test_mark_failed_overwrites_success(checker, sample_file) -> None:
    """成功后再处理失败：应覆盖为 failed，且不再跳过。"""
    file_hash = checker.compute_sha256(sample_file)
    checker.mark_success(file_hash, sample_file)
    checker.mark_failed(file_hash, "rerun failed")
    assert checker.get_status(file_hash) == "failed"
    assert checker.should_skip(file_hash) is False


def test_mark_success_can_be_called_without_processing(checker, sample_file) -> None:
    """即使未先调用 mark_processing，mark_success 也应能创建记录（upsert 语义）。"""
    file_hash = checker.compute_sha256(sample_file)
    checker.mark_success(file_hash, sample_file, chunk_count=1)
    assert checker.get_status(file_hash) == "success"


# ---------- 生命周期管理（供 DocumentManager 复用） ----------

def test_remove_record_deletes(checker, sample_file) -> None:
    file_hash = checker.compute_sha256(sample_file)
    checker.mark_success(file_hash, sample_file)
    assert checker.should_skip(file_hash) is True
    assert checker.remove_record(file_hash) == 1
    assert checker.should_skip(file_hash) is False
    assert checker.get_status(file_hash) is None


def test_remove_record_missing_returns_zero(checker) -> None:
    assert checker.remove_record("does-not-exist") == 0


def test_list_processed_returns_dicts(checker, sample_file) -> None:
    file_hash = checker.compute_sha256(sample_file)
    checker.mark_success(file_hash, sample_file, chunk_count=2)
    records = checker.list_processed()
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, dict)
    assert record["file_hash"] == file_hash
    assert record["file_path"] == sample_file
    assert record["status"] == "success"
    assert record["chunk_count"] == 2


def test_list_processed_orders_newest_first(checker, tmp_path) -> None:
    for name in ("a.txt", "b.txt"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        file_hash = checker.compute_sha256(str(path))
        checker.mark_success(file_hash, str(path))
    records = checker.list_processed()
    assert len(records) == 2
    assert records[0]["file_path"].endswith("b.txt")  # 后写入的排前面


def test_list_processed_empty(checker) -> None:
    assert checker.list_processed() == []


# ---------- 抽象接口 ----------

def test_file_integrity_checker_is_abstract() -> None:
    with pytest.raises(TypeError):
        FileIntegrityChecker()  # type: ignore[abstract]


def test_sqlite_checker_is_a_file_integrity_checker(checker) -> None:
    assert isinstance(checker, FileIntegrityChecker)


# ---------- 并发写入（WAL 模式） ----------

def test_concurrent_connections_share_state(tmp_path) -> None:
    """两个连接写同一数据库文件，各自能看到对方写入的数据（WAL 跨连接可见）。"""
    db_path = tmp_path / "ingestion_history.db"
    c1 = SQLiteIntegrityChecker(db_path)
    c2 = SQLiteIntegrityChecker(db_path)
    try:
        c1.mark_success("hash-a", "a.pdf", chunk_count=1)
        assert c2.should_skip("hash-a") is True  # c2 能读到 c1 的写入
        c2.mark_success("hash-b", "b.pdf", chunk_count=2)
        assert c1.should_skip("hash-b") is True  # c1 能读到 c2 的写入
        assert len(c1.list_processed()) == 2
    finally:
        c1.close()
        c2.close()


# ---------- 数据库约束 ----------

def test_invalid_status_rejected(tmp_path) -> None:
    c = SQLiteIntegrityChecker(tmp_path / "ingestion_history.db")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            c._conn.execute(
                "INSERT INTO ingestion_history (file_hash, file_path, status) "
                "VALUES ('h', 'p', 'invalid_status')"
            )
    finally:
        c.close()
