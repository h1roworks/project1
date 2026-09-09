"""C13: ImageStorage 单元测试。

验证：
- 存储初始化：自动建目录/建库/建表、WAL 模式、重复初始化幂等
- 保存：文件落盘到 ``{images_dir}/{collection}/{image_id}.png``、返回相对路径、
  索引记录写入 SQLite（page_num/doc_hash 等字段）
- 查询：get_path / get_record / exists / read_image
- Upsert 幂等：同一 image_id 重复保存覆盖文件与记录，不产生重复条目
- mime_type → 扩展名推断（png/jpeg/...）
- 批量查询：list_images 按 collection / doc_hash 过滤
- 生命周期：delete_image / delete_images 同时移除磁盘文件与索引记录
- 统计：stats 返回图片总数与集合数
- 持久化：关闭后以同一 db_path 重建实例可恢复映射
- 与 core.types.ImageRef 的契约衔接
"""

import pytest

from core.types import ImageRef
from ingestion.storage.image_storage import DEFAULT_DB_PATH, DEFAULT_IMAGES_DIR, ImageStorage


@pytest.fixture
def storage(tmp_path):
    """基于临时目录的图片存储，测试结束关闭连接。"""
    s = ImageStorage(
        images_dir=tmp_path / "images",
        db_path=tmp_path / "db" / "image_index.db",
    )
    yield s
    s.close()


# ---------- 存储初始化 ----------


def test_db_and_dirs_auto_created(tmp_path) -> None:
    db_path = tmp_path / "nested" / "db" / "image_index.db"
    s = ImageStorage(
        images_dir=tmp_path / "nested" / "images",
        db_path=db_path,
    )
    try:
        assert db_path.exists()
    finally:
        s.close()


def test_init_is_idempotent(tmp_path) -> None:
    db_path = tmp_path / "db" / "image_index.db"
    ImageStorage(db_path=db_path).close()
    ImageStorage(db_path=db_path).close()  # 重复初始化不报错


def test_table_has_expected_schema(tmp_path) -> None:
    db_path = tmp_path / "image_index.db"
    s = ImageStorage(db_path=db_path)
    try:
        cols = {row[1]: row[2] for row in s._conn.execute("PRAGMA table_info(image_index)")}
        assert cols == {
            "image_id": "TEXT",
            "file_path": "TEXT",
            "collection": "TEXT",
            "doc_hash": "TEXT",
            "page_num": "INTEGER",
            "created_at": "TIMESTAMP",
        }
    finally:
        s.close()


def test_wal_mode_enabled(tmp_path) -> None:
    s = ImageStorage(db_path=tmp_path / "image_index.db")
    try:
        mode = s._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
    finally:
        s.close()


def test_default_paths() -> None:
    assert DEFAULT_IMAGES_DIR == "data/images"
    assert DEFAULT_DB_PATH == "data/db/image_index.db"


# ---------- 保存 ----------


def test_save_image_writes_file_and_index(storage, tmp_path) -> None:
    data = b"\x89PNG\r\n\x1a\nfake-png-bytes"
    path = storage.save_image(
        "img1", data, collection="docs", doc_hash="abc123", page_num=2
    )
    # 文件按约定落盘
    stored = tmp_path / "images" / "docs" / "img1.png"
    assert stored.exists()
    assert stored.read_bytes() == data
    assert path == stored.as_posix()
    # 索引记录完整
    record = storage.get_record("img1")
    assert record["file_path"] == path
    assert record["collection"] == "docs"
    assert record["doc_hash"] == "abc123"
    assert record["page_num"] == 2


def test_save_image_default_collection(storage, tmp_path) -> None:
    storage.save_image("img1", b"data")
    assert (tmp_path / "images" / "default" / "img1.png").exists()
    assert storage.get_record("img1")["collection"] == "default"


def test_save_image_infers_extension_from_mime(storage, tmp_path) -> None:
    jpg_path = storage.save_image("img1", b"jpeg-bytes", mime_type="image/jpeg")
    assert jpg_path.endswith(".jpg")
    assert (tmp_path / "images" / "default" / "img1.jpg").exists()

    storage.save_image("img2", b"webp-bytes", mime_type="image/webp")
    assert (tmp_path / "images" / "default" / "img2.webp").exists()


def test_save_image_unknown_mime_falls_back_to_png(storage) -> None:
    path = storage.save_image("img1", b"data", mime_type="image/tiff")
    assert path.endswith(".png")


def test_save_same_image_id_is_idempotent(storage, tmp_path) -> None:
    storage.save_image("img1", b"v1", collection="docs", doc_hash="abc", page_num=1)
    storage.save_image("img1", b"v2", collection="docs", doc_hash="abc", page_num=3)

    rows = storage._conn.execute(
        "SELECT COUNT(*) AS n FROM image_index WHERE image_id = 'img1'"
    ).fetchone()
    assert rows["n"] == 1  # 不产生重复条目
    assert storage.read_image("img1") == b"v2"  # 文件被覆盖
    assert storage.get_record("img1")["page_num"] == 3
    assert list((tmp_path / "images" / "docs").iterdir()) == [
        tmp_path / "images" / "docs" / "img1.png"
    ]


# ---------- 查询 ----------


def test_get_path_unknown_returns_none(storage) -> None:
    assert storage.get_path("nope") is None


def test_exists(storage) -> None:
    assert storage.exists("img1") is False
    storage.save_image("img1", b"data")
    assert storage.exists("img1") is True


def test_read_image_roundtrip(storage) -> None:
    storage.save_image("img1", b"fake-image-bytes", collection="docs")
    assert storage.read_image("img1") == b"fake-image-bytes"


def test_read_image_unknown_returns_none(storage) -> None:
    assert storage.read_image("nope") is None


# ---------- 批量查询 ----------


def test_list_images_filters_by_collection(storage) -> None:
    storage.save_image("a1", b"1", collection="docs", doc_hash="h1")
    storage.save_image("a2", b"2", collection="docs", doc_hash="h2")
    storage.save_image("b1", b"3", collection="other", doc_hash="h1")

    assert len(storage.list_images(collection="docs")) == 2
    assert len(storage.list_images(collection="other")) == 1
    assert len(storage.list_images()) == 3  # 不带条件返回全部


def test_list_images_filters_by_doc_hash(storage) -> None:
    storage.save_image("a1", b"1", collection="docs", doc_hash="h1")
    storage.save_image("a2", b"2", collection="docs", doc_hash="h1")
    storage.save_image("b1", b"3", collection="docs", doc_hash="h2")

    result = storage.list_images(collection="docs", doc_hash="h1")
    assert {r["image_id"] for r in result} == {"a1", "a2"}
    assert {r["image_id"] for r in result} != {"b1"}


def test_list_images_empty(storage) -> None:
    assert storage.list_images() == []
    assert storage.list_images(collection="docs") == []


# ---------- 生命周期 ----------


def test_delete_image_removes_file_and_row(storage, tmp_path) -> None:
    storage.save_image("img1", b"data", collection="docs")
    assert storage.delete_image("img1") is True
    assert not (tmp_path / "images" / "docs" / "img1.png").exists()
    assert storage.get_path("img1") is None


def test_delete_image_unknown_returns_false(storage) -> None:
    assert storage.delete_image("nope") is False


def test_delete_images_by_doc_hash(storage, tmp_path) -> None:
    storage.save_image("a1", b"1", collection="docs", doc_hash="h1")
    storage.save_image("a2", b"2", collection="docs", doc_hash="h1")
    storage.save_image("b1", b"3", collection="docs", doc_hash="h2")
    storage.save_image("c1", b"4", collection="other", doc_hash="h1")

    removed = storage.delete_images("docs", doc_hash="h1")
    assert removed == 2
    assert not (tmp_path / "images" / "docs" / "a1.png").exists()
    assert not (tmp_path / "images" / "docs" / "a2.png").exists()
    assert (tmp_path / "images" / "docs" / "b1.png").exists()  # 其他 doc_hash 保留
    assert (tmp_path / "images" / "other" / "c1.png").exists()  # 其他集合保留
    assert len(storage.list_images(collection="docs")) == 1


def test_delete_images_whole_collection(storage) -> None:
    storage.save_image("a1", b"1", collection="docs", doc_hash="h1")
    storage.save_image("a2", b"2", collection="docs", doc_hash="h2")
    storage.save_image("b1", b"3", collection="other", doc_hash="h1")

    assert storage.delete_images("docs") == 2
    assert storage.list_images(collection="docs") == []
    assert len(storage.list_images(collection="other")) == 1


def test_delete_images_unknown_collection_returns_zero(storage) -> None:
    assert storage.delete_images("nothing") == 0


# ---------- 统计 ----------


def test_stats(storage) -> None:
    assert storage.stats() == {"total_images": 0, "total_collections": 0}
    storage.save_image("a1", b"1", collection="docs")
    storage.save_image("a2", b"2", collection="docs")
    storage.save_image("b1", b"3", collection="other")
    assert storage.stats() == {"total_images": 3, "total_collections": 2}


# ---------- 持久化 ----------


def test_mapping_persists_across_reopen(tmp_path) -> None:
    db_path = tmp_path / "db" / "image_index.db"
    images_dir = tmp_path / "images"

    s1 = ImageStorage(images_dir=images_dir, db_path=db_path)
    s1.save_image("img1", b"data", collection="docs", doc_hash="abc", page_num=5)
    s1.close()

    s2 = ImageStorage(images_dir=images_dir, db_path=db_path)
    try:
        assert s2.get_path("img1").endswith("docs/img1.png")
        assert s2.read_image("img1") == b"data"
        assert s2.get_record("img1")["doc_hash"] == "abc"
    finally:
        s2.close()


# ---------- 并发（WAL 模式） ----------


def test_concurrent_connections_share_state(tmp_path) -> None:
    """两个连接写同一数据库，各自能看到对方写入的映射（WAL 跨连接可见）。"""
    db_path = tmp_path / "image_index.db"
    s1 = ImageStorage(images_dir=tmp_path / "imgs", db_path=db_path)
    s2 = ImageStorage(images_dir=tmp_path / "imgs", db_path=db_path)
    try:
        s1.save_image("a", b"1", collection="docs")
        assert s2.get_path("a") is not None  # s2 能读到 s1 的写入
        s2.save_image("b", b"2", collection="docs")
        assert s1.get_path("b") is not None  # s1 能读到 s2 的写入
        assert len(s1.list_images(collection="docs")) == 2
    finally:
        s1.close()
        s2.close()


# ---------- 与 core.types.ImageRef 契约衔接 ----------


def test_builds_image_refs_for_document(storage) -> None:
    """保存后可按约定构造 metadata.images 的 ImageRef 条目，供 Loader/Chunk 使用。"""
    storage.save_image(
        "doc1_p1_0", b"\x89PNG", collection="docs", doc_hash="doc1", page_num=1
    )
    record = storage.get_record("doc1_p1_0")
    ref = ImageRef(id=record["image_id"], path=record["file_path"], page=record["page_num"])
    assert ref.id == "doc1_p1_0"
    assert ref.path.endswith("docs/doc1_p1_0.png")
    assert ref.page == 1
