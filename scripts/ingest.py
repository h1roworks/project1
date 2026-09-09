#!/usr/bin/env python3
"""C15: 离线数据摄取脚本（CLI 入口）。

对齐 DEV_SPEC 5.3.3：CLI 参数解析，调用 ``IngestionPipeline``，支持
``--path`` / ``--collection`` / ``--force``：

- ``--path``：单个文件，或一个目录（按已注册 Loader 的扩展名过滤后逐个摄取）；
- ``--collection``：目标集合，缺省由 ``settings.vector_store.collection`` 决定；
- ``--force``：忽略 FileIntegrity 的增量跳过，删除处理记录后强制重新摄取；
- ``--config``：settings.yaml 路径，缺省 ``config/settings.yaml``。

``on_progress`` 进度回调写入 stderr（stage, current, total），stdout 保持干净的
结果摘要，便于管道重定向。

用法示例::

    python scripts/ingest.py --path docs/论文.pdf --collection 研究
    python scripts/ingest.py --path data/documents/ --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 管道/重定向场景下 Windows 默认用 GBK 写 stdout，统一为 UTF-8 避免中文乱码。
# （终端场景 Python 已是 UTF-8，此处幂等无副作用。）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # 流不可重配置时保持原样
        pass

# 允许未安装包时也能直接运行：把项目 src/ 加入模块搜索路径（幂等，已存在则跳过）。
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.settings import Settings, load_settings  # noqa: E402
from ingestion.pipeline import IngestionPipeline, IngestionResult  # noqa: E402
from libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker  # noqa: E402

DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器（独立函数便于测试）。"""
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="把文档离线摄取进知识库（Chroma 稠密向量 + BM25 稀疏索引 + 图片）。",
    )
    parser.add_argument("--path", required=True, help="待摄取的文件或目录路径")
    parser.add_argument(
        "--collection",
        default=None,
        help="目标集合（缺省取 settings.vector_store.collection）",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"settings.yaml 路径（缺省 {DEFAULT_CONFIG}）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略增量跳过，删除 FileIntegrity 记录后强制重新摄取",
    )
    return parser


def build_pipeline(settings: Settings) -> IngestionPipeline:
    """从 Settings 经工厂自动装配完整 pipeline（真实后端）。"""
    return IngestionPipeline(settings=settings)


def _resolve_targets(pipeline: IngestionPipeline, path: str) -> list[Path]:
    """把 ``--path`` 解析为待摄取文件列表。

    - 单文件：原样返回；
    - 目录：按各 Loader 的 ``supported_extensions`` 过滤，递归收集后按路径排序
      （访问 ``_loaders`` 是 CLI 对装配结果的只读借用，避免为此给 Pipeline 加接口）。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    if p.is_file():
        return [p]

    supported = {
        ext for loader in pipeline._loaders for ext in loader.supported_extensions
    }
    return sorted(
        f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in supported
    )


def ingest_file(
    pipeline: IngestionPipeline,
    path: str,
    collection: str | None = None,
    force: bool = False,
    on_progress=None,
    integrity: FileIntegrityChecker | None = None,
) -> IngestionResult:
    """摄取单个文件；``force=True`` 时先删除完整性记录（可注入 checker 便于测试）。"""
    if force:
        checker = integrity or SQLiteIntegrityChecker()
        owned = integrity is None
        try:
            checker.remove_record(checker.compute_sha256(path))
        finally:
            if owned:
                checker.close()
    return pipeline.run(path, collection=collection, on_progress=on_progress)


def _make_progress_callback():
    """进度回调写入 stderr，保持 stdout 只有结果摘要。"""

    def on_progress(stage: str, current: int, total: int) -> None:
        sys.stderr.write(f"[progress] {stage}: {current}/{total}\n")
        sys.stderr.flush()

    return on_progress


def _print_result(result: IngestionResult) -> None:
    """按结果状态输出一行摘要（OK/SKIP/FAIL）。"""
    if result.skipped:
        print(f"[ingest] SKIP  {result.source_path}  （文件未变更，增量跳过）")
        return
    if result.error:
        print(f"[ingest] FAIL  {result.source_path}: {result.error}", file=sys.stderr)
        return
    print(f"[ingest] OK    {result.source_path}")
    print(f"         collection={result.collection}  doc_id={result.doc_id}")
    print(f"         chunks={result.total_chunks}  images={result.total_images}")


def main(argv: list[str] | None = None) -> int:
    """CLI 主流程。返回进程退出码：全成功/跳过 0，有失败 1。"""
    args = build_parser().parse_args(argv)

    try:
        settings = load_settings(args.config)
        pipeline = build_pipeline(settings)
    except Exception as exc:  # noqa: BLE001 - CLI 兜底，给出可读错误
        print(f"[ingest] 配置加载失败: {exc}", file=sys.stderr)
        return 1

    try:
        targets = _resolve_targets(pipeline, args.path)
    except FileNotFoundError as exc:
        print(f"[ingest] {exc}", file=sys.stderr)
        return 1

    if not targets:
        print("[ingest] 目录中没有受支持的文档文件（仅处理 Loader 可解析的类型）", file=sys.stderr)
        return 1

    on_progress = _make_progress_callback()
    ok = skipped = failed = 0
    for target in targets:
        try:
            result = ingest_file(
                pipeline, str(target), args.collection, force=args.force,
                on_progress=on_progress,
            )
        except Exception as exc:  # noqa: BLE001 - 单文件异常不中断整体
            print(f"[ingest] FAIL  {target}: {exc}", file=sys.stderr)
            failed += 1
            continue
        if result.skipped:
            skipped += 1
        elif result.error:
            failed += 1
        else:
            ok += 1
        _print_result(result)

    print(f"[ingest] 完成: {ok} 成功, {skipped} 跳过, {failed} 失败（共 {len(targets)} 个文件）")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
