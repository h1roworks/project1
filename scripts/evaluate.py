#!/usr/bin/env python3
"""H3：对 golden test set 执行检索评估并输出 JSON 报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.settings import load_settings  # noqa: E402
from libs.evaluator.evaluator_factory import EvaluatorFactory  # noqa: E402
from observability.evaluation.eval_runner import EvalRunner  # noqa: E402
from scripts.query import build_query_engine  # noqa: E402

DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_TEST_SET = _PROJECT_ROOT / "tests" / "fixtures" / "golden_test_set.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行知识库 Golden Test Set 评估")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="settings.yaml 路径")
    parser.add_argument("--test-set", default=str(DEFAULT_TEST_SET), help="Golden test set JSON 路径")
    parser.add_argument(
        "--collection",
        default=None,
        help="要评估的 BM25 collection（默认使用 settings.vector_store.collection）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
        hybrid_search, _ = build_query_engine(settings, args.collection)
        evaluator = EvaluatorFactory.create(settings.evaluation)
        report = EvalRunner(settings, hybrid_search, evaluator).run(args.test_set)
    except Exception as exc:  # noqa: BLE001 - CLI converts setup errors to readable output
        print(f"[evaluate] 评估失败：{exc}", file=sys.stderr)
        return 1

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
