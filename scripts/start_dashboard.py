#!/usr/bin/env python3
"""Start the local Streamlit Dashboard with the project's configured host/port."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.settings import load_settings  # noqa: E402

DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"
APP_PATH = _PROJECT_ROOT / "src" / "observability" / "dashboard" / "app.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 Smart Knowledge Hub Dashboard")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="settings.yaml 的路径")
    parser.add_argument("--host", default=None, help="覆盖配置中的 Dashboard host")
    parser.add_argument("--port", type=int, default=None, help="覆盖配置中的 Dashboard port")
    return parser


def build_command(host: str, port: int) -> list[str]:
    """Build the subprocess command separately so it is easy to inspect/test."""
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
    except Exception as exc:  # noqa: BLE001 - command-line users need a clear error.
        print(f"[dashboard] 配置加载失败：{exc}", file=sys.stderr)
        return 1

    dashboard_config = settings.observability.dashboard
    host = args.host or str(dashboard_config.get("host", "localhost"))
    port = args.port or int(dashboard_config.get("port", 8501))
    if not 1 <= port <= 65535:
        print("[dashboard] port 必须在 1 到 65535 之间", file=sys.stderr)
        return 1

    return subprocess.run(build_command(host, port), cwd=_PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
