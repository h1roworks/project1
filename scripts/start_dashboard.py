#!/usr/bin/env python3
"""Start and manage the local Streamlit Dashboard.

By default the Dashboard runs in a detached background process, so closing the
terminal that launched it does not stop the local web application. Use
``--foreground`` while developing, and ``--status`` / ``--stop`` to manage an
existing background instance.
"""

from __future__ import annotations

import argparse
import os
import signal
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
LOG_DIR = _PROJECT_ROOT / "logs"
PID_FILE = LOG_DIR / "dashboard.pid"
LOG_FILE = LOG_DIR / "dashboard.log"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 Knowledge Hub v1 Dashboard")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="settings.yaml 路径")
    parser.add_argument("--host", default=None, help="覆盖 Dashboard host")
    parser.add_argument("--port", type=int, default=None, help="覆盖 Dashboard port")
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--foreground",
        action="store_true",
        help="在当前终端前台运行（关闭终端后会停止）",
    )
    action_group.add_argument(
        "--status",
        action="store_true",
        help="显示后台 Dashboard 的运行状态",
    )
    action_group.add_argument(
        "--stop",
        action="store_true",
        help="停止当前后台 Dashboard",
    )
    return parser


def build_command(host: str, port: int) -> list[str]:
    """Build the Streamlit command separately so it is easy to inspect/test."""
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
        "--browser.gatherUsageStats=false",
    ]


def _read_pid() -> int | None:
    """Return the managed dashboard PID, discarding an invalid PID file."""
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _is_running(pid: int) -> bool:
    """Check whether a process is still alive on Windows and POSIX systems."""
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _clear_stale_pid() -> int | None:
    pid = _read_pid()
    if pid is not None and _is_running(pid):
        return pid
    PID_FILE.unlink(missing_ok=True)
    return None


def _start_background(command: list[str]) -> int:
    """Launch Streamlit independently from this terminal and return its PID."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    popen_args: dict[str, object] = {
        "cwd": _PROJECT_ROOT,
        "stdin": subprocess.DEVNULL,
    }
    with LOG_FILE.open("a", encoding="utf-8") as log_file:
        popen_args["stdout"] = log_file
        popen_args["stderr"] = subprocess.STDOUT
        if os.name == "nt":
            popen_args["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
        else:
            popen_args["start_new_session"] = True
        process = subprocess.Popen(command, **popen_args)  # type: ignore[arg-type]
    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    return process.pid


def _stop_background(pid: int) -> None:
    """Stop the managed process and any child Streamlit worker it owns."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        os.kill(pid, signal.SIGTERM)
    PID_FILE.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    running_pid = _clear_stale_pid()
    if args.status:
        if running_pid is None:
            print("[dashboard] not running")
            return 1
        print(f"[dashboard] running (PID {running_pid})")
        return 0
    if args.stop:
        if running_pid is None:
            print("[dashboard] not running")
            return 0
        _stop_background(running_pid)
        print(f"[dashboard] stopped (PID {running_pid})")
        return 0

    try:
        settings = load_settings(args.config)
    except Exception as exc:  # noqa: BLE001 - command-line users need a clear error.
        print(f"[dashboard] 配置加载失败: {exc}", file=sys.stderr)
        return 1

    dashboard_config = settings.observability.dashboard
    host = args.host or str(dashboard_config.get("host", "localhost"))
    port = args.port or int(dashboard_config.get("port", 8501))
    if not 1 <= port <= 65535:
        print("[dashboard] port 必须在 1 到 65535 之间", file=sys.stderr)
        return 1

    url = f"http://{host}:{port}"
    if args.foreground:
        return subprocess.run(build_command(host, port), cwd=_PROJECT_ROOT, check=False).returncode
    if running_pid is not None:
        print(f"[dashboard] already running (PID {running_pid}): {url}")
        return 0

    pid = _start_background(build_command(host, port))
    print(f"[dashboard] started in background (PID {pid}): {url}")
    print(f"[dashboard] logs: {LOG_FILE}")
    print("[dashboard] stop with: python scripts/start_dashboard.py --stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
