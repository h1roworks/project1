"""Tests for persistent Dashboard process management."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "start_dashboard.py"
SPEC = importlib.util.spec_from_file_location("start_dashboard", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


def test_parser_uses_background_mode_by_default() -> None:
    args = dashboard.build_parser().parse_args([])
    assert args.foreground is False
    assert args.status is False
    assert args.stop is False


def test_parser_accepts_management_actions() -> None:
    assert dashboard.build_parser().parse_args(["--foreground"]).foreground is True
    assert dashboard.build_parser().parse_args(["--status"]).status is True
    assert dashboard.build_parser().parse_args(["--stop"]).stop is True


def test_background_start_records_pid_and_uses_detached_mode(monkeypatch, tmp_path) -> None:
    pid_file = tmp_path / "dashboard.pid"
    log_file = tmp_path / "dashboard.log"
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 24680

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(dashboard, "LOG_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "PID_FILE", pid_file)
    monkeypatch.setattr(dashboard, "LOG_FILE", log_file)
    monkeypatch.setattr(dashboard.subprocess, "Popen", fake_popen)

    assert dashboard._start_background(["streamlit", "run", "app.py"]) == 24680
    assert pid_file.read_text(encoding="utf-8") == "24680"
    assert captured["stdin"] == dashboard.subprocess.DEVNULL
    if dashboard.os.name == "nt":
        assert "creationflags" in captured
    else:
        assert captured["start_new_session"] is True
