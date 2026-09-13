#!/usr/bin/env python3
"""E 阶段学习实验：用一个最小 MCP Client 测试本地 Server。

默认实验不依赖 Ollama、向量库或已摄取文档，因此适合先理解 E 阶段的
stdio / JSON-RPC / tools 三个概念：

    .venv\\Scripts\\python.exe scripts/e_phase_mcp_demo.py

若已经完成摄取并启动了 Ollama，可额外测试真实检索：

    .venv\\Scripts\\python.exe scripts/e_phase_mcp_demo.py --query "什么是 RAG？"

若知道摄取输出的 doc_id，可测试 E5：

    .venv\\Scripts\\python.exe scripts/e_phase_mcp_demo.py --doc-id "你的文档ID"
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Windows PowerShell may otherwise use GBK for this script's explanatory text,
# while MCP itself always uses UTF-8 JSON.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试并学习 E 阶段 MCP Server。")
    parser.add_argument("--query", help="可选：调用真实 query_knowledge_hub 工具")
    parser.add_argument("--doc-id", help="可选：调用 get_document_summary 工具")
    return parser


def send(process: subprocess.Popen[str], message: dict[str, Any]) -> None:
    """Write exactly one JSON-RPC message to the server's stdin."""
    assert process.stdin is not None
    process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
    process.stdin.flush()


def receive(process: subprocess.Popen[str]) -> dict[str, Any]:
    """Read exactly one JSON-RPC response from the server's stdout."""
    assert process.stdout is not None
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("Server closed stdout before returning a response.")
    return json.loads(line)


def display(title: str, response: dict[str, Any]) -> None:
    """Print a response without filling the terminal with Base64 image data."""
    visible = copy.deepcopy(response)
    content = visible.get("result", {}).get("content", [])
    for item in content:
        if item.get("type") == "image" and isinstance(item.get("data"), str):
            item["data"] = f"<Base64 image, {len(item['data'])} characters>"
    print(f"\n{'=' * 12} {title} {'=' * 12}")
    print(json.dumps(visible, ensure_ascii=False, indent=2))


def request(
    process: subprocess.Popen[str], request_id: int, method: str, params: dict[str, Any]
) -> dict[str, Any]:
    send(
        process,
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
    )
    return receive(process)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    environment = os.environ.copy()
    source_path = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")

    process = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        cwd=PROJECT_ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        print("实验开始：本脚本扮演 MCP Client，子进程扮演 MCP Server。")
        initialize = request(
            process,
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "e-phase-learning-demo", "version": "1.0"},
            },
        )
        display("E1/E2：initialize 握手", initialize)

        # This is a notification, so a correct server sends no response.
        send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        tools = request(process, 2, "tools/list", {})
        display("E2：发现已注册工具", tools)

        collections = request(process, 3, "tools/call", {"name": "list_collections", "arguments": {}})
        display("E4：浏览本地集合", collections)

        if args.query:
            query_response = request(
                process,
                4,
                "tools/call",
                {"name": "query_knowledge_hub", "arguments": {"query": args.query}},
            )
            display("E3/E6：检索结果（可能含图片）", query_response)

        if args.doc_id:
            summary_response = request(
                process,
                5,
                "tools/call",
                {"name": "get_document_summary", "arguments": {"doc_id": args.doc_id}},
            )
            display("E5：文档摘要", summary_response)
    finally:
        process.terminate()
        _, stderr = process.communicate(timeout=5)
        if stderr:
            print("\n========== E1：Server stderr 日志 ==========")
            print(stderr.strip())

    print("\n实验完成：stdout 中只有 JSON-RPC 响应；日志被独立放在 stderr。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
