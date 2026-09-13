"""Integration tests for the MCP stdio entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from mcp import types
from mcp_server.server import create_server

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_server_initializes_and_keeps_stdout_protocol_only() -> None:
    """A real child process completes initialize without log pollution."""
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
        assert process.stdin is not None
        assert process.stdout is not None
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0.1.0"},
            },
        }
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()

        response_line = process.stdout.readline()
        response = json.loads(response_line)
        assert response["id"] == 1
        assert response["result"]["serverInfo"]["name"] == "smart-knowledge-hub"
        assert "capabilities" in response["result"]

        process.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
            + "\n"
        )
        process.stdin.write(
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            + "\n"
        )
        process.stdin.flush()

        tools_response = json.loads(process.stdout.readline())
        assert tools_response["id"] == 2
        assert tools_response["result"]["tools"][0]["name"] == "query_knowledge_hub"
    finally:
        process.terminate()
        _, stderr = process.communicate(timeout=5)

    assert "Starting MCP server with stdio transport" in stderr


def test_server_lists_and_calls_the_query_tool() -> None:
    """The SDK server exposes the E3 tool without initializing real backends."""

    class FakeQueryTool:
        def __call__(self, query: str, **_kwargs):
            return {
                "content": [{"type": "text", "text": f"Answer for: {query}"}],
                "structuredContent": {"answer": query, "citations": []},
            }

    async def exercise_handlers() -> None:
        server = create_server(query_tool=FakeQueryTool())
        list_handler = server.get_request_handler("tools/list")
        call_handler = server.get_request_handler("tools/call")
        assert list_handler is not None
        assert call_handler is not None

        listed = await list_handler.handler(None, types.PaginatedRequestParams())
        assert listed.tools[0].name == "query_knowledge_hub"
        assert "query" in listed.tools[0].input_schema["properties"]

        called = await call_handler.handler(
            None,
            types.CallToolRequestParams(
                name="query_knowledge_hub", arguments={"query": "What is RAG?"}
            ),
        )
        assert called.content[0].text == "Answer for: What is RAG?"
        assert called.structured_content == {"answer": "What is RAG?", "citations": []}

    import asyncio

    asyncio.run(exercise_handlers())
