"""Integration tests for the MCP stdio entry point."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys

from mcp import types
from core.response.multimodal_assembler import MultimodalAssembler
from core.types import RetrievalResult
from ingestion.storage.image_storage import ImageStorage
from mcp_server.server import create_server
from mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool

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
        assert {tool["name"] for tool in tools_response["result"]["tools"]} == {
            "query_knowledge_hub",
            "list_collections",
            "get_document_summary",
        }
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

    class FakeCollectionsTool:
        def __call__(self):
            return {
                "content": [{"type": "text", "text": "Collections: course-notes"}],
                "structuredContent": {"collections": [{"name": "course-notes"}]},
            }

    class FakeDocumentSummaryTool:
        def __call__(self, doc_id: str):
            return {
                "content": [{"type": "text", "text": f"Summary: {doc_id}"}],
                "structuredContent": {"document": {"doc_id": doc_id}},
            }

    async def exercise_handlers() -> None:
        server = create_server(
            query_tool=FakeQueryTool(),
            collections_tool=FakeCollectionsTool(),
            document_summary_tool=FakeDocumentSummaryTool(),
        )
        list_handler = server.get_request_handler("tools/list")
        call_handler = server.get_request_handler("tools/call")
        assert list_handler is not None
        assert call_handler is not None

        listed = await list_handler.handler(None, types.PaginatedRequestParams())
        schemas = {tool.name: tool.input_schema for tool in listed.tools}
        assert "query" in schemas["query_knowledge_hub"]["properties"]
        assert schemas["list_collections"]["properties"] == {}
        assert "doc_id" in schemas["get_document_summary"]["properties"]

        called = await call_handler.handler(
            None,
            types.CallToolRequestParams(
                name="query_knowledge_hub", arguments={"query": "What is RAG?"}
            ),
        )
        assert called.content[0].text == "Answer for: What is RAG?"
        assert called.structured_content == {"answer": "What is RAG?", "citations": []}

        collections_called = await call_handler.handler(
            None, types.CallToolRequestParams(name="list_collections", arguments={})
        )
        assert collections_called.content[0].text == "Collections: course-notes"

        summary_called = await call_handler.handler(
            None,
            types.CallToolRequestParams(
                name="get_document_summary", arguments={"doc_id": "doc-123"}
            ),
        )
        assert summary_called.content[0].text == "Summary: doc-123"

    import asyncio

    asyncio.run(exercise_handlers())


def test_server_returns_base64_image_content_for_retrieved_image(tmp_path) -> None:
    """A chunk image reference becomes a standard MCP ImageContent item."""
    image_bytes = b"\x89PNG\r\n\x1a\nimage-bytes"
    storage = ImageStorage(
        images_dir=tmp_path / "images", db_path=tmp_path / "db" / "images.db"
    )
    storage.save_image("architecture", image_bytes, collection="course")

    class FakeHybridSearch:
        def search(self, _query, top_k, filters=None, trace=None):
            assert top_k == 1
            assert filters is None
            return [
                RetrievalResult(
                    chunk_id="chunk-1",
                    score=0.9,
                    text="系统架构图说明。",
                    metadata={
                        "source_path": "docs/architecture.md",
                        "image_refs": ["architecture"],
                    },
                )
            ]

    class FakeReranker:
        def rerank(self, _query, candidates, trace=None):
            return candidates

    query_tool = QueryKnowledgeHubTool(
        hybrid_search=FakeHybridSearch(),
        reranker=FakeReranker(),
        default_top_k=1,
        multimodal_assembler=MultimodalAssembler(image_storage=storage),
        trace_writer=lambda _trace: None,
    )

    async def call_query_tool() -> None:
        server = create_server(
            query_tool=query_tool,
            collections_tool=lambda: {"content": [], "structuredContent": {}},
            document_summary_tool=lambda _doc_id: {"content": [], "structuredContent": {}},
        )
        handler = server.get_request_handler("tools/call")
        assert handler is not None

        result = await handler.handler(
            None,
            types.CallToolRequestParams(
                name="query_knowledge_hub", arguments={"query": "展示架构图"}
            ),
        )
        assert result.content[0].type == "text"
        assert result.content[1].type == "image"
        assert result.content[1].data == base64.b64encode(image_bytes).decode("ascii")
        assert result.content[1].mime_type == "image/png"

    import asyncio

    try:
        asyncio.run(call_query_tool())
    finally:
        storage.close()
