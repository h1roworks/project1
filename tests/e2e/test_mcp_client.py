"""I1: exercise the MCP server from a real MCP client process boundary."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_SERVER_PATH = Path(__file__).with_name("mcp_test_server.py")


def _server_parameters() -> StdioServerParameters:
    """Build child-process settings without relying on the active shell."""
    environment = os.environ.copy()
    source_path = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    return StdioServerParameters(
        command=sys.executable,
        args=[str(TEST_SERVER_PATH)],
        env=environment,
        cwd=PROJECT_ROOT,
    )


@pytest.mark.e2e
def test_mcp_client_lists_tools_and_receives_query_citations() -> None:
    """A client can discover and call the query tool through stdio MCP."""

    async def exercise_client() -> None:
        async with stdio_client(_server_parameters()) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                tools = await session.list_tools()
                assert "query_knowledge_hub" in {tool.name for tool in tools.tools}

                result = await session.call_tool(
                    "query_knowledge_hub",
                    arguments={"query": "What is RAG?", "top_k": 1},
                )

                assert result.is_error is False
                assert result.structured_content is not None
                citations = result.structured_content["citations"]
                assert citations == [
                    {
                        "source": "fixtures/rag-introduction.md",
                        "page": 1,
                        "chunk_id": "e2e-rag-chunk",
                        "score": 0.99,
                        "text": "RAG combines retrieved context with a language model.",
                    }
                ]
                assert result.content[0].type == "text"
                assert "fixtures/rag-introduction.md" in result.content[0].text

    asyncio.run(exercise_client())
