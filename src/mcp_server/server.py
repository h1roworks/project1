"""MCP server entry point using the standard input/output transport.

The transport has one important rule: stdout is reserved for JSON-RPC messages
that an MCP client reads.  Application logs must therefore use stderr.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from mcp_server.tools.query_knowledge_hub import (
    QUERY_KNOWLEDGE_HUB_SCHEMA,
    QueryKnowledgeHubTool,
)
from observability.logger import get_logger

SERVER_NAME = "smart-knowledge-hub"
SERVER_VERSION = "0.1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = get_logger(__name__)


def create_server(query_tool: QueryKnowledgeHubTool | None = None) -> Server:
    """Create the MCP server.

    The query tool is created lazily: starting the server does not require an
    embedding model or a populated local knowledge base.
    """
    server = Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        title="Smart Knowledge Hub",
        description="MCP interface for the Smart Knowledge Hub.",
    )
    tool = query_tool or QueryKnowledgeHubTool.from_config(
        PROJECT_ROOT / "config" / "settings.yaml"
    )
    _register_query_tool(server, tool)
    return server


def _register_query_tool(server: Server, query_tool: QueryKnowledgeHubTool) -> None:
    """Expose the E3 query function through the official MCP SDK handlers."""

    async def list_tools(_context: Any, _params: types.PaginatedRequestParams) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="query_knowledge_hub",
                    description="使用混合检索与重排查询本地知识库，并返回可追溯引用。",
                    inputSchema=QUERY_KNOWLEDGE_HUB_SCHEMA,
                )
            ]
        )

    async def call_tool(_context: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name != "query_knowledge_hub":
            return types.CallToolResult(
                content=[types.TextContent(text=f"Unknown tool: {params.name}")],
                isError=True,
            )
        try:
            response = query_tool(**(params.arguments or {}))
        except (TypeError, ValueError) as exc:
            return types.CallToolResult(
                content=[types.TextContent(text=f"Invalid tool arguments: {exc}")],
                isError=True,
            )
        except Exception:
            logger.exception("query_knowledge_hub failed")
            return types.CallToolResult(
                content=[types.TextContent(text="Knowledge base query failed.")],
                isError=True,
            )

        text = response["content"][0]["text"]
        return types.CallToolResult(
            content=[types.TextContent(text=text)],
            structuredContent=response["structuredContent"],
        )

    server.add_request_handler("tools/list", types.PaginatedRequestParams, list_tools)
    server.add_request_handler("tools/call", types.CallToolRequestParams, call_tool)


async def run_server() -> None:
    """Run one MCP connection over stdin/stdout."""
    server = create_server()
    logger.info("Starting MCP server with stdio transport")

    # The MCP SDK serializes JSON-RPC messages to stdout.  Do not use print()
    # in this module: it would corrupt the protocol stream.
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Run the server as ``python -m mcp_server.server``."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
