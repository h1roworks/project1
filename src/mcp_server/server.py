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
from mcp_server.tools.list_collections import (
    LIST_COLLECTIONS_SCHEMA,
    ListCollectionsTool,
)
from observability.logger import get_logger

SERVER_NAME = "smart-knowledge-hub"
SERVER_VERSION = "0.1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = get_logger(__name__)


def create_server(
    query_tool: QueryKnowledgeHubTool | None = None,
    collections_tool: ListCollectionsTool | None = None,
) -> Server:
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
    collection_lister = collections_tool or ListCollectionsTool(
        PROJECT_ROOT / "data" / "documents"
    )
    _register_tools(server, tool, collection_lister)
    return server


def _register_tools(
    server: Server,
    query_tool: QueryKnowledgeHubTool,
    collections_tool: ListCollectionsTool,
) -> None:
    """Expose the E3/E4 functions through the official MCP SDK handlers."""

    async def list_tools(_context: Any, _params: types.PaginatedRequestParams) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="query_knowledge_hub",
                    description="使用混合检索与重排查询本地知识库，并返回可追溯引用。",
                    inputSchema=QUERY_KNOWLEDGE_HUB_SCHEMA,
                ),
                types.Tool(
                    name="list_collections",
                    description="列出 data/documents/ 下可用的知识库集合及文档数量。",
                    inputSchema=LIST_COLLECTIONS_SCHEMA,
                ),
            ]
        )

    async def call_tool(_context: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        tools = {
            "query_knowledge_hub": query_tool,
            "list_collections": collections_tool,
        }
        tool = tools.get(params.name)
        if tool is None:
            return types.CallToolResult(
                content=[types.TextContent(text=f"Unknown tool: {params.name}")],
                isError=True,
            )
        try:
            response = tool(**(params.arguments or {}))
        except (TypeError, ValueError) as exc:
            return types.CallToolResult(
                content=[types.TextContent(text=f"Invalid tool arguments: {exc}")],
                isError=True,
            )
        except Exception:
            logger.exception("MCP tool failed: %s", params.name)
            return types.CallToolResult(
                content=[types.TextContent(text="MCP tool execution failed.")],
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
