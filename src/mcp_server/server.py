"""MCP server entry point using the standard input/output transport.

The transport has one important rule: stdout is reserved for JSON-RPC messages
that an MCP client reads.  Application logs must therefore use stderr.
"""

from __future__ import annotations

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server

from observability.logger import get_logger

SERVER_NAME = "smart-knowledge-hub"
SERVER_VERSION = "0.1.0"

logger = get_logger(__name__)


def create_server() -> Server:
    """Create the MCP server.

    Tools are added in later E-stage tasks.  Creating the SDK server here is
    already enough to support the mandatory ``initialize`` handshake.
    """
    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        title="Smart Knowledge Hub",
        description="MCP interface for the Smart Knowledge Hub.",
    )


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
