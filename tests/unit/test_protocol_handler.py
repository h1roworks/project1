"""Unit tests for MCP JSON-RPC parsing and tool routing."""

from __future__ import annotations

import pytest

from mcp_server.protocol_handler import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    ProtocolHandler,
)


@pytest.fixture
def handler() -> ProtocolHandler:
    protocol_handler = ProtocolHandler(server_version="test-version")
    protocol_handler.register_tool(
        name="echo",
        description="Return the supplied text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda text: {"content": [{"type": "text", "text": text}]},
    )
    return protocol_handler


def test_initialize_returns_identity_and_tools_capability(handler: ProtocolHandler) -> None:
    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "knowledge-hub-v1", "version": "test-version"},
        },
    }


def test_tools_list_returns_registered_schema(handler: ProtocolHandler) -> None:
    response = handler.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

    assert response is not None
    assert response["result"]["tools"] == [
        {
            "name": "echo",
            "description": "Return the supplied text.",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
    ]


def test_tools_call_routes_to_registered_tool(handler: ProtocolHandler) -> None:
    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hello MCP"}},
        }
    )

    assert response is not None
    assert response["result"] == {"content": [{"type": "text", "text": "hello MCP"}]}


@pytest.mark.parametrize(
    ("rpc_request", "expected_code"),
    [
        ({"jsonrpc": "1.0", "id": 1, "method": "initialize"}, INVALID_REQUEST),
        ({"jsonrpc": "2.0", "id": 2, "method": "not/a/method"}, METHOD_NOT_FOUND),
        (
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "echo"}},
            INVALID_PARAMS,
        ),
        (
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "missing", "arguments": {}},
            },
            INVALID_PARAMS,
        ),
    ],
)
def test_invalid_requests_return_standard_error_codes(
    handler: ProtocolHandler, rpc_request: dict[str, object], expected_code: int
) -> None:
    response = handler.handle_request(rpc_request)

    assert response is not None
    assert response["error"]["code"] == expected_code


def test_tool_exception_is_hidden_behind_internal_error(handler: ProtocolHandler) -> None:
    def broken_tool() -> None:
        raise RuntimeError("database password leaked")

    handler.register_tool(
        name="broken",
        description="Always fails.",
        input_schema={"type": "object"},
        handler=broken_tool,
    )

    response = handler.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "broken", "arguments": {}},
        }
    )

    assert response is not None
    assert response["error"] == {"code": INTERNAL_ERROR, "message": "Internal error"}
    assert "password" not in str(response)
