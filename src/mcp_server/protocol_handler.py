"""Small JSON-RPC dispatcher for the MCP methods used by this project.

The official MCP SDK owns the stdio wire protocol in :mod:`mcp_server.server`.
This class keeps the project-specific concerns--tool registration, routing and
consistent error responses--in a small unit that can be tested without opening
a subprocess.  Later E-stage tasks register the real knowledge-base tools here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2025-06-18"

INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class InvalidParamsError(ValueError):
    """Raised when a request has valid JSON but invalid method parameters."""


@dataclass(frozen=True)
class RegisteredTool:
    """The public schema and Python implementation of one MCP tool."""

    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]


class ProtocolHandler:
    """Parse the small JSON-RPC surface used by the MCP server.

    ``handle_request`` accepts a decoded JSON object and returns a decoded
    JSON-RPC response.  A request without an ``id`` is a notification, so it
    is processed but deliberately produces no response.
    """

    def __init__(
        self,
        *,
        server_name: str = "knowledge-hub-v1",
        server_version: str = "0.1.0",
    ) -> None:
        self.server_name = server_name
        self.server_version = server_version
        self._tools: dict[str, RegisteredTool] = {}

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register a callable and the schema a client sees in ``tools/list``."""
        if not isinstance(name, str) or not name:
            raise ValueError("Tool name must be a non-empty string.")
        if not isinstance(description, str):
            raise ValueError("Tool description must be a string.")
        if not isinstance(input_schema, dict):
            raise ValueError("Tool input_schema must be a dictionary.")
        if not callable(handler):
            raise ValueError("Tool handler must be callable.")
        self._tools[name] = RegisteredTool(description, input_schema, handler)

    def handle_initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Negotiate capabilities and return this server's identity."""
        if not isinstance(params, Mapping):
            raise InvalidParamsError("initialize params must be an object.")
        protocol_version = params.get("protocolVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise InvalidParamsError("initialize requires protocolVersion.")

        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version,
            },
        }

    def handle_tools_list(self) -> dict[str, list[dict[str, Any]]]:
        """Return schemas for all registered tools, without Python callables."""
        tools = [
            {
                "name": name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for name, tool in self._tools.items()
        ]
        return {"tools": tools}

    def handle_tools_call(self, name: Any, arguments: Any) -> Any:
        """Validate a call and pass its keyword arguments to the named tool."""
        if not isinstance(name, str) or not name:
            raise InvalidParamsError("tools/call requires a non-empty name.")
        if not isinstance(arguments, Mapping):
            raise InvalidParamsError("tools/call arguments must be an object.")

        tool = self._tools.get(name)
        if tool is None:
            raise InvalidParamsError(f"Unknown tool: {name}")
        self._validate_tool_arguments(tool.input_schema, arguments)
        return tool.handler(**dict(arguments))

    def handle_request(self, request: Any) -> dict[str, Any] | None:
        """Dispatch one decoded JSON-RPC request into a result or error object."""
        if not isinstance(request, Mapping):
            return self._error_response(None, INVALID_REQUEST, "Invalid Request")

        request_id = request.get("id")
        is_notification = "id" not in request
        if request.get("jsonrpc") != JSONRPC_VERSION or not isinstance(request.get("method"), str):
            return self._error_response(request_id, INVALID_REQUEST, "Invalid Request")

        method = request["method"]
        params = request.get("params", {})
        if not isinstance(params, Mapping):
            return self._error_response(request_id, INVALID_PARAMS, "Invalid params")

        try:
            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "tools/list":
                result = self.handle_tools_list()
            elif method == "tools/call":
                result = self.handle_tools_call(params.get("name"), params.get("arguments", {}))
            else:
                return self._error_response(request_id, METHOD_NOT_FOUND, "Method not found")
        except InvalidParamsError:
            return self._error_response(request_id, INVALID_PARAMS, "Invalid params")
        except Exception:
            # A client gets a stable message; implementation details belong in
            # stderr logs, never in a protocol response.
            return self._error_response(request_id, INTERNAL_ERROR, "Internal error")

        if is_notification:
            return None
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}

    @staticmethod
    def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    @staticmethod
    def _validate_tool_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> None:
        """Validate the common top-level JSON Schema constraints for a tool call.

        Full JSON Schema validation is unnecessary at this boundary.  The
        compact checks below catch the client mistakes that matter before a
        Python callable runs: missing required fields and obvious type errors.
        A tool can still perform richer domain-specific validation itself.
        """
        required_fields = schema.get("required", [])
        if not isinstance(required_fields, list):
            raise InvalidParamsError("Tool schema has an invalid required field list.")
        for field in required_fields:
            if not isinstance(field, str) or field not in arguments:
                raise InvalidParamsError("Missing required tool argument.")

        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise InvalidParamsError("Tool schema has invalid properties.")
        for field, value in arguments.items():
            field_schema = properties.get(field)
            if not isinstance(field_schema, Mapping):
                continue
            expected_type = field_schema.get("type")
            if expected_type is not None and not ProtocolHandler._matches_json_type(value, expected_type):
                raise InvalidParamsError("Tool argument has an invalid type.")

    @staticmethod
    def _matches_json_type(value: Any, expected_type: Any) -> bool:
        """Return whether a Python value corresponds to a basic JSON Schema type."""
        if isinstance(expected_type, list):
            return any(ProtocolHandler._matches_json_type(value, item) for item in expected_type)
        type_checks: dict[str, type[Any] | tuple[type[Any], ...]] = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "object": Mapping,
            "array": list,
            "null": type(None),
        }
        python_type = type_checks.get(expected_type)
        if python_type is None:
            return True
        # bool is an int subclass in Python, but not an integer in JSON Schema.
        if expected_type in {"number", "integer"} and isinstance(value, bool):
            return False
        return isinstance(value, python_type)
