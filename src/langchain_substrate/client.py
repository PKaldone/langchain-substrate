"""HTTP client for the SUBSTRATE MCP server.

Encapsulates JSON-RPC communication with the SUBSTRATE Streamable HTTP
transport. All tool calls are dispatched via ``tools/call`` and responses
are parsed from the standard MCP content envelope.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://substrate.garmolabs.com/mcp-server"
_DEFAULT_TIMEOUT = 30.0
_JSONRPC_VERSION = "2.0"


class SubstrateError(Exception):
    """Raised when the SUBSTRATE MCP server returns an error."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"SUBSTRATE error {code}: {message}")


@dataclass(frozen=True)
class ToolResult:
    """Parsed result from a SUBSTRATE MCP ``tools/call`` response."""

    content: list[dict[str, Any]]
    is_error: bool = False

    @property
    def text(self) -> str:
        """Concatenate all text content blocks into a single string."""
        parts: list[str] = []
        for block in self.content:
            if block.get("type") == "text":
                parts.append(block["text"])
        return "\n".join(parts)

    @property
    def data(self) -> Any:
        """Attempt to parse the first text block as JSON; fall back to raw text."""
        raw = self.text
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw


@dataclass
class SubstrateClient:
    """Synchronous and asynchronous HTTP client for the SUBSTRATE MCP server.

    Parameters
    ----------
    api_key:
        SUBSTRATE API key (Bearer token).
    base_url:
        MCP server base URL. Defaults to the production endpoint.
    timeout:
        HTTP request timeout in seconds.
    """

    api_key: str
    base_url: str = _DEFAULT_BASE_URL
    timeout: float = _DEFAULT_TIMEOUT
    _request_id: int = field(default=0, init=False, repr=False)
    _sync_client: httpx.Client | None = field(default=None, init=False, repr=False)
    _async_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    # -- internal helpers --------------------------------------------------

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _build_payload(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "jsonrpc": _JSONRPC_VERSION,
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _parse_response(body: dict[str, Any]) -> dict[str, Any]:
        """Extract the result from a JSON-RPC response, raising on error."""
        if "error" in body:
            err = body["error"]
            raise SubstrateError(err.get("code", -1), err.get("message", "Unknown error"))
        return body.get("result", {})

    # -- sync transport ----------------------------------------------------

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(timeout=self.timeout)
        return self._sync_client

    def _post_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_sync_client()
        url = f"{self.base_url}/mcp"
        response = client.post(url, json=payload, headers=self._headers())
        response.raise_for_status()
        return self._parse_response(response.json())

    # -- async transport ---------------------------------------------------

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    async def _post_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_async_client()
        url = f"{self.base_url}/mcp"
        response = await client.post(url, json=payload, headers=self._headers())
        response.raise_for_status()
        return self._parse_response(response.json())

    # -- public sync API ---------------------------------------------------

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Call a SUBSTRATE MCP tool synchronously.

        Parameters
        ----------
        tool_name:
            Name of the tool (e.g. ``memory_search``, ``hybrid_search``).
        arguments:
            Tool arguments dict.

        Returns
        -------
        ToolResult
            Parsed response with ``.text``, ``.data``, and ``.content`` accessors.
        """
        payload = self._build_payload(
            "tools/call",
            {"name": tool_name, "arguments": arguments or {}},
        )
        result = self._post_sync(payload)
        return ToolResult(
            content=result.get("content", []),
            is_error=result.get("isError", False),
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """List available tools on the SUBSTRATE MCP server."""
        payload = self._build_payload("tools/list")
        result = self._post_sync(payload)
        return result.get("tools", [])

    # -- public async API --------------------------------------------------

    async def acall_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        """Call a SUBSTRATE MCP tool asynchronously."""
        payload = self._build_payload(
            "tools/call",
            {"name": tool_name, "arguments": arguments or {}},
        )
        result = await self._post_async(payload)
        return ToolResult(
            content=result.get("content", []),
            is_error=result.get("isError", False),
        )

    async def alist_tools(self) -> list[dict[str, Any]]:
        """List available tools asynchronously."""
        payload = self._build_payload("tools/list")
        result = await self._post_async(payload)
        return result.get("tools", [])

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying sync HTTP client."""
        if self._sync_client is not None and not self._sync_client.is_closed:
            self._sync_client.close()

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.close()
