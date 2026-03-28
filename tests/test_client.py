"""Tests for SubstrateClient."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from langchain_substrate.client import SubstrateClient, SubstrateError, ToolResult

BASE_URL = "https://test.substrate.local/mcp-server"


@pytest.fixture
def client() -> SubstrateClient:
    return SubstrateClient(api_key="sk-test-key", base_url=BASE_URL)


def _mock_success(content: list | None = None, tools: list | None = None) -> httpx.Response:
    """Build a mock JSON-RPC success response."""
    result: dict = {}
    if content is not None:
        result["content"] = content
        result["isError"] = False
    if tools is not None:
        result["tools"] = tools
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": result})


def _mock_error(code: int, message: str) -> httpx.Response:
    return httpx.Response(
        200, json={"jsonrpc": "2.0", "id": 1, "error": {"code": code, "message": message}}
    )


# -- ToolResult unit tests -------------------------------------------------


class TestToolResult:
    def test_text_concatenation(self) -> None:
        result = ToolResult(
            content=[
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ]
        )
        assert result.text == "hello\nworld"

    def test_text_empty(self) -> None:
        result = ToolResult(content=[])
        assert result.text == ""

    def test_data_json_parse(self) -> None:
        result = ToolResult(content=[{"type": "text", "text": '{"key": "val"}'}])
        assert result.data == {"key": "val"}

    def test_data_raw_fallback(self) -> None:
        result = ToolResult(content=[{"type": "text", "text": "not json"}])
        assert result.data == "not json"

    def test_is_error_default(self) -> None:
        result = ToolResult(content=[])
        assert result.is_error is False

    def test_ignores_non_text_blocks(self) -> None:
        result = ToolResult(
            content=[
                {"type": "image", "data": "..."},
                {"type": "text", "text": "only this"},
            ]
        )
        assert result.text == "only this"


# -- Sync client tests -----------------------------------------------------


class TestClientSync:
    @respx.mock
    def test_call_tool_success(self, client: SubstrateClient) -> None:
        respx.post(f"{BASE_URL}/mcp").mock(
            return_value=_mock_success(content=[{"type": "text", "text": "response text"}])
        )
        result = client.call_tool("memory_search", {"query": "test"})
        assert result.text == "response text"
        assert result.is_error is False

    @respx.mock
    def test_call_tool_jsonrpc_error(self, client: SubstrateClient) -> None:
        respx.post(f"{BASE_URL}/mcp").mock(
            return_value=_mock_error(-32000, "Invalid API key")
        )
        with pytest.raises(SubstrateError, match="Invalid API key"):
            client.call_tool("memory_search", {"query": "test"})

    @respx.mock
    def test_list_tools(self, client: SubstrateClient) -> None:
        tools = [{"name": "respond"}, {"name": "memory_search"}]
        respx.post(f"{BASE_URL}/mcp").mock(return_value=_mock_success(tools=tools))
        result = client.list_tools()
        assert len(result) == 2
        assert result[0]["name"] == "respond"

    @respx.mock
    def test_headers_include_bearer(self, client: SubstrateClient) -> None:
        route = respx.post(f"{BASE_URL}/mcp").mock(
            return_value=_mock_success(content=[])
        )
        client.call_tool("respond", {"message": "hi"})
        request = route.calls[0].request
        assert request.headers["authorization"] == "Bearer sk-test-key"

    @respx.mock
    def test_payload_structure(self, client: SubstrateClient) -> None:
        route = respx.post(f"{BASE_URL}/mcp").mock(
            return_value=_mock_success(content=[])
        )
        client.call_tool("memory_search", {"query": "test"})
        body = json.loads(route.calls[0].request.content)
        assert body["jsonrpc"] == "2.0"
        assert body["method"] == "tools/call"
        assert body["params"]["name"] == "memory_search"
        assert body["params"]["arguments"] == {"query": "test"}
        assert isinstance(body["id"], int)

    def test_request_id_increments(self, client: SubstrateClient) -> None:
        assert client._next_id() == 1
        assert client._next_id() == 2
        assert client._next_id() == 3

    @respx.mock
    def test_http_error_raises(self, client: SubstrateClient) -> None:
        respx.post(f"{BASE_URL}/mcp").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        with pytest.raises(httpx.HTTPStatusError):
            client.call_tool("respond", {"message": "hi"})


# -- Async client tests ----------------------------------------------------


class TestClientAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_acall_tool_success(self, client: SubstrateClient) -> None:
        respx.post(f"{BASE_URL}/mcp").mock(
            return_value=_mock_success(content=[{"type": "text", "text": "async response"}])
        )
        result = await client.acall_tool("hybrid_search", {"query": "test"})
        assert result.text == "async response"

    @respx.mock
    @pytest.mark.asyncio
    async def test_acall_tool_error(self, client: SubstrateClient) -> None:
        respx.post(f"{BASE_URL}/mcp").mock(
            return_value=_mock_error(-32029, "Rate limit")
        )
        with pytest.raises(SubstrateError, match="Rate limit"):
            await client.acall_tool("respond", {"message": "hi"})

    @respx.mock
    @pytest.mark.asyncio
    async def test_alist_tools(self, client: SubstrateClient) -> None:
        respx.post(f"{BASE_URL}/mcp").mock(
            return_value=_mock_success(tools=[{"name": "respond"}])
        )
        result = await client.alist_tools()
        assert len(result) == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_http_error_raises(self, client: SubstrateClient) -> None:
        respx.post(f"{BASE_URL}/mcp").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )
        with pytest.raises(httpx.HTTPStatusError):
            await client.acall_tool("respond", {"message": "hi"})
