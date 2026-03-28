"""Tests for SubstrateStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_substrate.client import SubstrateClient, ToolResult
from langchain_substrate.store import SubstrateStore, _encode_namespace, _make_item


# -- Namespace encoding tests ----------------------------------------------


class TestNamespaceEncoding:
    def test_single_segment(self) -> None:
        assert _encode_namespace(("users",)) == "users"

    def test_multi_segment(self) -> None:
        assert _encode_namespace(("user", "alice", "prefs")) == "user.alice.prefs"

    def test_empty_tuple(self) -> None:
        assert _encode_namespace(()) == ""


class TestMakeItem:
    def test_creates_item_with_key_and_value(self) -> None:
        item = _make_item(("ns",), "k1", {"data": 42})
        assert item.key == "k1"
        assert item.value == {"data": 42}
        assert item.namespace == ("ns",)

    def test_timestamps_populated(self) -> None:
        item = _make_item(("ns",), "k", {})
        assert item.created_at is not None
        assert item.updated_at is not None


# -- Store tests with mocked client ----------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock(spec=SubstrateClient)


@pytest.fixture
def store(mock_client: MagicMock) -> SubstrateStore:
    return SubstrateStore(client=mock_client)


@pytest.fixture
def prefixed_store(mock_client: MagicMock) -> SubstrateStore:
    return SubstrateStore(client=mock_client, namespace_prefix="app.test")


class TestStorePut:
    def test_put_calls_respond(self, store: SubstrateStore, mock_client: MagicMock) -> None:
        store.put(("user", "alice"), "prefs", {"theme": "dark"})
        mock_client.call_tool.assert_called_once()
        call_args = mock_client.call_tool.call_args
        assert call_args[0][0] == "respond"
        assert "user.alice" in call_args[0][1]["message"]
        assert "prefs" in call_args[0][1]["message"]
        assert '"theme":"dark"' in call_args[0][1]["message"]

    def test_put_with_prefix(
        self, prefixed_store: SubstrateStore, mock_client: MagicMock
    ) -> None:
        prefixed_store.put(("user",), "key1", {"val": 1})
        message = mock_client.call_tool.call_args[0][1]["message"]
        assert "app.test.user" in message

    def test_put_propagates_error(
        self, store: SubstrateStore, mock_client: MagicMock
    ) -> None:
        mock_client.call_tool.side_effect = RuntimeError("network")
        with pytest.raises(RuntimeError, match="network"):
            store.put(("ns",), "k", {"v": 1})


class TestStoreGet:
    def test_get_returns_item_on_match(
        self, store: SubstrateStore, mock_client: MagicMock
    ) -> None:
        mock_client.call_tool.return_value = ToolResult(
            content=[{"type": "text", "text": '[STORE namespace=user.alice key=prefs] {"theme":"dark"}'}]
        )
        item = store.get(("user", "alice"), "prefs")
        assert item is not None
        assert item.value == {"theme": "dark"}
        assert item.key == "prefs"

    def test_get_returns_none_on_no_match(
        self, store: SubstrateStore, mock_client: MagicMock
    ) -> None:
        mock_client.call_tool.return_value = ToolResult(
            content=[{"type": "text", "text": "No matching episodes found."}]
        )
        assert store.get(("ns",), "k") is None

    def test_get_fallback_wraps_raw_text(
        self, store: SubstrateStore, mock_client: MagicMock
    ) -> None:
        mock_client.call_tool.return_value = ToolResult(
            content=[{"type": "text", "text": "some raw memory text"}]
        )
        item = store.get(("ns",), "k")
        assert item is not None
        assert item.value == {"text": "some raw memory text"}

    def test_get_returns_none_on_error(
        self, store: SubstrateStore, mock_client: MagicMock
    ) -> None:
        mock_client.call_tool.side_effect = RuntimeError("timeout")
        assert store.get(("ns",), "k") is None


class TestStoreSearch:
    def test_search_returns_items(
        self, store: SubstrateStore, mock_client: MagicMock
    ) -> None:
        mock_client.call_tool.return_value = ToolResult(
            content=[
                {
                    "type": "text",
                    "text": '[{"id":"1","text":"first"},{"id":"2","text":"second"}]',
                }
            ]
        )
        items = store.search(("user",), query="test", limit=2)
        assert len(items) == 2
        assert items[0].key == "1"
        assert items[0].value["text"] == "first"

    def test_search_calls_hybrid_search(
        self, store: SubstrateStore, mock_client: MagicMock
    ) -> None:
        mock_client.call_tool.return_value = ToolResult(content=[{"type": "text", "text": "[]"}])
        store.search(("app",), query="hello", limit=3)
        call_args = mock_client.call_tool.call_args
        assert call_args[0][0] == "hybrid_search"
        assert call_args[0][1]["top_k"] == 3
        assert "hello" in call_args[0][1]["query"]

    def test_search_empty_on_error(
        self, store: SubstrateStore, mock_client: MagicMock
    ) -> None:
        mock_client.call_tool.side_effect = RuntimeError("fail")
        assert store.search(("ns",), query="q") == []


class TestStoreDelete:
    def test_delete_is_noop(self, store: SubstrateStore, mock_client: MagicMock) -> None:
        store.delete(("ns",), "k")
        mock_client.call_tool.assert_not_called()


# -- Async tests -----------------------------------------------------------


class TestStoreAsync:
    @pytest.mark.asyncio
    async def test_aput(self, store: SubstrateStore, mock_client: MagicMock) -> None:
        mock_client.acall_tool = AsyncMock()
        await store.aput(("ns",), "k", {"v": 1})
        mock_client.acall_tool.assert_called_once()
        assert mock_client.acall_tool.call_args[0][0] == "respond"

    @pytest.mark.asyncio
    async def test_aget_returns_item(
        self, store: SubstrateStore, mock_client: MagicMock
    ) -> None:
        mock_client.acall_tool = AsyncMock(
            return_value=ToolResult(
                content=[{"type": "text", "text": '[STORE namespace=ns key=k] {"v":1}'}]
            )
        )
        item = await store.aget(("ns",), "k")
        assert item is not None
        assert item.value == {"v": 1}

    @pytest.mark.asyncio
    async def test_adelete_is_noop(
        self, store: SubstrateStore, mock_client: MagicMock
    ) -> None:
        mock_client.acall_tool = AsyncMock()
        await store.adelete(("ns",), "k")
        mock_client.acall_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_asearch(self, store: SubstrateStore, mock_client: MagicMock) -> None:
        mock_client.acall_tool = AsyncMock(
            return_value=ToolResult(
                content=[{"type": "text", "text": '[{"id":"1","text":"hit"}]'}]
            )
        )
        items = await store.asearch(("ns",), query="q", limit=1)
        assert len(items) == 1
