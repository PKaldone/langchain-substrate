"""Tests for SubstrateRetriever."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_substrate.client import ToolResult
from langchain_substrate.retriever import SubstrateRetriever


@pytest.fixture
def retriever() -> SubstrateRetriever:
    r = SubstrateRetriever(api_key="sk-test", base_url="https://test.local/mcp-server")
    return r


class TestRetrieverParsing:
    def test_parse_dict_list(self) -> None:
        data = [
            {"text": "first result", "score": 0.9},
            {"content": "second result", "id": "2"},
        ]
        docs = SubstrateRetriever._parse_to_documents(data, "query")
        assert len(docs) == 2
        assert docs[0].page_content == "first result"
        assert docs[0].metadata["source"] == "substrate"
        assert docs[0].metadata["score"] == 0.9
        assert docs[1].page_content == "second result"

    def test_parse_string_list(self) -> None:
        data = ["result one", "result two", ""]
        docs = SubstrateRetriever._parse_to_documents(data, "q")
        assert len(docs) == 2  # empty string filtered

    def test_parse_raw_string(self) -> None:
        docs = SubstrateRetriever._parse_to_documents("raw text", "q")
        assert len(docs) == 1
        assert docs[0].page_content == "raw text"

    def test_parse_empty_string(self) -> None:
        docs = SubstrateRetriever._parse_to_documents("", "q")
        assert len(docs) == 0

    def test_parse_dict_fallback_to_json(self) -> None:
        data = [{"key": "value", "num": 42}]
        docs = SubstrateRetriever._parse_to_documents(data, "q")
        assert len(docs) == 1
        parsed = json.loads(docs[0].page_content)
        assert parsed["key"] == "value"

    def test_parse_context_summary(self) -> None:
        data = [{"context_summary": "A summary of events"}]
        docs = SubstrateRetriever._parse_to_documents(data, "q")
        assert docs[0].page_content == "A summary of events"


class TestRetrieverSync:
    def test_get_relevant_documents(self, retriever: SubstrateRetriever) -> None:
        mock_result = ToolResult(
            content=[{"type": "text", "text": '[{"text": "found"}]'}]
        )
        with patch.object(retriever._client, "call_tool", return_value=mock_result) as mock:
            docs = retriever.invoke("search query")
            mock.assert_called_once()
            call_args = mock.call_args
            assert call_args[0][0] == "hybrid_search"
            assert call_args[0][1]["query"] == "search query"
            assert call_args[0][1]["top_k"] == 5

    def test_get_relevant_documents_with_namespace(self) -> None:
        retriever = SubstrateRetriever(
            api_key="sk-test",
            base_url="https://test.local/mcp-server",
            namespace="app.context",
        )
        mock_result = ToolResult(content=[{"type": "text", "text": "[]"}])
        with patch.object(retriever._client, "call_tool", return_value=mock_result) as mock:
            retriever.invoke("query")
            query = mock.call_args[0][1]["query"]
            assert query.startswith("app.context")

    def test_returns_empty_on_error(self, retriever: SubstrateRetriever) -> None:
        with patch.object(
            retriever._client, "call_tool", side_effect=RuntimeError("fail")
        ):
            docs = retriever.invoke("query")
            assert docs == []

    def test_memory_search_fallback(self) -> None:
        retriever = SubstrateRetriever(
            api_key="sk-test",
            base_url="https://test.local/mcp-server",
            search_tool="memory_search",
        )
        mock_result = ToolResult(
            content=[{"type": "text", "text": "Episode: test\n  Probability: 0.850"}]
        )
        with patch.object(retriever._client, "call_tool", return_value=mock_result) as mock:
            docs = retriever.invoke("test")
            assert mock.call_args[0][0] == "memory_search"
            # memory_search does not send top_k
            assert "top_k" not in mock.call_args[0][1]


class TestRetrieverAsync:
    @pytest.mark.asyncio
    async def test_aget_relevant_documents(self, retriever: SubstrateRetriever) -> None:
        mock_result = ToolResult(
            content=[{"type": "text", "text": '[{"text": "async hit"}]'}]
        )
        retriever._client.acall_tool = AsyncMock(return_value=mock_result)
        docs = await retriever.ainvoke("async query")
        assert len(docs) == 1
        assert docs[0].page_content == "async hit"

    @pytest.mark.asyncio
    async def test_async_returns_empty_on_error(self, retriever: SubstrateRetriever) -> None:
        retriever._client.acall_tool = AsyncMock(side_effect=RuntimeError("fail"))
        docs = await retriever.ainvoke("query")
        assert docs == []
