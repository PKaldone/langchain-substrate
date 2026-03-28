"""SubstrateRetriever -- LangChain BaseRetriever backed by SUBSTRATE hybrid search.

Plugs directly into any LangChain RAG chain as a retriever component.
Uses SUBSTRATE's ``hybrid_search`` tool which combines semantic similarity
and keyword matching across the entity's causal memory and knowledge graph.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr

from langchain_substrate.client import SubstrateClient

logger = logging.getLogger(__name__)


class SubstrateRetriever(BaseRetriever):
    """A LangChain retriever that queries SUBSTRATE's hybrid search.

    Combines semantic and keyword search across the entity's causal
    memory, knowledge graph, and reflection layers.

    Parameters
    ----------
    api_key:
        SUBSTRATE API key.
    base_url:
        MCP server base URL (defaults to production).
    top_k:
        Number of results to return per query.
    namespace:
        Optional namespace to scope searches (dot-separated).
    search_tool:
        MCP tool to use for retrieval. Defaults to ``hybrid_search``
        (requires pro tier). Falls back to ``memory_search`` on free tier.

    Examples
    --------
    >>> from langchain_substrate import SubstrateRetriever
    >>> retriever = SubstrateRetriever(api_key="sk-sub-...")
    >>> docs = retriever.invoke("What are the core values?")
    """

    api_key: str
    base_url: str = "https://substrate.garmolabs.com/mcp-server"
    top_k: int = Field(default=5, ge=1, le=50)
    namespace: str = ""
    search_tool: str = "hybrid_search"

    _client: SubstrateClient = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize the underlying SubstrateClient after Pydantic validation."""
        self._client = SubstrateClient(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Retrieve documents matching the query via SUBSTRATE hybrid search.

        Parameters
        ----------
        query:
            Natural language search query.
        run_manager:
            LangChain callback manager (unused but required by interface).

        Returns
        -------
        list[Document]
            Matching documents with content and metadata.
        """
        search_query = f"{self.namespace} {query}".strip() if self.namespace else query

        arguments: dict[str, Any] = {"query": search_query}
        if self.search_tool == "hybrid_search":
            arguments["top_k"] = self.top_k

        try:
            result = self._client.call_tool(self.search_tool, arguments)
        except Exception:
            logger.exception("SubstrateRetriever search failed for query=%s", query)
            return []

        return self._parse_to_documents(result.data, query)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any,
    ) -> list[Document]:
        """Async variant of ``_get_relevant_documents``."""
        search_query = f"{self.namespace} {query}".strip() if self.namespace else query

        arguments: dict[str, Any] = {"query": search_query}
        if self.search_tool == "hybrid_search":
            arguments["top_k"] = self.top_k

        try:
            result = await self._client.acall_tool(self.search_tool, arguments)
        except Exception:
            logger.exception("SubstrateRetriever async search failed for query=%s", query)
            return []

        return self._parse_to_documents(result.data, query)

    @staticmethod
    def _parse_to_documents(data: Any, query: str) -> list[Document]:
        """Convert SUBSTRATE search results into LangChain Documents.

        Handles multiple response shapes:
        - ``list[dict]``: each dict becomes a Document
        - ``list[str]``: each string becomes a Document
        - ``str``: single Document from raw text
        """
        documents: list[Document] = []

        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    content = entry.get("text", entry.get("content", ""))
                    if not content:
                        # Use context_summary or full JSON as fallback
                        content = entry.get("context_summary", "")
                    if not content:
                        import json

                        content = json.dumps(entry, default=str)
                    metadata = {
                        k: v
                        for k, v in entry.items()
                        if k not in ("text", "content") and not k.startswith("_")
                    }
                    metadata["source"] = "substrate"
                    metadata["query"] = query
                    documents.append(Document(page_content=content, metadata=metadata))
                elif isinstance(entry, str) and entry.strip():
                    documents.append(
                        Document(
                            page_content=entry,
                            metadata={"source": "substrate", "query": query},
                        )
                    )
        elif isinstance(data, str) and data.strip():
            documents.append(
                Document(
                    page_content=data,
                    metadata={"source": "substrate", "query": query},
                )
            )

        return documents
