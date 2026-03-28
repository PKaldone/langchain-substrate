"""SubstrateStore -- LangGraph BaseStore backed by SUBSTRATE causal memory.

Namespaces are encoded as dot-separated prefixes (e.g. ``("user", "123")``
becomes ``"user.123"``).  This lets SUBSTRATE's flat memory model emulate
LangGraph's hierarchical namespace scheme without server-side changes.

Operations mapping:
  - ``put``   -> ``respond`` (stores message with namespace:key metadata)
  - ``get``   -> ``memory_search`` (queries for namespace:key)
  - ``search`` -> ``hybrid_search`` (semantic + keyword retrieval)
  - ``list``  -> ``memory_search`` (lists recent memories in namespace)
  - ``delete`` -> no-op (SUBSTRATE memory is append-only / immutable)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterable, Iterator, Optional, Sequence

from langgraph.store.base import BaseStore, Item, Op, PutOp, GetOp, SearchOp, ListNamespacesOp

from langchain_substrate.client import SubstrateClient

logger = logging.getLogger(__name__)

_NAMESPACE_SEP = "."


def _encode_namespace(namespace: tuple[str, ...]) -> str:
    """Encode a namespace tuple into a dot-separated prefix string."""
    return _NAMESPACE_SEP.join(namespace)


def _make_item(
    namespace: tuple[str, ...],
    key: str,
    value: dict[str, Any],
    created_at: str | None = None,
    updated_at: str | None = None,
) -> Item:
    """Create a LangGraph Item from raw parts."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return Item(
        value=value,
        key=key,
        namespace=namespace,
        created_at=created_at or now,
        updated_at=updated_at or now,
    )


class SubstrateStore(BaseStore):
    """A LangGraph ``BaseStore`` backed by SUBSTRATE's causal memory.

    This store enables any LangGraph agent to persist and retrieve state
    through SUBSTRATE's cognitive memory layer, providing semantic search,
    causal reasoning, and identity-aware storage.

    Parameters
    ----------
    client:
        A configured ``SubstrateClient`` instance.
    namespace_prefix:
        Optional global prefix prepended to all namespace encodings.
        Useful for multi-tenant isolation (e.g. ``"app.mybot"``).

    Examples
    --------
    >>> from langchain_substrate import SubstrateStore, SubstrateClient
    >>> client = SubstrateClient(api_key="sk-sub-...")
    >>> store = SubstrateStore(client=client)
    >>> store.put(("user", "alice"), "prefs", {"theme": "dark"})
    >>> item = store.get(("user", "alice"), "prefs")
    """

    def __init__(
        self,
        client: SubstrateClient,
        *,
        namespace_prefix: str = "",
    ) -> None:
        self._client = client
        self._prefix = namespace_prefix

    def _full_namespace(self, namespace: tuple[str, ...]) -> str:
        encoded = _encode_namespace(namespace)
        if self._prefix:
            return f"{self._prefix}{_NAMESPACE_SEP}{encoded}"
        return encoded

    def _format_store_message(
        self, namespace: tuple[str, ...], key: str, value: dict[str, Any]
    ) -> str:
        """Build a structured message for SUBSTRATE's ``respond`` tool."""
        ns = self._full_namespace(namespace)
        return (
            f"[STORE namespace={ns} key={key}] "
            f"{json.dumps(value, separators=(',', ':'), default=str)}"
        )

    # -- batch interface (required by BaseStore) ---------------------------

    def batch(self, ops: Iterable[Op]) -> list[Any]:
        """Execute a batch of operations sequentially."""
        results: list[Any] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self.get(op.namespace, op.key))
            elif isinstance(op, PutOp):
                self.put(op.namespace, op.key, op.value)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(
                    self.search(
                        op.namespace_prefix,
                        query=op.query or "",
                        limit=op.limit,
                    )
                )
            elif isinstance(op, ListNamespacesOp):
                results.append(self.list_namespaces(prefix=op.match_conditions))
                results.append([])
            else:
                results.append(None)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Any]:
        """Execute a batch of operations asynchronously."""
        results: list[Any] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(await self.aget(op.namespace, op.key))
            elif isinstance(op, PutOp):
                await self.aput(op.namespace, op.key, op.value)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(
                    await self.asearch(
                        op.namespace_prefix,
                        query=op.query or "",
                        limit=op.limit,
                    )
                )
            elif isinstance(op, ListNamespacesOp):
                results.append([])
            else:
                results.append(None)
        return results

    # -- sync operations ---------------------------------------------------

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        """Store a value in SUBSTRATE memory via the ``respond`` tool.

        The value is serialized as a structured message with namespace and
        key metadata so it can be retrieved later via ``get`` or ``search``.
        """
        message = self._format_store_message(namespace, key, value)
        try:
            self._client.call_tool("respond", {"message": message})
        except Exception:
            logger.exception("SubstrateStore.put failed for namespace=%s key=%s", namespace, key)
            raise

    def get(self, namespace: tuple[str, ...], key: str) -> Optional[Item]:
        """Retrieve a value by namespace and key via ``memory_search``.

        Returns the best matching ``Item`` or ``None`` if no match is found.
        """
        ns = self._full_namespace(namespace)
        query = f"[STORE namespace={ns} key={key}]"
        try:
            result = self._client.call_tool("memory_search", {"query": query})
        except Exception:
            logger.exception("SubstrateStore.get failed for namespace=%s key=%s", namespace, key)
            return None

        return self._parse_memory_result(namespace, key, result.text)

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete is a no-op; SUBSTRATE memory is append-only."""
        logger.debug(
            "SubstrateStore.delete called (no-op) for namespace=%s key=%s", namespace, key
        )

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: str = "",
        limit: int = 5,
        filter: Optional[dict[str, Any]] = None,
        offset: int = 0,
    ) -> list[Item]:
        """Semantic search across SUBSTRATE memory via ``hybrid_search``.

        Parameters
        ----------
        namespace_prefix:
            Namespace prefix to scope the search.
        query:
            Natural language search query.
        limit:
            Maximum number of results.
        filter:
            Not used (SUBSTRATE does not support structured filters).
        offset:
            Not used (reserved for future pagination).
        """
        ns = self._full_namespace(namespace_prefix)
        search_query = f"{ns} {query}".strip() if query else ns
        try:
            result = self._client.call_tool(
                "hybrid_search", {"query": search_query, "top_k": limit}
            )
        except Exception:
            logger.exception("SubstrateStore.search failed for query=%s", query)
            return []

        return self._parse_search_results(namespace_prefix, result.data)

    def list_namespaces(
        self,
        *,
        prefix: Optional[tuple[str, ...]] = None,
        suffix: Optional[tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List namespaces. Limited implementation via memory_search."""
        logger.debug("SubstrateStore.list_namespaces called (limited support)")
        return []

    # -- async operations --------------------------------------------------

    async def aput(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        """Async variant of ``put``."""
        message = self._format_store_message(namespace, key, value)
        try:
            await self._client.acall_tool("respond", {"message": message})
        except Exception:
            logger.exception("SubstrateStore.aput failed for namespace=%s key=%s", namespace, key)
            raise

    async def aget(self, namespace: tuple[str, ...], key: str) -> Optional[Item]:
        """Async variant of ``get``."""
        ns = self._full_namespace(namespace)
        query = f"[STORE namespace={ns} key={key}]"
        try:
            result = await self._client.acall_tool("memory_search", {"query": query})
        except Exception:
            logger.exception("SubstrateStore.aget failed for namespace=%s key=%s", namespace, key)
            return None

        return self._parse_memory_result(namespace, key, result.text)

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Async delete is also a no-op."""
        logger.debug(
            "SubstrateStore.adelete called (no-op) for namespace=%s key=%s", namespace, key
        )

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: str = "",
        limit: int = 5,
        filter: Optional[dict[str, Any]] = None,
        offset: int = 0,
    ) -> list[Item]:
        """Async variant of ``search``."""
        ns = self._full_namespace(namespace_prefix)
        search_query = f"{ns} {query}".strip() if query else ns
        try:
            result = await self._client.acall_tool(
                "hybrid_search", {"query": search_query, "top_k": limit}
            )
        except Exception:
            logger.exception("SubstrateStore.asearch failed for query=%s", query)
            return []

        return self._parse_search_results(namespace_prefix, result.data)

    async def alist_namespaces(
        self,
        *,
        prefix: Optional[tuple[str, ...]] = None,
        suffix: Optional[tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """Async list namespaces. Limited implementation."""
        return []

    # -- parsing helpers ---------------------------------------------------

    @staticmethod
    def _parse_memory_result(
        namespace: tuple[str, ...], key: str, text: str
    ) -> Optional[Item]:
        """Parse a memory_search response into an Item.

        Attempts to extract JSON from a ``[STORE ...]`` tagged line.
        Falls back to wrapping raw text in ``{"text": ...}``.
        """
        if not text or text.strip() == "No matching episodes found.":
            return None

        # Try to find a STORE-tagged JSON payload
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Look for the structured marker
            marker_end = line.find("]")
            if line.startswith("[STORE") and marker_end > 0:
                payload = line[marker_end + 1 :].strip()
                try:
                    value = json.loads(payload)
                    return _make_item(namespace, key, value)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Fall back: wrap raw text as the value
        return _make_item(namespace, key, {"text": text})

    @staticmethod
    def _parse_search_results(
        namespace: tuple[str, ...], data: Any
    ) -> list[Item]:
        """Parse hybrid_search results into a list of Items."""
        items: list[Item] = []
        if isinstance(data, list):
            for i, entry in enumerate(data):
                if isinstance(entry, dict):
                    key = entry.get("id", str(i))
                    value = entry
                elif isinstance(entry, str):
                    key = str(i)
                    value = {"text": entry}
                else:
                    continue
                items.append(_make_item(namespace, key, value))
        elif isinstance(data, str):
            if data and data != "No matching episodes found.":
                items.append(_make_item(namespace, "0", {"text": data}))
        return items
