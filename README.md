# langchain-substrate

SUBSTRATE cognitive memory integration for LangChain and LangGraph.

Use SUBSTRATE as a persistent memory store for LangGraph agents or as a retriever in LangChain RAG pipelines. SUBSTRATE provides causal memory, semantic search, knowledge graphs, emotion state, identity verification, and 61 cognitive capability layers.

## Installation

```bash
pip install langchain-substrate
```

Or install from source:

```bash
cd integrations/langchain
pip install -e ".[dev]"
```

## Quick Start

### Environment Setup

```python
import os
os.environ["SUBSTRATE_API_KEY"] = "sk-sub-..."
```

### As a LangGraph Memory Store

Use `SubstrateStore` as the backing store for any LangGraph agent. This gives your agent persistent, semantically searchable memory across conversations.

```python
from langchain_substrate import SubstrateStore, SubstrateClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Create the SUBSTRATE-backed store
client = SubstrateClient(api_key=os.environ["SUBSTRATE_API_KEY"])
store = SubstrateStore(client=client)

# Create a LangGraph agent with SUBSTRATE memory
model = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(model, tools=[], store=store)

# The agent now persists state to SUBSTRATE
config = {"configurable": {"thread_id": "conversation-1"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Remember that my favorite color is blue."}]},
    config=config,
)
```

#### Store Operations

```python
# Store a value
store.put(("user", "alice"), "preferences", {"theme": "dark", "language": "en"})

# Retrieve by key
item = store.get(("user", "alice"), "preferences")
print(item.value)  # {"theme": "dark", "language": "en"}

# Semantic search across memory
results = store.search(("user", "alice"), query="color preferences", limit=5)
for item in results:
    print(item.key, item.value)

# Delete is a no-op (SUBSTRATE memory is append-only)
store.delete(("user", "alice"), "preferences")
```

#### Multi-Tenant Isolation

```python
# Use namespace_prefix for tenant isolation
store = SubstrateStore(
    client=client,
    namespace_prefix="myapp.prod",
)
# All operations are scoped under "myapp.prod.*"
store.put(("user", "bob"), "state", {"step": 3})
```

### As a LangChain Retriever (RAG)

Use `SubstrateRetriever` in any LangChain RAG chain. It uses SUBSTRATE's hybrid search (semantic + keyword) to find relevant memories.

```python
from langchain_substrate import SubstrateRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Create the retriever
retriever = SubstrateRetriever(
    api_key=os.environ["SUBSTRATE_API_KEY"],
    top_k=5,
)

# Build a RAG chain
prompt = ChatPromptTemplate.from_template(
    "Answer based on the following context:\n{context}\n\nQuestion: {question}"
)
model = ChatOpenAI(model="gpt-4o")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = chain.invoke("What are the entity's core values?")
```

#### Retriever with Namespace Scoping

```python
retriever = SubstrateRetriever(
    api_key=os.environ["SUBSTRATE_API_KEY"],
    namespace="app.conversations",
    top_k=10,
)
```

#### Free Tier Fallback

`hybrid_search` requires the Pro tier. On the free tier, fall back to `memory_search`:

```python
retriever = SubstrateRetriever(
    api_key=os.environ["SUBSTRATE_API_KEY"],
    search_tool="memory_search",
)
```

### Async Support

All operations support async for use in async LangGraph workflows:

```python
import asyncio
from langchain_substrate import SubstrateStore, SubstrateClient

async def main():
    client = SubstrateClient(api_key="sk-sub-...")
    store = SubstrateStore(client=client)

    await store.aput(("user", "alice"), "mood", {"current": "happy"})
    item = await store.aget(("user", "alice"), "mood")
    print(item.value)

    results = await store.asearch(("user",), query="emotional state")
    for r in results:
        print(r.value)

asyncio.run(main())
```

## Architecture

```
LangGraph Agent / RAG Chain
        |
   SubstrateStore / SubstrateRetriever
        |
   SubstrateClient (httpx)
        |
   SUBSTRATE MCP Server (JSON-RPC over HTTP)
        |
   Causal Memory + Knowledge Graph + 61 Layers
```

### Namespace Encoding

LangGraph uses tuple namespaces like `("user", "alice", "prefs")`. SUBSTRATE uses flat string keys. The store encodes namespaces as dot-separated prefixes:

| LangGraph Namespace | SUBSTRATE Prefix |
|---|---|
| `("user", "alice")` | `user.alice` |
| `("app", "v2", "state")` | `app.v2.state` |

### Tool Mapping

| Store Operation | SUBSTRATE Tool | Tier |
|---|---|---|
| `put()` | `respond` | Free |
| `get()` | `memory_search` | Free |
| `search()` | `hybrid_search` | Pro |
| `list_namespaces()` | N/A (limited) | -- |
| `delete()` | No-op | -- |

## SUBSTRATE MCP Tools Available

| Tool | Description | Tier |
|---|---|---|
| `respond` | Send a message, get a response | Free |
| `memory_search` | Search causal memory episodes | Free |
| `hybrid_search` | Semantic + keyword search | Pro |
| `get_emotion_state` | Affective state vector | Free |
| `verify_identity` | Cryptographic identity check | Free |
| `knowledge_graph_query` | Query knowledge graph | Pro |
| `get_values` | Core value architecture | Free |
| `theory_of_mind` | User model | Free |
| `get_trust_state` | Trust scores | Pro |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=langchain_substrate --cov-report=term-missing

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## License

MIT -- Garmo Labs
