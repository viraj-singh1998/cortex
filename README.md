# Cortex

A monorepo of production-grade ML engineering packages. Each package maps to a distinct build target — independently installable, framework-agnostic, and designed to compose.

## Packages

| Package | Description | Status |
|---|---|---|
| [`cortex-memory`](#cortex-memory) | Time-aware, graph-structured preference memory engine for LLM agents | In progress |
| `cortex-rag` | Production RAG pipeline | Planned |
| `cortex-evals` | Agent evaluation framework | Planned |
| `cortex-codegen` | Safe code-generation agent | Planned |
| `cortex-routing` | Test-time compute routing | Planned |
| `cortex-ai` | Meta-package | Planned |

---

## cortex-memory

A memory engine for LLM agents that tracks, scores, and retrieves preferences over time.

```
pip install cortex-memory
```

### What it does

Most memory systems are flat vector stores — retrieve the nearest chunk, inject it, move on. `cortex-memory` treats memory as a graph that evolves with use:

- **Typed links with belief revision semantics.** When a new memory is added, it's linked to semantically related existing ones. As memories are reinforced or contradicted, an LLM reasons over those links and upgrades each relationship to `SUPPORTS`, `CONTRADICTS`, or `SUPERSEDES`. A superseded memory is retired from active retrieval but kept for audit. The graph stays coherent.

- **PAMU confidence scoring.** Confidence isn't a single number — it's the fusion of an EMA (long-term tendency) and a sliding window (recent shifts). A preference that was high-confidence last month but contradicted twice this week ranks accordingly.

- **Ebbinghaus decay, per type.** WORKING memories are ephemeral (stability = 0.1 days). SEMANTIC preferences survive weeks (7 days). PROCEDURAL workflows change rarely (30 days). All configurable via `DecayConfig`.

- **3-stage retrieval.** Cosine similarity + BM25 (on content, keywords, contextual description, and retrieval notes) + entity Jaccard, fused via RRF. Optional LLM reranker, off by default. Retrieval quality dominates write quality by 3–5×; this is where the investment went.

- **Bi-temporal timestamps.** Every memory carries `event_time` (when the fact was true) and `ingestion_time` (when the agent learned it). Temporal queries work correctly even when information arrives late.

- **Thread-isolated writes.** Memory capture runs on a background thread. The main agent thread never blocks. An `atexit` handler flushes the queue on process exit so nothing is silently dropped.

### Public API

```python
engine.add(request)          # enrich → conflict detect → resolve → store → link
engine.add_many(requests)    # batch: N memories → 1 LLM call
engine.inject(query)         # retrieve + format memories for system prompt
engine.search(query)         # raw retrieval with scores
engine.get(memory_id)
engine.delete(memory_id)
engine.reinforce(memory_id)  # external verification signal
engine.flush()               # drain background queue
```

### LangGraph integration

```
pip install cortex-memory[langgraph]
```

`cortex-memory` ships a `CortexLangGraphStore` that implements LangGraph's native `BaseStore` interface — any graph that accepts `store: BaseStore` works with it automatically.

For most use cases, drop in the pre-built nodes:

```python
from cortex.memory.integrations.langgraph import recall_node, retain_node
from langgraph.graph import StateGraph

graph = (
    StateGraph(AgentState)
    .add_node("recall", recall_node)   # injects memories into system prompt before LLM call
    .add_node("llm", llm_node)
    .add_node("retain", retain_node)   # extracts facts and stores them after LLM call
    .add_edge("recall", "llm")
    .add_edge("llm", "retain")
    .compile(store=CortexLangGraphStore(engine))
)
```

For tool-calling agents that need explicit memory control:

```python
from cortex.memory.integrations.langgraph import make_recall_tool, make_retain_tool

tools = [make_recall_tool(engine), make_retain_tool(engine)]
```

### Other integrations (planned)

| Framework | What ships |
|---|---|
| CrewAI | `CortexCrewAIMemory` — implements CrewAI's `Memory` protocol, drop into `memory_config` |
| AutoGen / AG2 | `CortexAutoGenMemory` — `get_all()` + `add()` matching AutoGen's `MemoryClient` interface |
| OpenAI Agents SDK | `make_search_tool()` + `make_add_tool()` — SDK-compatible function tools |
| MCP | `cortex-memory[mcp]` — FastMCP server; any MCP-capable agent gets memory for free |
| Everything else | `CortexMemoryMiddleware` — `before_call(query)` / `after_call(content)` universal adapter |

### Status

Foundation complete and tested (99 tests passing): `types`, `store`, `scoring`, `resolution`.

In progress: `embedders` → `retrieval` → `isolation` → `engine` → `integrations/langgraph`.

---

## Repo setup

This is a [uv](https://docs.astral.sh/uv/) workspace. All packages share a lockfile.

```bash
uv sync                                         # install all packages + dev deps
uv run pytest packages/cortex-memory/tests/ -v  # run cortex-memory tests
```
