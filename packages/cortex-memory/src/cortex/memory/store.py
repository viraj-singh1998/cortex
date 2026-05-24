from __future__ import annotations

from typing import Optional

from .types import Memory


class InMemoryStore:
    """MemoryStore backed by a plain Python dict.

    All writes come in through the MemoryIsolator's single background thread,
    so there is never more than one concurrent writer. CPython's GIL makes
    individual dict operations atomic, which is sufficient here.

    To swap to a persistent backend (SQLite, Redis, Postgres), implement the
    same four methods + filter() on a new class — the engine needs nothing else.
    """

    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}

    def get(self, memory_id: str) -> Optional[Memory]:
        return self._memories.get(memory_id)

    def put(self, memory: Memory) -> None:
        self._memories[memory.id] = memory

    def delete(self, memory_id: str) -> None:
        # pop with a default so deleting a non-existent id is a silent no-op,
        # not a KeyError. The engine treats missing ids as already-deleted.
        self._memories.pop(memory_id, None)

    def all(self) -> list[Memory]:
        # Snapshot the values so callers iterating the result aren't affected
        # by concurrent puts/deletes mid-loop.
        return list(self._memories.values())

    def filter(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[Memory]:
        """Return memories matching ALL provided scope constraints.

        How the matching works:
        - If you pass user_id="alice", only Alice's memories are returned.
        - If you pass user_id="alice" AND run_id="session-42", only Alice's
          memories from that specific session are returned.
        - If you pass nothing (all None), this is equivalent to calling all().

        A SQLiteStore would translate these into WHERE clauses for efficiency.
        Here we just iterate the dict — fine for in-memory workloads.
        """
        results = []
        for memory in self._memories.values():
            if user_id is not None and memory.user_id != user_id:
                continue
            if agent_id is not None and memory.agent_id != agent_id:
                continue
            if run_id is not None and memory.run_id != run_id:
                continue
            results.append(memory)
        return results


class InMemoryEmbeddingStore:
    """EmbeddingStore backed by a plain Python dict.

    Stores raw float vectors keyed by memory_id. Kept separate from InMemoryStore
    because embedding backends can differ from memory backends — you might want
    in-memory embeddings (fast cosine scan) with SQLite memories (persistence),
    or a dedicated vector DB for embeddings with Postgres for memories.

    The retriever owns this store and is the only thing that reads/writes it.
    """

    def __init__(self) -> None:
        self._embeddings: dict[str, list[float]] = {}

    def put(self, memory_id: str, embedding: list[float]) -> None:
        self._embeddings[memory_id] = embedding

    def get(self, memory_id: str) -> Optional[list[float]]:
        return self._embeddings.get(memory_id)

    def all(self) -> dict[str, list[float]]:
        # Returns a shallow copy so callers get a stable snapshot.
        # The values (lists of floats) are not copied — they're treated as
        # immutable once stored (embeddings never change for a given memory).
        return dict(self._embeddings)

    def delete(self, memory_id: str) -> None:
        self._embeddings.pop(memory_id, None)
