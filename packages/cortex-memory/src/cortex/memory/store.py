from __future__ import annotations

from typing import Optional

from .types import EmbeddingStore, Memory, MemoryStore


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}

    def get(self, memory_id: str) -> Optional[Memory]:
        return self._memories.get(memory_id)

    def put(self, memory: Memory) -> None:
        self._memories[memory.id] = memory

    def delete(self, memory_id: str) -> None:
        self._memories.pop(memory_id, None)

    def all(self) -> list[Memory]:
        return list(self._memories.values())

    def filter(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[Memory]:
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


class InMemoryEmbeddingStore(EmbeddingStore):
    def __init__(self) -> None:
        self._embeddings: dict[str, list[float]] = {}

    def put(self, memory_id: str, embedding: list[float]) -> None:
        self._embeddings[memory_id] = embedding

    def get(self, memory_id: str) -> Optional[list[float]]:
        return self._embeddings.get(memory_id)

    def all(self) -> dict[str, list[float]]:
        return dict(self._embeddings)

    def delete(self, memory_id: str) -> None:
        self._embeddings.pop(memory_id, None)
