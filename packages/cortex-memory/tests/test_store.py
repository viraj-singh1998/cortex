import pytest

from cortex.memory.store import InMemoryEmbeddingStore, InMemoryStore
from cortex.memory.types import Memory, MemoryStore, MemoryType, EmbeddingStore


def make_memory(
    content: str = "test",
    memory_type: MemoryType = MemoryType.SEMANTIC,
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> Memory:
    return Memory(
        content=content,
        memory_type=memory_type,
        user_id=user_id,
        agent_id=agent_id,
        run_id=run_id,
    )


class TestInMemoryStore:
    def setup_method(self):
        self.store = InMemoryStore()

    def test_protocol_conformance(self):
        assert isinstance(self.store, MemoryStore)

    def test_put_and_get(self):
        m = make_memory("hello")
        self.store.put(m)
        assert self.store.get(m.id) == m

    def test_get_missing_returns_none(self):
        assert self.store.get("nonexistent") is None

    def test_delete_removes_memory(self):
        m = make_memory()
        self.store.put(m)
        self.store.delete(m.id)
        assert self.store.get(m.id) is None

    def test_delete_nonexistent_is_silent(self):
        # Should not raise
        self.store.delete("ghost-id")

    def test_all_returns_all_memories(self):
        a, b = make_memory("a"), make_memory("b")
        self.store.put(a)
        self.store.put(b)
        result = self.store.all()
        assert len(result) == 2
        ids = {m.id for m in result}
        assert a.id in ids and b.id in ids

    def test_all_returns_snapshot_not_live_view(self):
        m = make_memory()
        self.store.put(m)
        snapshot = self.store.all()
        self.store.delete(m.id)
        # Snapshot should still contain m — deletion after snapshotting doesn't affect it
        assert len(snapshot) == 1

    def test_put_overwrites_existing(self):
        m = make_memory("original")
        self.store.put(m)
        updated = m.model_copy(update={"content": "updated"})
        self.store.put(updated)
        assert self.store.get(m.id).content == "updated"
        assert len(self.store.all()) == 1

    def test_filter_by_user_id(self):
        alice = make_memory(user_id="alice")
        bob = make_memory(user_id="bob")
        unscoped = make_memory()
        for m in [alice, bob, unscoped]:
            self.store.put(m)

        result = self.store.filter(user_id="alice")
        assert len(result) == 1
        assert result[0].id == alice.id

    def test_filter_by_multiple_scopes(self):
        target = make_memory(user_id="alice", run_id="run-1")
        wrong_run = make_memory(user_id="alice", run_id="run-2")
        wrong_user = make_memory(user_id="bob", run_id="run-1")
        for m in [target, wrong_run, wrong_user]:
            self.store.put(m)

        result = self.store.filter(user_id="alice", run_id="run-1")
        assert len(result) == 1
        assert result[0].id == target.id

    def test_filter_all_none_returns_everything(self):
        for _ in range(3):
            self.store.put(make_memory())
        assert len(self.store.filter()) == 3

    def test_filter_no_match_returns_empty(self):
        self.store.put(make_memory(user_id="alice"))
        assert self.store.filter(user_id="nobody") == []


class TestInMemoryEmbeddingStore:
    def setup_method(self):
        self.store = InMemoryEmbeddingStore()

    def test_protocol_conformance(self):
        assert isinstance(self.store, EmbeddingStore)

    def test_put_and_get(self):
        self.store.put("mem-1", [0.1, 0.2, 0.3])
        assert self.store.get("mem-1") == [0.1, 0.2, 0.3]

    def test_get_missing_returns_none(self):
        assert self.store.get("ghost") is None

    def test_delete_removes_embedding(self):
        self.store.put("mem-1", [1.0, 2.0])
        self.store.delete("mem-1")
        assert self.store.get("mem-1") is None

    def test_delete_nonexistent_is_silent(self):
        self.store.delete("ghost")

    def test_all_returns_copy(self):
        self.store.put("mem-1", [1.0])
        snapshot = self.store.all()
        self.store.delete("mem-1")
        # Snapshot unaffected
        assert "mem-1" in snapshot

    def test_all_contains_all_entries(self):
        self.store.put("a", [1.0])
        self.store.put("b", [2.0])
        result = self.store.all()
        assert set(result.keys()) == {"a", "b"}
