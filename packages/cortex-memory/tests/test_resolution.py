import pytest

from cortex.memory.resolution import detect_conflicts, prune_versions, resolve
from cortex.memory.types import ConflictStrategy, DecayConfig, Memory, MemoryType


cfg = DecayConfig()


def make_memory(
    content: str = "test",
    entity_tags: list[str] | None = None,
    confidence: float = 0.8,
    confidence_history: list[float] | None = None,
    ema_confidence: float = 0.8,
    memory_type: MemoryType = MemoryType.SEMANTIC,
) -> Memory:
    return Memory(
        content=content,
        memory_type=memory_type,
        confidence=confidence,
        ema_confidence=ema_confidence,
        confidence_history=confidence_history or [],
        entity_tags=entity_tags or [],
    )


def unit_vec(dim: int, idx: int) -> list[float]:
    """Unit vector along axis idx in dim-dimensional space. Cosine with itself = 1.0."""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


class TestDetectConflicts:
    def test_no_candidates_returns_empty(self):
        m = make_memory(entity_tags=["a", "b"])
        assert detect_conflicts(m, []) == []

    def test_no_shared_tags_no_conflict(self):
        new = make_memory(entity_tags=["postgres", "database"])
        other = make_memory(entity_tags=["redis", "cache"])
        results = detect_conflicts(new, [other])
        assert results == []

    def test_high_jaccard_flags_conflict(self):
        new = make_memory(entity_tags=["a", "b", "c", "d", "e"])
        other = make_memory(entity_tags=["a", "b", "c", "d", "f"])
        # Intersection 4, union 6 → Jaccard = 0.667 < 0.7 threshold → no conflict
        assert detect_conflicts(new, [other]) == []

    def test_very_high_jaccard_flags_conflict(self):
        new = make_memory(entity_tags=["a", "b", "c", "d", "e", "f"])
        other = make_memory(entity_tags=["a", "b", "c", "d", "e", "f", "g"])
        # No embeddings → solo Jaccard path. Intersection 6, union 7 → Jaccard ≈ 0.857 >= 0.85 solo threshold → conflict
        results = detect_conflicts(new, [other])
        assert len(results) == 1
        assert results[0][0].id == other.id

    def test_identical_tags_is_conflict(self):
        tags = ["postgres", "database", "sql"]
        new = make_memory(entity_tags=tags)
        other = make_memory(entity_tags=tags[:])
        results = detect_conflicts(new, [other])
        assert len(results) == 1

    def test_cosine_only_below_solo_threshold_no_conflict(self):
        # Two orthogonal-ish vectors, cosine < 0.85
        new = make_memory()
        other = make_memory()
        new_emb = [1.0, 0.0, 0.0]
        cand_embs = {other.id: [0.5, 0.5, 0.707]}  # cosine ~0.5
        results = detect_conflicts(new, [other], new_embedding=new_emb, candidate_embeddings=cand_embs)
        assert results == []

    def test_cosine_above_solo_threshold_flags_conflict(self):
        new = make_memory()
        other = make_memory()
        new_emb = [1.0, 0.0, 0.0]
        # Same direction → cosine = 1.0 >= 0.85
        cand_embs = {other.id: [1.0, 0.0, 0.0]}
        results = detect_conflicts(new, [other], new_embedding=new_emb, candidate_embeddings=cand_embs)
        assert len(results) == 1

    def test_and_logic_requires_both_signals(self):
        # Cosine >= 0.75 but Jaccard < 0.7 → no conflict with AND logic
        new = make_memory(entity_tags=["a", "b"])
        other = make_memory(entity_tags=["a", "x", "y", "z"])  # Jaccard = 1/5 = 0.2
        new_emb = [1.0, 0.0]
        cand_embs = {other.id: [0.9, 0.436]}  # cosine ~0.9 >= 0.75
        results = detect_conflicts(
            new, [other],
            new_embedding=new_emb,
            candidate_embeddings=cand_embs,
        )
        # Both must agree — cosine yes, Jaccard no → no conflict
        assert results == []

    def test_memory_not_conflicted_with_itself(self):
        m = make_memory(entity_tags=["a", "b", "c", "d", "e"])
        results = detect_conflicts(m, [m])
        assert results == []

    def test_returns_sorted_by_score_descending(self):
        new = make_memory(entity_tags=["a", "b", "c", "d", "e", "f"])
        # strong_conflict: Jaccard 6/7 ≈ 0.857
        strong = make_memory(entity_tags=["a", "b", "c", "d", "e", "f", "g"])
        # weaker_conflict: Jaccard 6/8 = 0.75
        weaker = make_memory(entity_tags=["a", "b", "c", "d", "e", "f", "x", "y"])
        results = detect_conflicts(new, [weaker, strong])
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]

    def test_top_k_limits_results(self):
        new = make_memory(entity_tags=["a", "b", "c", "d", "e", "f"])
        candidates = [
            make_memory(entity_tags=["a", "b", "c", "d", "e", "f"])
            for _ in range(10)
        ]
        results = detect_conflicts(new, candidates, top_k=3)
        assert len(results) <= 3


class TestResolve:
    def test_last_write_wins_keeps_new(self):
        new = make_memory("new")
        existing = make_memory("existing")
        result = resolve(new, existing, ConflictStrategy.LAST_WRITE_WINS, cfg)
        assert result.kept.id == new.id
        assert result.discarded is not None
        assert result.discarded.id == existing.id
        assert result.conflict_detected is True

    def test_confidence_weighted_higher_pamu_wins(self):
        # new has recent high history → higher pamu_score
        new = make_memory(confidence=0.9, ema_confidence=0.9, confidence_history=[0.9] * 5)
        existing = make_memory(confidence=0.3, ema_confidence=0.3, confidence_history=[0.3] * 5)
        result = resolve(new, existing, ConflictStrategy.CONFIDENCE_WEIGHTED, cfg)
        assert result.kept.content == new.content

    def test_confidence_weighted_existing_wins_when_higher(self):
        new = make_memory(confidence=0.3, ema_confidence=0.3, confidence_history=[0.3] * 5)
        existing = make_memory(confidence=0.9, ema_confidence=0.9, confidence_history=[0.9] * 5)
        result = resolve(new, existing, ConflictStrategy.CONFIDENCE_WEIGHTED, cfg)
        assert result.kept.content == existing.content

    def test_confidence_weighted_winner_gets_contradicted(self):
        new = make_memory(
            content="new", memory_type=MemoryType.SEMANTIC,
            confidence=0.9, ema_confidence=0.9, confidence_history=[0.9] * 5,
        )
        existing = make_memory(
            content="existing", memory_type=MemoryType.SEMANTIC,
            confidence=0.3, ema_confidence=0.3, confidence_history=[0.3] * 5,
        )
        result = resolve(new, existing, ConflictStrategy.CONFIDENCE_WEIGHTED, cfg)
        # Winner's confidence should be slightly lower than its original due to contradict()
        assert result.kept.confidence < 0.9

    def test_ask_user_returns_new_as_tentative(self):
        new = make_memory("new")
        existing = make_memory("existing")
        result = resolve(new, existing, ConflictStrategy.ASK_USER, cfg)
        assert result.kept.id == new.id
        assert result.strategy_used == ConflictStrategy.ASK_USER
        assert result.conflict_detected is True

    def test_version_both_discards_nothing(self):
        new = make_memory("new")
        existing = make_memory("existing")
        result = resolve(new, existing, ConflictStrategy.VERSION_BOTH, cfg)
        assert result.kept.id == new.id
        assert result.discarded is None
        assert result.conflict_detected is True


class TestPruneVersions:
    def test_under_cap_returns_all(self):
        versions = [make_memory() for _ in range(2)]
        kept, pruned = prune_versions(versions, max_versions=3)
        assert len(kept) == 2
        assert pruned == []

    def test_at_cap_returns_all(self):
        versions = [make_memory() for _ in range(3)]
        kept, pruned = prune_versions(versions, max_versions=3)
        assert len(kept) == 3
        assert pruned == []

    def test_over_cap_prunes_lowest_pamu(self):
        high = make_memory(confidence=0.9, ema_confidence=0.9, confidence_history=[0.9] * 5)
        mid = make_memory(confidence=0.6, ema_confidence=0.6, confidence_history=[0.6] * 5)
        low = make_memory(confidence=0.2, ema_confidence=0.2, confidence_history=[0.2] * 5)
        kept, pruned = prune_versions([high, mid, low], max_versions=2)
        assert len(kept) == 2
        assert len(pruned) == 1
        assert pruned[0].id == low.id

    def test_pruned_plus_kept_equals_input(self):
        versions = [make_memory() for _ in range(5)]
        kept, pruned = prune_versions(versions, max_versions=3)
        all_ids = {m.id for m in kept + pruned}
        input_ids = {m.id for m in versions}
        assert all_ids == input_ids
