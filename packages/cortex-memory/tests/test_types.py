from datetime import datetime, timezone

import pytest

from cortex.memory.types import (
    AddRequest,
    DecayConfig,
    Link,
    LinkRelationship,
    Memory,
    MemoryStatus,
    MemoryType,
    SearchQuery,
)


def make_memory(memory_type: MemoryType, stability: float = 0.0) -> Memory:
    return Memory(content="test", memory_type=memory_type, stability=stability)


class TestStabilityValidator:
    def test_working_gets_default_stability(self):
        m = make_memory(MemoryType.WORKING)
        assert m.stability == DecayConfig().stability_working

    def test_episodic_gets_default_stability(self):
        m = make_memory(MemoryType.EPISODIC)
        assert m.stability == DecayConfig().stability_episodic

    def test_semantic_gets_default_stability(self):
        m = make_memory(MemoryType.SEMANTIC)
        assert m.stability == DecayConfig().stability_semantic

    def test_procedural_gets_default_stability(self):
        m = make_memory(MemoryType.PROCEDURAL)
        assert m.stability == DecayConfig().stability_procedural

    def test_explicit_stability_preserved(self):
        # Sentinel is 0.0 — any other value should survive the validator
        m = Memory(content="test", memory_type=MemoryType.SEMANTIC, stability=3.5)
        assert m.stability == 3.5

    def test_model_copy_bypasses_validator(self):
        m = make_memory(MemoryType.SEMANTIC)
        m2 = m.model_copy(update={"stability": 99.0})
        assert m2.stability == 99.0


class TestDecayConfigHelpers:
    def setup_method(self):
        self.cfg = DecayConfig()

    def test_stability_for_all_types(self):
        assert self.cfg.stability_for(MemoryType.WORKING) == 0.1
        assert self.cfg.stability_for(MemoryType.EPISODIC) == 1.0
        assert self.cfg.stability_for(MemoryType.SEMANTIC) == 7.0
        assert self.cfg.stability_for(MemoryType.PROCEDURAL) == 30.0

    def test_reinforce_factor_working_is_zero(self):
        assert self.cfg.reinforce_factor_for(MemoryType.WORKING) == 0.0

    def test_reinforce_factor_procedural_is_nonzero(self):
        assert self.cfg.reinforce_factor_for(MemoryType.PROCEDURAL) > 0.0

    def test_contradict_weight_working_is_zero(self):
        assert self.cfg.contradict_weight_for(MemoryType.WORKING) == 0.0

    def test_tier_boost_semantic_highest(self):
        assert (
            self.cfg.tier_boost_for(MemoryType.SEMANTIC)
            >= self.cfg.tier_boost_for(MemoryType.EPISODIC)
            >= self.cfg.tier_boost_for(MemoryType.PROCEDURAL)
        )


class TestMemoryDefaults:
    def test_id_is_assigned(self):
        m = Memory(content="x", memory_type=MemoryType.EPISODIC)
        assert m.id and len(m.id) == 36  # uuid4 format

    def test_two_memories_have_different_ids(self):
        a = Memory(content="x", memory_type=MemoryType.EPISODIC)
        b = Memory(content="x", memory_type=MemoryType.EPISODIC)
        assert a.id != b.id

    def test_timestamps_are_utc(self):
        m = Memory(content="x", memory_type=MemoryType.EPISODIC)
        assert m.event_time.tzinfo == timezone.utc
        assert m.ingestion_time.tzinfo == timezone.utc
        assert m.confidence_updated_at.tzinfo == timezone.utc

    def test_a_mem_fields_default_empty(self):
        m = Memory(content="x", memory_type=MemoryType.SEMANTIC)
        assert m.keywords == []
        assert m.contextual_description == ""
        assert m.links == []
        assert m.retrieval_notes == ""

    def test_scope_fields_default_none(self):
        m = Memory(content="x", memory_type=MemoryType.SEMANTIC)
        assert m.user_id is None
        assert m.agent_id is None
        assert m.run_id is None

    def test_source_defaults_to_user(self):
        m = Memory(content="x", memory_type=MemoryType.SEMANTIC)
        assert m.source == "user"


class TestLinkAndStatus:
    def test_link_defaults_related(self):
        link = Link(target_id="abc-123")
        assert link.relationship == LinkRelationship.RELATED

    def test_link_relationship_settable(self):
        link = Link(target_id="abc-123", relationship=LinkRelationship.SUPERSEDES)
        assert link.relationship == LinkRelationship.SUPERSEDES

    def test_memory_links_hold_typed_links(self):
        m = Memory(
            content="x",
            memory_type=MemoryType.SEMANTIC,
            links=[Link(target_id="other-id", relationship=LinkRelationship.SUPPORTS)],
        )
        assert len(m.links) == 1
        assert m.links[0].target_id == "other-id"
        assert m.links[0].relationship == LinkRelationship.SUPPORTS

    def test_status_defaults_active(self):
        m = Memory(content="x", memory_type=MemoryType.SEMANTIC)
        assert m.status == MemoryStatus.ACTIVE

    def test_superseded_by_defaults_none(self):
        m = Memory(content="x", memory_type=MemoryType.SEMANTIC)
        assert m.superseded_by is None

    def test_superseded_status_and_pointer(self):
        m = Memory(
            content="x",
            memory_type=MemoryType.SEMANTIC,
            status=MemoryStatus.SUPERSEDED,
            superseded_by="other-memory-id",
        )
        assert m.status == MemoryStatus.SUPERSEDED
        assert m.superseded_by == "other-memory-id"


class TestAddRequestDefaults:
    def test_memory_type_defaults_semantic(self):
        r = AddRequest(content="x")
        assert r.memory_type == MemoryType.SEMANTIC

    def test_context_defaults_empty(self):
        r = AddRequest(content="x")
        assert r.context == []

    def test_scope_fields_default_none(self):
        r = AddRequest(content="x")
        assert r.user_id is None
        assert r.agent_id is None
        assert r.run_id is None


class TestSearchQueryDefaults:
    def test_event_time_filters_default_none(self):
        q = SearchQuery(query="test")
        assert q.event_time_from is None
        assert q.event_time_to is None

    def test_scope_filters_default_none(self):
        q = SearchQuery(query="test")
        assert q.user_id is None
        assert q.agent_id is None
        assert q.run_id is None

    def test_context_defaults_empty(self):
        q = SearchQuery(query="test")
        assert q.context == []
