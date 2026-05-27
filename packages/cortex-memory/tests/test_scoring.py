import math
from datetime import datetime, timedelta, timezone

import pytest

from cortex.memory.scoring import contradict, current_confidence, pamu_score, reinforce
from cortex.memory.types import DecayConfig, Memory, MemoryType


def make_memory(
    memory_type: MemoryType = MemoryType.SEMANTIC,
    confidence: float = 1.0,
    ema_confidence: float = 1.0,
    confidence_history: list[float] | None = None,
    reinforcement_count: int = 0,
) -> Memory:
    return Memory(
        content="test",
        memory_type=memory_type,
        confidence=confidence,
        ema_confidence=ema_confidence,
        confidence_history=confidence_history or [],
        reinforcement_count=reinforcement_count,
    )


cfg = DecayConfig()
NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class TestCurrentConfidence:
    def test_no_elapsed_time_returns_stored_confidence(self):
        m = make_memory(confidence=0.8)
        # confidence_updated_at defaults to now; passing same now → t=0 → e^0 = 1 → 0.8 * 1 = 0.8
        result = current_confidence(m, now=m.confidence_updated_at)
        assert result == pytest.approx(0.8)

    def test_confidence_decays_over_time(self):
        m = make_memory(memory_type=MemoryType.EPISODIC, confidence=1.0)
        # EPISODIC stability = 1.0 day. After 1 day: R = e^(-1/1) ≈ 0.368
        one_day_later = m.confidence_updated_at + timedelta(days=1)
        result = current_confidence(m, now=one_day_later)
        assert result == pytest.approx(math.exp(-1.0), rel=1e-5)

    def test_higher_stability_decays_slower(self):
        episodic = make_memory(memory_type=MemoryType.EPISODIC, confidence=1.0)
        semantic = make_memory(memory_type=MemoryType.SEMANTIC, confidence=1.0)
        one_day_later = episodic.confidence_updated_at + timedelta(days=1)

        r_episodic = current_confidence(episodic, now=one_day_later)
        r_semantic = current_confidence(semantic, now=one_day_later)

        assert r_semantic > r_episodic

    def test_working_decays_fastest(self):
        working = make_memory(memory_type=MemoryType.WORKING, confidence=1.0)
        procedural = make_memory(memory_type=MemoryType.PROCEDURAL, confidence=1.0)
        one_day_later = working.confidence_updated_at + timedelta(days=1)

        assert current_confidence(working, now=one_day_later) < current_confidence(
            procedural, now=one_day_later
        )

    def test_result_bounded_between_zero_and_one(self):
        m = make_memory(confidence=0.5)
        far_future = m.confidence_updated_at + timedelta(days=365)
        result = current_confidence(m, now=far_future)
        assert 0.0 <= result <= 1.0


class TestReinforce:
    def test_boosts_confidence(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC, confidence=0.5)
        reinforced = reinforce(m, cfg)
        assert reinforced.confidence > m.confidence

    def test_confidence_never_exceeds_one(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC, confidence=1.0)
        reinforced = reinforce(m, cfg)
        assert reinforced.confidence <= 1.0

    def test_working_is_noop(self):
        m = make_memory(memory_type=MemoryType.WORKING)
        assert reinforce(m, cfg) is m

    def test_procedural_is_not_noop(self):
        m = make_memory(memory_type=MemoryType.PROCEDURAL, confidence=0.5)
        reinforced = reinforce(m, cfg)
        assert reinforced is not m
        assert reinforced.confidence > m.confidence

    def test_stability_grows(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC)
        reinforced = reinforce(m, cfg)
        assert reinforced.stability > m.stability

    def test_ema_updated(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC, confidence=0.5, ema_confidence=0.5)
        reinforced = reinforce(m, cfg)
        assert reinforced.ema_confidence != m.ema_confidence

    def test_confidence_updated_at_advances(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC)
        reinforced = reinforce(m, cfg, now=NOW)
        assert reinforced.confidence_updated_at == NOW

    def test_reinforcement_count_increments(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC, reinforcement_count=3)
        reinforced = reinforce(m, cfg)
        assert reinforced.reinforcement_count == 4

    def test_history_appended(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC)
        reinforced = reinforce(m, cfg)
        assert len(reinforced.confidence_history) == 1

    def test_history_capped_at_20(self):
        m = make_memory(
            memory_type=MemoryType.SEMANTIC,
            confidence_history=[0.5] * 20,
        )
        reinforced = reinforce(m, cfg)
        assert len(reinforced.confidence_history) == 20

    def test_history_is_new_list_not_mutation(self):
        original_history = [0.5, 0.6]
        m = make_memory(
            memory_type=MemoryType.SEMANTIC,
            confidence_history=original_history,
        )
        reinforce(m, cfg)
        # Original list must not be mutated
        assert original_history == [0.5, 0.6]

    def test_returns_new_object(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC)
        reinforced = reinforce(m, cfg)
        assert reinforced is not m


class TestContradict:
    def test_reduces_confidence(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC, confidence=0.8)
        contradicted = contradict(m, cfg)
        assert contradicted.confidence < m.confidence

    def test_confidence_never_below_zero(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC, confidence=0.01)
        contradicted = contradict(m, cfg)
        assert contradicted.confidence >= 0.0

    def test_working_is_noop(self):
        m = make_memory(memory_type=MemoryType.WORKING)
        assert contradict(m, cfg) is m

    def test_ema_updated(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC, confidence=0.9, ema_confidence=0.9)
        contradicted = contradict(m, cfg)
        assert contradicted.ema_confidence < m.ema_confidence

    def test_confidence_updated_at_advances(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC)
        contradicted = contradict(m, cfg, now=NOW)
        assert contradicted.confidence_updated_at == NOW

    def test_stability_unchanged(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC)
        contradicted = contradict(m, cfg)
        assert contradicted.stability == m.stability

    def test_history_is_new_list_not_mutation(self):
        original_history = [0.8, 0.9]
        m = make_memory(
            memory_type=MemoryType.SEMANTIC,
            confidence_history=original_history,
        )
        contradict(m, cfg)
        assert original_history == [0.8, 0.9]

    def test_returns_new_object(self):
        m = make_memory(memory_type=MemoryType.SEMANTIC)
        assert contradict(m, cfg) is not m

    def test_high_confidence_loses_more_than_low(self):
        high = make_memory(memory_type=MemoryType.SEMANTIC, confidence=0.9)
        low = make_memory(memory_type=MemoryType.SEMANTIC, confidence=0.3)
        drop_high = high.confidence - contradict(high, cfg).confidence
        drop_low = low.confidence - contradict(low, cfg).confidence
        assert drop_high > drop_low


class TestPamuScore:
    def test_no_history_returns_ema(self):
        m = make_memory(ema_confidence=0.75)
        assert pamu_score(m) == pytest.approx(0.75)

    def test_fuses_ema_and_sliding_window(self):
        # history last 5: all 0.5 → SW = 0.5. EMA = 0.9.
        # expected: 0.6 * 0.9 + 0.4 * 0.5 = 0.54 + 0.20 = 0.74
        m = make_memory(ema_confidence=0.9, confidence_history=[0.5] * 5)
        assert pamu_score(m) == pytest.approx(0.74)

    def test_uses_only_last_five_entries(self):
        # First 10 entries at 1.0, last 5 at 0.2
        history = [1.0] * 10 + [0.2] * 5
        m = make_memory(ema_confidence=0.8, confidence_history=history)
        # SW = mean([0.2]*5) = 0.2. score = 0.6*0.8 + 0.4*0.2 = 0.48 + 0.08 = 0.56
        assert pamu_score(m) == pytest.approx(0.56)

    def test_recent_drop_lowers_score_below_ema(self):
        # High EMA from history, recent contradictions drag SW down
        m = make_memory(ema_confidence=0.9, confidence_history=[0.2, 0.2, 0.2, 0.2, 0.2])
        score = pamu_score(m)
        assert score < m.ema_confidence
