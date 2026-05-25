from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .types import DecayConfig, Memory


def current_confidence(memory: Memory, now: Optional[datetime] = None) -> float:
    """How confident should we be in this memory right now, accounting for time-based decay?

    Uses the Ebbinghaus forgetting curve: R = stored_confidence × e^(-t / stability)
    where t = days elapsed since confidence_updated_at (the last time confidence changed).

    confidence_updated_at is the clock start — the moment when the stored confidence
    value was accurate. Both reinforce() and contradict() reset it so decay always
    runs forward from the most recent update, not from some earlier reinforcement.

    Returns a float in [0.0, 1.0]. Does not mutate the memory.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elapsed_days = (now - memory.confidence_updated_at).total_seconds() / 86_400
    return memory.confidence * math.exp(-elapsed_days / memory.stability)


def reinforce(
    memory: Memory,
    decay_config: DecayConfig,
    now: Optional[datetime] = None,
) -> Memory:
    """Boost a memory's confidence and grow its stability (the spacing effect).

    The engine calls this in three situations:
      1. inject() — every retrieved memory is reinforced (retrieval = implicit confirmation
         via spaced repetition. WORKING memories bypass this — they're always present,
         never ranked/retrieved, so there's nothing to reinforce.)
      2. _store() — when new content is near-identical to an existing memory,
         reinforce the existing one rather than creating a duplicate.
      3. engine.reinforce(memory_id) — public API for external verification signals.

    WORKING is the only true no-op (factor=0.0). PROCEDURAL gets weak reinforcement
    (factor=0.1 by default) — workflows shouldn't flip from casual mentions, but a
    confirmed workflow deserves a small durability bump. What IS skipped for PROCEDURAL
    is retroactive propagation to linked memories — that happens in the engine, not here.

    Returns a NEW Memory with updated fields. Caller must write it back to the store.
    We never mutate in place — the engine always does store.put(reinforced_memory).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    factor = decay_config.reinforce_factor_for(memory.memory_type)
    if factor == 0.0:
        return memory  # WORKING only — ephemeral, reinforcement is meaningless

    # Asymptotic confidence boost: adds factor × (headroom remaining).
    # As confidence approaches 1.0, each boost contributes less — prevents a single
    # strong signal from snapping confidence to 1.0 unrealistically.
    new_confidence = min(1.0, memory.confidence + factor * (1.0 - memory.confidence))

    # Stability grows with each reinforcement — the spacing effect.
    # (Research finding #4: "each reinforce() call increases stability, making the
    # memory more resistant to decay. This is what makes frequently-accessed memories durable.")
    # Scale growth by the reinforce factor so stronger memory types compound faster.
    new_stability = memory.stability * (1.0 + factor * 0.5)

    # EMA update (alpha=0.3): weights recent observations without discarding history.
    # 0.3 is the PAMU paper's value — sluggish enough for a stable long-term signal.
    new_ema = 0.3 * new_confidence + 0.7 * memory.ema_confidence

    # IMPORTANT: always build a new list, never mutate in place.
    # model_copy() is shallow — confidence_history would be shared between the old and
    # new Memory instances if you appended to the existing list. Cap at 20 entries.
    new_history = (memory.confidence_history + [new_confidence])[-20:]

    return memory.model_copy(update={
        "confidence": new_confidence,
        "stability": new_stability,
        "ema_confidence": new_ema,
        "confidence_history": new_history,
        "confidence_updated_at": now,  # reset decay clock to now
        "reinforcement_count": memory.reinforcement_count + 1,
        "access_count": memory.access_count + 1,
    })


def contradict(
    memory: Memory,
    decay_config: DecayConfig,
    now: Optional[datetime] = None,
) -> Memory:
    """Drag a memory's confidence down when contradicting evidence is found.

    The engine calls this in two situations:
      1. _apply_resolution() CONFIDENCE_WEIGHTED — even the winning memory gets a small
         contradict() because being challenged is a weak negative signal. Prevents
         overconfident stale memories from being unassailable.
      2. _retroactive_reasoning() — when the LLM determines a linked memory opposes
         newly ingested information.

    WORKING is the only no-op (weight=0.0). Stability is NOT changed — contradiction
    affects trustworthiness, not durability.

    confidence_updated_at IS updated — critical for the decay formula. The stored
    confidence value changes here, so the decay clock must reset to now. Without this,
    current_confidence() would apply decay from the old timestamp to the already-reduced
    confidence, double-penalising the memory.

    Returns a NEW Memory. Caller must write it back to the store.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    weight = decay_config.contradict_weight_for(memory.memory_type)
    if weight == 0.0:
        return memory  # WORKING only — no-op

    # Proportional confidence drag: high-confidence memories lose more on contradiction.
    # A 0.9-confidence memory hit with weight=0.4 drops to 0.54.
    # A 0.3-confidence memory hit with the same weight drops to 0.18.
    new_confidence = max(0.0, memory.confidence * (1.0 - weight))

    new_ema = 0.3 * new_confidence + 0.7 * memory.ema_confidence
    new_history = (memory.confidence_history + [new_confidence])[-20:]

    return memory.model_copy(update={
        "confidence": new_confidence,
        "ema_confidence": new_ema,
        "confidence_history": new_history,
        "confidence_updated_at": now,  # confidence changed — reset decay clock
    })


def pamu_score(memory: Memory) -> float:
    """Fused confidence score combining long-term trend (EMA) and recent shifts (SW).

    PAMU = Preference-Aware Memory Update (arXiv:2510.09720). The score itself is a
    confidence measurement — it's not about injection formatting. It's USED as the
    ranking signal that decides which memories get injected, and which memory wins
    a CONFIDENCE_WEIGHTED conflict.

    Formula: score = 0.6 × EMA + 0.4 × mean(last 5 confidence_history entries)

    Why two signals instead of raw confidence?
    - EMA (alpha=0.3): slow-moving, tracks long-term trustworthiness. A memory
      confirmed 20 times over months has high EMA even if raw confidence dipped recently.
    - SW (last 5): fast-moving, catches sudden recent shifts. If a memory was
      contradicted twice last week, SW drops even while EMA is still high.

    The fusion means a stale high-confidence memory loses to a consistently stable
    lower-confidence one — which is the right behaviour for preference injection.

    Falls back to ema_confidence alone if no history exists yet (brand new memory).
    """
    history = memory.confidence_history
    if not history:
        return memory.ema_confidence

    recent = history[-5:]
    sw = sum(recent) / len(recent)
    return 0.6 * memory.ema_confidence + 0.4 * sw
