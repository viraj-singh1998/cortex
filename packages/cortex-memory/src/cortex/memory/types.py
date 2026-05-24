from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class ConflictStrategy(str, Enum):
    LAST_WRITE_WINS = "last_write_wins"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ASK_USER = "ask_user"
    VERSION_BOTH = "version_both"


@dataclass
class DecayConfig:
    # Ebbinghaus stability per type (days). Formula: R = confidence * e^(-t/stability)
    # Higher = slower decay. WORKING is ephemeral; PROCEDURAL survives weeks.
    stability_working: float = 0.1
    stability_episodic: float = 1.0
    stability_semantic: float = 7.0
    stability_procedural: float = 30.0

    # How much reinforce() boosts confidence, per type.
    # SEMANTIC gets a stronger boost because preference signals matter more.
    # PROCEDURAL gets a weak boost because workflows shouldn't flip from casual mentions.
    reinforce_factor_episodic: float = 0.2
    reinforce_factor_semantic: float = 0.25
    reinforce_factor_procedural: float = 0.1

    # How much contradict() drags confidence down, per type.
    contradict_weight_episodic: float = 0.3
    contradict_weight_semantic: float = 0.4
    contradict_weight_procedural: float = 0.15

    # When a memory is reinforced/contradicted, its directly-linked neighbours (1 hop only,
    # never recursive) also receive a weaker version of the same signal.
    # hop_reinforce_decay=0.5 means a neighbour gets 50% of the reinforcement boost.
    # hop_contradict_decay=0.4 means a neighbour gets 40% of the contradiction penalty.
    hop_reinforce_decay: float = 0.5
    hop_contradict_decay: float = 0.4

    # inject() ranks memories by pamu_score × recency_weight.
    # recency_weight = e^(-days_since_ingestion / recency_halflife_days)
    recency_halflife_days: float = 14.0

    def stability_for(self, memory_type: MemoryType) -> float:
        return {
            MemoryType.WORKING: self.stability_working,
            MemoryType.EPISODIC: self.stability_episodic,
            MemoryType.SEMANTIC: self.stability_semantic,
            MemoryType.PROCEDURAL: self.stability_procedural,
        }[memory_type]

    def reinforce_factor_for(self, memory_type: MemoryType) -> float:
        # WORKING and PROCEDURAL never get reinforced — ephemeral and deliberate respectively
        return {
            MemoryType.WORKING: 0.0,
            MemoryType.EPISODIC: self.reinforce_factor_episodic,
            MemoryType.SEMANTIC: self.reinforce_factor_semantic,
            MemoryType.PROCEDURAL: self.reinforce_factor_procedural,
        }[memory_type]

    def contradict_weight_for(self, memory_type: MemoryType) -> float:
        return {
            MemoryType.WORKING: 0.0,
            MemoryType.EPISODIC: self.contradict_weight_episodic,
            MemoryType.SEMANTIC: self.contradict_weight_semantic,
            MemoryType.PROCEDURAL: self.contradict_weight_procedural,
        }[memory_type]


# Module-level default used by Memory's validator so the validator doesn't need
# an engine reference. The engine can override stability via model_copy() when it
# has a custom DecayConfig.
_DEFAULT_DECAY = DecayConfig()


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    memory_type: MemoryType

    # Multi-scope identity fields. All optional — if None, the memory is unscoped
    # (shared across all users/agents/sessions). Set these to isolate memories in
    # multi-user or multi-agent deployments. The store pre-filters on these before
    # any BM25 or semantic scoring runs.
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None

    # Who generated this memory.
    # "user"  → the user explicitly stated this fact ("I prefer dark mode")
    # "agent" → the LLM inferred this from context (retain_node extraction)
    # Agent inferences can be wrong; this field lets callers weight or filter accordingly.
    source: str = "user"

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # Ebbinghaus stability. Sentinel 0.0 means "set from type" — the validator below
    # replaces it with the type-specific default from _DEFAULT_DECAY.
    # Engine overrides this via model_copy() when using a custom DecayConfig.
    stability: float = Field(default=0.0, ge=0.0)

    # A-MEM fields — populated by engine._enrich() at add() time.
    # keywords and contextual_description feed BM25; links form the semantic graph.
    # content and contextual_description are immutable after creation (production safety rule):
    # evolution only ever touches keywords and entity_tags.
    keywords: list[str] = Field(default_factory=list)
    contextual_description: str = ""
    links: list[str] = Field(default_factory=list)

    # Bi-temporal timestamps (Graphiti pattern).
    # event_time = when the fact was true in the world.
    # ingestion_time = when the agent learned it.
    # These diverge constantly: "Last Thursday I quit coffee" told on Monday →
    # event_time=Thursday, ingestion_time=Monday.
    event_time: datetime = Field(default_factory=_now)
    ingestion_time: datetime = Field(default_factory=_now)
    last_reinforced: datetime = Field(default_factory=_now)

    access_count: int = 0
    reinforcement_count: int = 0
    entity_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # PAMU-style dual confidence tracking.
    # confidence_history stores raw values (capped at 20 entries to bound memory).
    # ema_confidence is the exponential moving average (alpha=0.3), updated on every
    # reinforce/contradict call. pamu_score() fuses EMA + sliding window.
    confidence_history: list[float] = Field(default_factory=list)
    ema_confidence: float = 1.0

    @model_validator(mode="after")
    def _set_stability_from_type(self) -> Memory:
        # Only fires when stability is the sentinel (not explicitly set by caller).
        # model_copy() bypasses validators, so engine overrides are safe.
        if self.stability == 0.0:
            self.stability = _DEFAULT_DECAY.stability_for(self.memory_type)
        return self


class AddRequest(BaseModel):
    content: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    event_time: Optional[datetime] = None
    entity_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Mirror of Memory's scope fields — pass these and the engine stamps them onto
    # the created Memory object. Leave None for unscoped (global) memories.
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    source: str = "user"

    # Surrounding conversation turns passed to _enrich() so the LLM can generate
    # richer keywords and contextual_description than it could from content alone.
    # Format: [{"role": "user"|"assistant", "content": "..."}]
    # Caller decides how many turns to include (retain_node default: last 6 turns).
    context: list[dict[str, str]] = Field(default_factory=list)


class SearchQuery(BaseModel):
    query: str
    memory_types: list[MemoryType] = Field(default_factory=list)
    max_results: int = 10
    min_confidence: float = 0.0
    entity_filter: list[str] = Field(default_factory=list)

    # Scope filters — applied as hard pre-filters before BM25/semantic scoring.
    # Only memories matching ALL provided scopes are candidates.
    # Leave None to search across that dimension (e.g. user_id=None searches all users).
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None

    # Bi-temporal range filter. Applied before BM25/semantic scoring so only
    # memories whose event_time falls in range are even candidates.
    event_time_from: Optional[datetime] = None
    event_time_to: Optional[datetime] = None

    # Same idea as AddRequest.context — recent turns make the query embedding richer
    # and retrieval more precise when the last message is ambiguous.
    context: list[dict[str, str]] = Field(default_factory=list)


class SearchResult(BaseModel):
    memory: Memory
    score: float


class ResolveResult(BaseModel):
    kept: Memory
    discarded: Optional[Memory]
    strategy_used: ConflictStrategy
    conflict_detected: bool


class InjectResult(BaseModel):
    memories: list[Memory]
    formatted: str
    token_estimate: int
    truncated: bool = False


# ---------------------------------------------------------------------------
# Storage protocols — structural subtyping (duck typing).
# Any class implementing these methods is a valid store without explicit inheritance.
# @runtime_checkable enables isinstance() checks at runtime.
# Swap InMemoryStore for SQLiteStore or RedisStore without touching the engine.
# ---------------------------------------------------------------------------

@runtime_checkable
class MemoryStore(Protocol):
    def get(self, memory_id: str) -> Optional[Memory]: ...
    def put(self, memory: Memory) -> None: ...
    def delete(self, memory_id: str) -> None: ...
    def all(self) -> list[Memory]: ...
    def filter(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[Memory]:
        # Return memories matching ALL provided scope constraints.
        # None on any dimension means "don't filter on this dimension".
        # Future stores (SQLite, Redis) translate these into WHERE clauses.
        ...


@runtime_checkable
class EmbeddingStore(Protocol):
    def put(self, memory_id: str, embedding: list[float]) -> None: ...
    def get(self, memory_id: str) -> Optional[list[float]]: ...
    def all(self) -> dict[str, list[float]]: ...
    def delete(self, memory_id: str) -> None: ...
