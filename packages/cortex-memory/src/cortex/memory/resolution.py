from __future__ import annotations

from typing import Optional

import numpy as np

from .scoring import contradict, pamu_score
from .types import ConflictStrategy, DecayConfig, Memory, ResolveResult


def _jaccard(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def detect_conflicts(
    new_memory: Memory,
    candidates: list[Memory],
    new_embedding: Optional[list[float]] = None,
    candidate_embeddings: Optional[dict[str, list[float]]] = None,
    jaccard_threshold: float = 0.7,
    cosine_threshold: float = 0.75,
    solo_threshold: float = 0.85,
    top_k: int = 5,
) -> list[tuple[Memory, float]]:
    """Fast heuristic pre-filter that returns the top_k most likely conflict candidates,
    ranked by conflict score (highest first).

    This is intentionally NOT the final word on whether a conflict exists — it returns
    candidates for the engine to optionally verify with an LLM call. The engine calls
    resolve() only on confirmed conflicts.

    Signal logic:
    - Both embedder AND entity tags available: AND logic.
      Both signals must agree: cosine >= cosine_threshold AND Jaccard >= jaccard_threshold.
      Individual thresholds are lower (0.75 / 0.7) because they corroborate each other.
    - Only cosine available: cosine >= solo_threshold (0.85, stricter to compensate).
    - Only entity tags available: Jaccard >= solo_threshold.
    - Neither available: no candidates returned.

    Performance: cosine is computed via numpy matrix multiply across all candidates at
    once — single BLAS call rather than a Python loop. Jaccard is O(1) per candidate
    via set operations.

    Returns list of (candidate, score) sorted by score descending.
    The score is the combined signal: cosine + jaccard (normalised to [0, 2]) when
    both are available, or the single signal (normalised to [0, 1]) otherwise.
    """
    results: list[tuple[Memory, float]] = []

    has_embeddings = (
        new_embedding is not None
        and candidate_embeddings is not None
        and len(candidate_embeddings) > 0
    )

    cosine_scores: dict[str, float] = {}
    if has_embeddings:
        assert new_embedding is not None
        assert candidate_embeddings is not None

        ids_with_embs = [
            c.id for c in candidates
            if c.id != new_memory.id and c.id in candidate_embeddings
        ]
        if ids_with_embs:
            matrix = np.array([candidate_embeddings[cid] for cid in ids_with_embs])
            new_vec = np.array(new_embedding)

            # Vectorised cosine: (matrix @ new_vec) / (row_norms * new_norm)
            # All N cosines computed in a single matrix multiply — BLAS-level speed.
            dots = matrix @ new_vec
            row_norms = np.linalg.norm(matrix, axis=1)
            new_norm = np.linalg.norm(new_vec)
            denom = row_norms * new_norm
            # Where denom is zero (zero-vector embeddings), cosine is 0
            cosines = np.where(denom > 0, dots / denom, 0.0)
            cosine_scores = dict(zip(ids_with_embs, cosines.tolist()))

    for candidate in candidates:
        if candidate.id == new_memory.id:
            continue

        cos = cosine_scores.get(candidate.id)
        jac = (
            _jaccard(new_memory.entity_tags, candidate.entity_tags)
            if new_memory.entity_tags and candidate.entity_tags
            else None
        )

        if cos is not None and jac is not None:
            # Both signals available — AND logic with lower individual thresholds
            if cos >= cosine_threshold and jac >= jaccard_threshold:
                results.append((candidate, cos + jac))

        elif cos is not None:
            # Cosine only — stricter solo threshold
            if cos >= solo_threshold:
                results.append((candidate, cos))

        elif jac is not None:
            # Jaccard only — stricter solo threshold
            if jac >= solo_threshold:
                results.append((candidate, jac))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def resolve(
    new_memory: Memory,
    existing: Memory,
    strategy: ConflictStrategy,
    decay_config: DecayConfig,
) -> ResolveResult:
    if strategy == ConflictStrategy.LAST_WRITE_WINS:
        return ResolveResult(
            kept=new_memory,
            discarded=existing,
            strategy_used=strategy,
            conflict_detected=True,
        )

    if strategy == ConflictStrategy.CONFIDENCE_WEIGHTED:
        new_wins = pamu_score(new_memory) >= pamu_score(existing)
        winner = new_memory if new_wins else existing
        loser = existing if new_wins else new_memory
        # Winner gets a small contradict() — being challenged is a weak negative signal
        # even when you win. Prevents memories from becoming unassailable.
        return ResolveResult(
            kept=contradict(winner, decay_config),
            discarded=loser,
            strategy_used=strategy,
            conflict_detected=True,
        )

    if strategy == ConflictStrategy.ASK_USER:
        # resolve() cannot block for user input. Return a tentative result — caller sees
        # strategy_used=ASK_USER and surfaces both options to the application layer.
        return ResolveResult(
            kept=new_memory,
            discarded=existing,
            strategy_used=strategy,
            conflict_detected=True,
        )

    if strategy == ConflictStrategy.VERSION_BOTH:
        # Both memories survive. Engine calls prune_versions() separately to enforce cap.
        return ResolveResult(
            kept=new_memory,
            discarded=None,
            strategy_used=strategy,
            conflict_detected=True,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def prune_versions(
    versions: list[Memory],
    max_versions: int,
) -> tuple[list[Memory], list[Memory]]:
    if len(versions) <= max_versions:
        return versions, []
    sorted_versions = sorted(versions, key=pamu_score, reverse=True)
    return sorted_versions[:max_versions], sorted_versions[max_versions:]
