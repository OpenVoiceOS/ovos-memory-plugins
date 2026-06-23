# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Shared retrieval primitives for memory plugins.

``MemoryHit`` is the common currency a *retrieval* memory returns from its
public ``search()`` method; the fusion helpers below merge ranked hit-lists from
several retrievers into one list. :class:`~ovos_memory_plugins.composite.CompositeMemory`
uses these to consolidate hybrid (e.g. semantic + lexical) recall.

A "retriever" memory is any object exposing::

    search(query: str, session_id: str = None, top_k: int = None) -> List[MemoryHit]

This is duck-typed (no ABC), so any third-party memory plugin can participate in
fusion just by exposing that method.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Tuple


@dataclass
class MemoryHit:
    """A single retrieved memory.

    Attributes:
        content: The recalled text (e.g. a ``"Q: ...\\nA: ..."`` exchange).
        source: Stored document/source id, if any.
        score: The retriever-native relevance score. Scales differ between
            retrievers (semantic cosine is ~0..1, lexical BM25 is unbounded), so
            do NOT compare scores across retrievers directly — fuse by rank
            instead (see :func:`fuse_rrf`).
        metadata: Free-form metadata (e.g. ``session_id``, ``retriever`` name).
    """
    content: str
    source: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Each fusion function takes a list of (name, weight, hits) per member retriever
# and returns a single fused, ranked list of MemoryHit.
RankedLists = List[Tuple[str, float, List[MemoryHit]]]

FUSION_MODES = ("rrf", "weighted", "merge", "priority", "interleave")


def _content_key(hit: MemoryHit) -> str:
    """Stable dedup key for a hit (normalized content hash)."""
    return hashlib.sha1(hit.content.strip().encode("utf-8")).hexdigest()


def fuse_rrf(lists: RankedLists, k: int = 60) -> List[MemoryHit]:
    """Reciprocal Rank Fusion — the recommended default.

    ``score(d) = Σ_lists  weight · 1 / (k + rank)`` where ``rank`` is the 1-based
    position of ``d`` within each list it appears in. Because it uses **rank, not
    raw score**, RRF is immune to the score-scale mismatch between different
    retrievers (semantic cosine vs lexical BM25), which is exactly what makes it
    the right default for hybrid recall. Duplicate content is merged; the fused
    hit keeps the highest per-list score for display.
    """
    fused: Dict[str, MemoryHit] = {}
    scores: Dict[str, float] = {}
    for _name, weight, hits in lists:
        for rank, hit in enumerate(hits, start=1):
            key = _content_key(hit)
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k + rank))
            if key not in fused or hit.score > fused[key].score:
                fused[key] = hit
    ranked = sorted(fused.values(), key=lambda h: scores[_content_key(h)], reverse=True)
    # return fresh objects — never mutate the hits a member retriever handed us
    return [replace(h, metadata={**h.metadata, "fusion_score": scores[_content_key(h)]})
            for h in ranked]


def _normalized(hits: List[MemoryHit]) -> Dict[str, float]:
    """Min-max normalize a list's scores into 0..1 keyed by content hash."""
    if not hits:
        return {}
    vals = [h.score for h in hits]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    out: Dict[str, float] = {}
    for h in hits:
        out[_content_key(h)] = 1.0 if span == 0 else (h.score - lo) / span
    return out


def fuse_weighted(lists: RankedLists) -> List[MemoryHit]:
    """Weighted sum of per-list min-max-normalized scores.

    Normalization neutralizes scale differences within each list before summing.
    Still assumes per-list scores are meaningful; prefer :func:`fuse_rrf` when in
    doubt.
    """
    fused: Dict[str, MemoryHit] = {}
    scores: Dict[str, float] = {}
    for _name, weight, hits in lists:
        norm = _normalized(hits)
        for hit in hits:
            key = _content_key(hit)
            scores[key] = scores.get(key, 0.0) + weight * norm[key]
            if key not in fused or hit.score > fused[key].score:
                fused[key] = hit
    ranked = sorted(fused.values(), key=lambda h: scores[_content_key(h)], reverse=True)
    return [replace(h, metadata={**h.metadata, "fusion_score": scores[_content_key(h)]})
            for h in ranked]


def fuse_merge(lists: RankedLists, dedup: bool = True) -> List[MemoryHit]:
    """Concatenate every list and sort by raw score (keeping max per duplicate).

    Simplest mode. Scores across retrievers are not strictly comparable, so this
    is best-effort; use when members share a scale or you just want a union.
    """
    if not dedup:
        merged = [h for _n, _w, hits in lists for h in hits]
        return sorted(merged, key=lambda h: h.score, reverse=True)
    best: Dict[str, MemoryHit] = {}
    for _name, _weight, hits in lists:
        for hit in hits:
            key = _content_key(hit)
            if key not in best or hit.score > best[key].score:
                best[key] = hit
    return sorted(best.values(), key=lambda h: h.score, reverse=True)


def fuse_priority(lists: RankedLists) -> List[MemoryHit]:
    """Return the first member (in order) that produced any hits; ignore the rest.

    Useful as a fallback chain (e.g. "prefer semantic, fall back to lexical").
    """
    for _name, _weight, hits in lists:
        if hits:
            return list(hits)
    return []


def fuse_interleave(lists: RankedLists, dedup: bool = True) -> List[MemoryHit]:
    """Round-robin: take rank-0 of each list, then rank-1, etc.

    Rank-based and scale-free like RRF, but coarser. Members contribute in config
    order.
    """
    out: List[MemoryHit] = []
    seen = set()
    depth = max((len(hits) for _n, _w, hits in lists), default=0)
    for i in range(depth):
        for _name, _weight, hits in lists:
            if i < len(hits):
                hit = hits[i]
                if dedup:
                    key = _content_key(hit)
                    if key in seen:
                        continue
                    seen.add(key)
                out.append(hit)
    return out


def fuse(mode: str, lists: RankedLists, *, rrf_k: int = 60,
         dedup: bool = True) -> List[MemoryHit]:
    """Dispatch to the fusion function named by ``mode``."""
    if mode == "rrf":
        return fuse_rrf(lists, k=rrf_k)
    if mode == "weighted":
        return fuse_weighted(lists)
    if mode == "merge":
        return fuse_merge(lists, dedup=dedup)
    if mode == "priority":
        return fuse_priority(lists)
    if mode == "interleave":
        return fuse_interleave(lists, dedup=dedup)
    raise ValueError(f"unknown fusion mode {mode!r}; expected one of {FUSION_MODES}")
