# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Composite (ensemble) memory: a pure OPM orchestrator over other memories.

``CompositeMemory`` owns no storage or retrieval of its own. It loads a list of
**member** memory plugins by name through ``ovos_plugin_manager`` and consolidates
them so a persona's single ``memory_module`` slot can combine many memories:

- members exposing a ``search()`` (retrievers — local-rag, lexical, …) have their
  hits **fused** into one ranked, deduplicated list (RRF by default, which is
  immune to the score-scale mismatch between e.g. semantic and lexical backends);
- members that only manage context (longterm summary, entity facts, recency)
  contribute their leading **system** block(s);
- the fused recall block is injected per the composite's own ``inject_mode`` and
  the verbatim history comes from a designated ``primary`` member.

``update_history`` is written through to **every** member, so each advances its own
state. Members that fail to load or raise at runtime are skipped — the composite
keeps working with whatever remains.

Configuration (persona JSON block)::

    {
      "memory_module": "ovos-memory-plugin-composite",
      "ovos-memory-plugin-composite": {
        "members": [
          {"module": "ovos-memory-plugin-local-rag", "weight": 1.0, "config": {...}},
          {"module": "ovos-memory-plugin-lexical",   "weight": 1.0, "config": {...}},
          {"module": "ovos-memory-plugin-entity",    "config": {...}}
        ],
        "primary": "ovos-memory-plugin-local-rag",  # history source (default: first member)
        "fusion": "rrf",            # rrf | weighted | merge | priority | interleave
        "rrf_k": 60,
        "max_num_results": 5,       # final fused top-k
        "dedup": true,
        "inject_mode": "system",
        "system_prompt": "You are a helpful assistant."
      }
    }

Note: give each member its own storage path/collection in its ``config`` block so
retriever members don't collide, and leave ``system_prompt`` to the composite
(member system prompts, if set, are folded in as extra system blocks).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ovos_plugin_manager.agents import load_memory_plugin
from ovos_plugin_manager.templates.agents import AgentContextManager, AgentMessage, MessageRole
from ovos_utils.log import LOG

from ovos_memory_plugins.base import BaseRetrievalMemory
from ovos_memory_plugins.common import MemoryHit, FUSION_MODES, fuse

COMPOSITE_PLUGIN_NAME = "ovos-memory-plugin-composite"

# member tuple: (module_name, instance, weight)
Member = Tuple[str, AgentContextManager, float]


class CompositeMemory(BaseRetrievalMemory):
    """Ensemble memory that loads and consolidates several member plugins.

    Subclasses :class:`~ovos_memory_plugins.base.BaseRetrievalMemory` only to reuse
    its context renderer / inject-mode assembly; its own retrieval is a no-op — the
    hits come from members. See module docstring for the config schema.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.fusion: str = self.config.get("fusion", "rrf")
        if self.fusion not in FUSION_MODES:
            raise ValueError(f"fusion must be one of {FUSION_MODES}, got {self.fusion!r}")
        self.rrf_k: int = int(self.config.get("rrf_k", 60))
        self.dedup: bool = bool(self.config.get("dedup", True))
        # the composite exposes max_num_results at the top level (more intuitive than
        # nesting it under `retrieval`); fall back to the base-parsed value.
        self.max_num_results = int(self.config.get("max_num_results", self.max_num_results))
        # how many candidates to pull per member before fusion (over-fetch a bit)
        self.member_fetch_k: int = int(
            self.config.get("member_fetch_k", max(self.max_num_results * 2, self.max_num_results)))

        self.members: List[Member] = self._load_members()
        self.retrievers: List[Member] = [m for m in self.members if hasattr(m[1], "search")]
        self.plain: List[Member] = [m for m in self.members if not hasattr(m[1], "search")]
        self.primary: Optional[AgentContextManager] = self._resolve_primary()

    # storage hooks are unused — the composite never stores/queries itself
    def _store_document(self, doc_id, text, session_id, metadata) -> None:  # pragma: no cover
        pass

    def _query_backend(self, query, top_k) -> List[MemoryHit]:
        return []

    # ------------------------------------------------------------------ loading
    def _load_members(self) -> List[Member]:
        members: List[Member] = []
        for spec in self.config.get("members", []):
            name = spec.get("module") if isinstance(spec, dict) else spec
            if not name:
                continue
            if name == COMPOSITE_PLUGIN_NAME:
                LOG.error("CompositeMemory: refusing to load itself as a member; skipping")
                continue
            try:
                cls = load_memory_plugin(name)
                if cls is None:
                    raise ValueError("plugin not found")
                inst = cls(config=(spec.get("config") if isinstance(spec, dict) else None) or {})
                weight = float(spec.get("weight", 1.0)) if isinstance(spec, dict) else 1.0
                members.append((name, inst, weight))
                LOG.debug(f"CompositeMemory: loaded member {name!r} (weight={weight})")
            except Exception as e:
                LOG.error(f"CompositeMemory: failed to load member {name!r}: {e}")
        if not members:
            LOG.warning("CompositeMemory: no members loaded; behaving as a passthrough")
        return members

    def _resolve_primary(self) -> Optional[AgentContextManager]:
        primary_name = self.config.get("primary")
        if primary_name:
            for name, inst, _w in self.members:
                if name == primary_name:
                    return inst
            LOG.warning(f"CompositeMemory: primary {primary_name!r} not among members; "
                        f"using first member")
        return self.members[0][1] if self.members else None

    # ----------------------------------------------- AgentContextManager API
    def get_history(self, session_id: str) -> List[AgentMessage]:
        """Return the designated ``primary`` member's history (or empty)."""
        if self.primary is None:
            return []
        try:
            return self.primary.get_history(session_id)
        except Exception as e:
            LOG.error(f"CompositeMemory: primary get_history failed: {e}")
            return []

    def update_history(self, new_messages: List[AgentMessage], session_id: str) -> None:
        """Write ``new_messages`` through to every member."""
        for name, inst, _w in self.members:
            try:
                inst.update_history(new_messages, session_id)
            except Exception as e:
                LOG.error(f"CompositeMemory: member {name!r} update_history failed: {e}")

    # ----------------------------------------------------------- consolidation
    def _gather(self, query: str, session_id: str) -> List[MemoryHit]:
        """Fuse hits from all retriever members into one ranked list."""
        ranked_lists = []
        for name, inst, weight in self.retrievers:
            try:
                hits = inst.search(query, session_id=session_id, top_k=self.member_fetch_k) or []
            except Exception as e:
                LOG.error(f"CompositeMemory: member {name!r} search failed: {e}")
                hits = []
            for h in hits:
                h.metadata = {**(h.metadata or {}), "retriever": name}
            ranked_lists.append((name, weight, hits))
        fused = fuse(self.fusion, ranked_lists, rrf_k=self.rrf_k, dedup=self.dedup)
        return fused[: self.max_num_results]

    def _plain_system_blocks(self, utterance: str, session_id: str) -> List[AgentMessage]:
        """Collect leading SYSTEM/DEVELOPER messages contributed by plain members."""
        blocks: List[AgentMessage] = []
        for name, inst, _w in self.plain:
            try:
                ctx = inst.build_conversation_context(utterance, session_id)
            except Exception as e:
                LOG.error(f"CompositeMemory: member {name!r} build_context failed: {e}")
                continue
            for msg in ctx:
                if msg.role in (MessageRole.SYSTEM, MessageRole.DEVELOPER):
                    blocks.append(msg)
                else:
                    break  # only the leading system block carries the augmentation
        return blocks

    def build_conversation_context(self, utterance: str, session_id: str) -> List[AgentMessage]:
        """Consolidate members into one augmented context (user utterance last)."""
        query = self._build_query(utterance, session_id)
        fused = self._gather(query, session_id)
        context = self._format_context(fused) if fused else ""
        extra_system = self._plain_system_blocks(utterance, session_id)
        history = self.get_history(session_id)
        return self.assemble_context(utterance, session_id, context, history,
                                     extra_system=extra_system)
