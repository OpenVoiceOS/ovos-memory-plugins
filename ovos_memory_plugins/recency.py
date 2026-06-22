# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Short-term recency / buffer memory for OVOS personas.

``RecencyMemory`` keeps a sliding window of the most recent turns — bounded by a
message count and, optionally, by age (time decay). It is the lightest possible
memory: no LLM, no embeddings, **no extra dependencies**, fully local. Use it as
a persona's ``memory_module`` for plain short-term context, or as the ``primary``
(history-providing) member inside
:class:`~ovos_memory_plugins.composite.CompositeMemory` alongside heavier recall
backends.

Configuration (persona JSON block)::

    {
      "memory_module": "ovos-memory-plugin-recency",
      "ovos-memory-plugin-recency": {
        "max_history": 10,          # messages kept verbatim (0 = unbounded)
        "max_age": 1800,            # optional: drop messages older than N seconds
        "system_prompt": "You are a helpful assistant.",
        "db_path": null             # optional JSON file for cross-restart persistence
      }
    }
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from ovos_plugin_manager.templates.agents import (
    AgentContextManager,
    AgentMessage,
    MessageRole,
)

from ovos_memory_plugins.longterm import _JsonStore


class RecencyMemory(AgentContextManager):
    """A sliding window of recent turns, bounded by count and optional age.

    Configuration keys
    ------------------
    max_history : int  (default 10)
        Number of recent messages kept verbatim. ``0`` means unbounded.
    max_age : float|None  (default None)
        If set, messages older than this many seconds are dropped on access.
    system_prompt : str
        Optional persona system prompt prepended to the context.
    db_path : str|None
        Optional JSON file path; when set, history persists across restarts.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_history: int = int(self.config.get("max_history", 10))
        self.max_age: Optional[float] = self.config.get("max_age")
        db_path = self.config.get("db_path")
        self._store: Optional[_JsonStore] = _JsonStore(db_path) if db_path else None
        # session_id → list of (timestamp, AgentMessage)
        self._buffers: Dict[str, List[Tuple[float, AgentMessage]]] = {}

    # ------------------------------------------------------------------ helpers
    def _load(self, session_id: str) -> List[Tuple[float, AgentMessage]]:
        if session_id not in self._buffers:
            buf: List[Tuple[float, AgentMessage]] = []
            if self._store is not None:
                for item in (self._store.load(session_id) or {}).get("buffer", []):
                    buf.append((item.get("ts", 0.0),
                                AgentMessage(role=MessageRole(item["role"]),
                                             content=item["content"])))
            self._buffers[session_id] = buf
        return self._buffers[session_id]

    def _persist(self, session_id: str) -> None:
        if self._store is None:
            return
        buf = self._buffers.get(session_id, [])
        self._store.save(session_id, {"buffer": [
            {"ts": ts, "role": m.role.value if hasattr(m.role, "value") else str(m.role),
             "content": m.content} for ts, m in buf]})

    def _prune(self, session_id: str) -> List[Tuple[float, AgentMessage]]:
        """Apply age + count bounds and return the live buffer."""
        buf = self._load(session_id)
        if self.max_age is not None:
            cutoff = time.time() - self.max_age
            buf = [(ts, m) for ts, m in buf if ts >= cutoff]
        if self.max_history:
            buf = buf[-self.max_history:]
        self._buffers[session_id] = buf
        return buf

    # ----------------------------------------------- AgentContextManager API
    def get_history(self, session_id: str) -> List[AgentMessage]:
        """Return the live window of recent messages (age/count pruned)."""
        return [m for _ts, m in self._prune(session_id)]

    def update_history(self, new_messages: List[AgentMessage], session_id: str) -> None:
        """Append ``new_messages`` (timestamped now) and prune the window."""
        buf = self._load(session_id)
        now = time.time()
        for msg in new_messages:
            buf.append((now, msg))
        self._prune(session_id)
        self._persist(session_id)

    def build_conversation_context(self, utterance: str, session_id: str) -> List[AgentMessage]:
        """Return ``[system?] + recent window + [user utterance]`` (user last)."""
        messages: List[AgentMessage] = []
        if self.system_prompt.strip():
            messages.append(AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt.strip()))
        history = self.get_history(session_id)
        # drop trailing unmatched user turns so the new utterance is the only tail user
        while history and history[-1].role == MessageRole.USER:
            history.pop()
        messages.extend(history)
        messages.append(AgentMessage(role=MessageRole.USER, content=utterance.strip()))
        return messages
