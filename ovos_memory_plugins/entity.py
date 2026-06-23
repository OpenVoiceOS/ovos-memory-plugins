# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Durable entity/fact memory for OVOS personas.

``EntityMemory`` distills *durable* facts about the user from the conversation —
name, preferences, relationships, constraints — and re-injects them into every
turn, so they are recalled regardless of how long ago they were said (unlike a
recency window) and without semantic search (unlike RAG).

It is "clever by chaining a prompt": after each exchange it asks a **local**
OpenAI-compatible endpoint to extract short fact lines from the latest turn, then
merges them (deduplicated) into a per-session fact store. Local-first — it talks
to the same kind of endpoint :class:`~ovos_memory_plugins.longterm.LongTermMemory`
uses, never a hosted provider. If the endpoint is unreachable, extraction simply
no-ops and the turn proceeds.

Configuration (persona JSON block)::

    {
      "memory_module": "ovos-memory-plugin-entity",
      "ovos-memory-plugin-entity": {
        "api_url": "http://localhost:8000/v1",
        "model": "",                 # auto-detected from /models when empty
        "max_facts": 50,
        "backend": "json",           # "json" | "memory"
        "db_path": "~/.local/share/ovos/entity_memory.json",
        "system_prompt": "You are a helpful assistant."
      }
    }
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import (
    AgentContextManager,
    AgentMessage,
    MessageRole,
)
from ovos_utils.log import LOG

from ovos_memory_plugins._llm import chat_complete, resolve_model
from ovos_memory_plugins.longterm import _JsonStore

DEFAULT_EXTRACTION_PROMPT = (
    "From the exchange below, extract durable facts about the USER worth "
    "remembering long-term (name, preferences, relationships, location, "
    "constraints, goals). Output each fact on its own line prefixed with '- '. "
    "Ignore one-off/transient details and small talk. If there is nothing worth "
    "remembering, output exactly NONE.\n\nExchange:\n{exchange}"
)
DEFAULT_FACTS_HEADER = "Known facts about the user:"


class EntityMemory(AgentContextManager):
    """Extracts and recalls durable user facts via a chained LLM prompt.

    Configuration keys
    ------------------
    api_url : str
        Base URL of the OpenAI-compatible server (``.../v1``).
    model : str
        Model name; auto-detected from ``/models`` when empty.
    max_facts : int  (default 50)
        Cap on stored facts per session (oldest dropped past the cap).
    max_extract_tokens : int  (default 128)
        ``max_tokens`` for the extraction request.
    request_timeout : int  (default 30)
        HTTP timeout in seconds.
    backend : str  (default ``"json"``)
        ``"json"`` (persisted) or ``"memory"`` (ephemeral).
    db_path : str
        JSON file path for the ``json`` backend.
    extraction_prompt : str
        Override the extraction prompt template (must contain ``{exchange}``).
    facts_header : str
        Heading for the injected facts block.
    system_prompt : str
        Optional persona system prompt.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_url: str = self.config.get("api_url", "http://localhost:8000/v1")
        self.model: str = self.config.get("model", "")
        self.max_facts: int = int(self.config.get("max_facts", 50))
        self.max_extract_tokens: int = int(self.config.get("max_extract_tokens", 128))
        self.request_timeout: int = int(self.config.get("request_timeout", 30))
        self.extraction_prompt: str = self.config.get(
            "extraction_prompt", DEFAULT_EXTRACTION_PROMPT)
        self.facts_header: str = self.config.get("facts_header", DEFAULT_FACTS_HEADER)

        backend = self.config.get("backend", "json").lower()
        if backend == "json":
            db_path = self.config.get("db_path", "~/.local/share/ovos/entity_memory.json")
            self._store: Optional[_JsonStore] = _JsonStore(db_path)
        else:
            self._store = None
        # session_id → list[str] facts (in-memory cache)
        self._facts: Dict[str, List[str]] = {}
        # model is resolved lazily on first extraction — no network I/O at construction

    # ------------------------------------------------------------------ facts
    def _load_facts(self, session_id: str) -> List[str]:
        if session_id not in self._facts:
            facts: List[str] = []
            if self._store is not None:
                facts = list((self._store.load(session_id) or {}).get("facts", []))
            self._facts[session_id] = facts
        return self._facts[session_id]

    def _persist_facts(self, session_id: str) -> None:
        if self._store is not None:
            self._store.save(session_id, {"facts": self._facts.get(session_id, [])})

    def _merge_facts(self, session_id: str, new_facts: List[str]) -> None:
        """Add new facts, deduplicating case-insensitively, capped at ``max_facts``."""
        facts = self._load_facts(session_id)
        seen = {f.lower() for f in facts}
        for fact in new_facts:
            fact = fact.strip()
            if fact and fact.lower() not in seen:
                facts.append(fact)
                seen.add(fact.lower())
        if self.max_facts and len(facts) > self.max_facts:
            del facts[: len(facts) - self.max_facts]
        self._facts[session_id] = facts
        self._persist_facts(session_id)

    @staticmethod
    def _parse_facts(text: str) -> List[str]:
        """Parse '- fact' lines from the LLM response; ``NONE`` yields nothing."""
        if not text or text.strip().upper() == "NONE":
            return []
        out = []
        for line in text.splitlines():
            # strip a leading bullet ("- ", "* ", "• ") or enumerator ("1.", "2)")
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if line and line.upper() != "NONE":
                out.append(line)
        return out

    def _extract(self, exchange: str, session_id: str) -> None:
        """Ask the LLM for durable facts in ``exchange`` and merge them; never raises."""
        if not self.model:
            self.model = resolve_model(self.api_url)
        prompt = self.extraction_prompt.format(exchange=exchange)
        try:
            reply = chat_complete(
                api_url=self.api_url, model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_extract_tokens, timeout=self.request_timeout)
        except Exception as exc:
            LOG.debug(f"EntityMemory: extraction skipped ({exc})")
            return
        new_facts = self._parse_facts(reply)
        if new_facts:
            self._merge_facts(session_id, new_facts)

    # ----------------------------------------------- AgentContextManager API
    def get_history(self, session_id: str) -> List[AgentMessage]:
        """Entity memory holds facts, not turns — history is always empty."""
        return []

    def update_history(self, new_messages: List[AgentMessage], session_id: str) -> None:
        """Extract durable facts from the latest user→assistant exchange."""
        last_user = ""
        for msg in new_messages:
            if msg.role == MessageRole.USER and msg.content:
                last_user = msg.content
            elif msg.role == MessageRole.ASSISTANT and msg.content:
                exchange = f"User: {last_user}\nAssistant: {msg.content}".strip()
                self._extract(exchange, session_id)

    def build_conversation_context(self, utterance: str, session_id: str) -> List[AgentMessage]:
        """Return ``[system?] + [facts system block?] + [user utterance]``."""
        messages: List[AgentMessage] = []
        if self.system_prompt.strip():
            messages.append(AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt.strip()))
        facts = self._load_facts(session_id)
        if facts:
            block = self.facts_header + "\n" + "\n".join(f"- {f}" for f in facts)
            messages.append(AgentMessage(role=MessageRole.SYSTEM, content=block))
        messages.append(AgentMessage(role=MessageRole.USER, content=utterance.strip()))
        return messages
