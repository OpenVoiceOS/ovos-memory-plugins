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
OpenAI-compatible endpoint to extract short facts, as a JSON array, from the
latest turn, then merges them (deduplicated) into a per-session fact store.
Local-first — it talks to the same kind of endpoint
:class:`~ovos_memory_plugins.longterm.LongTermMemory` uses, never a hosted
provider. If the endpoint is unreachable, extraction simply no-ops and the turn
proceeds.

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

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import (
    AgentContextManager,
    AgentMessage,
    MessageRole,
)
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home

from ovos_memory_plugins._llm import chat_complete, resolve_model
from ovos_memory_plugins.persistence import JsonStore

DEFAULT_EXTRACTION_PROMPT = (
    "Exchange:\n{exchange}\n\n"
    "List the durable facts the USER stated about themselves (name, "
    "preferences, relationships, location, constraints, goals), as a JSON array "
    "of short strings in the user's language, for example "
    '["Name is Bob", "Likes jazz"]. Output only the array. '
    "Use [] when the user stated no such fact."
)
DEFAULT_FACTS_HEADER = "Known facts about the user:"
# Seconds. Same as the model auto-detection timeout in ``_llm.resolve_model``.
DEFAULT_REQUEST_TIMEOUT = 10
# Only this many leading characters of a reply are scanned for JSON arrays.
MAX_SCANNED_REPLY_CHARS = 16384
# Scanning stops after this many "[" that start no valid JSON array.
MAX_FAILED_DECODES = 64


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
    request_timeout : int  (default 10)
        HTTP timeout in seconds for the extraction request. Extraction runs
        before the turn is reported as handled, so a slow server delays the
        end of the turn. Each request waits at most this long, and the
        timeout applies to each read, so a server that sends data slowly can
        take longer. While ``model`` is unset, model detection can add up to
        10 s to a turn. When a request fails, the facts of that exchange are
        not stored and a warning names this key. A CPU-only host needs a
        higher value.
    backend : str  (default ``"json"``)
        ``"json"`` (persisted) or ``"memory"`` (ephemeral).
    db_path : str  (default ``$XDG_DATA_HOME/ovos/entity_memory.json``)
        JSON file path for the ``json`` backend. ``XDG_DATA_HOME`` falls back
        to ``~/.local/share``.
    extraction_prompt : str
        Override the extraction prompt template (must contain ``{exchange}``).
        The model must reply with a JSON array of fact strings (``[]`` for none);
        any other reply stores nothing.
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
        self.request_timeout: int = int(self.config.get("request_timeout", DEFAULT_REQUEST_TIMEOUT))
        self.extraction_prompt: str = self.config.get(
            "extraction_prompt", DEFAULT_EXTRACTION_PROMPT)
        self.facts_header: str = self.config.get("facts_header", DEFAULT_FACTS_HEADER)

        backend = self.config.get("backend", "json").lower()
        if backend == "json":
            db_path = self.config.get(
                "db_path", str(xdg_data_home() / "ovos" / "entity_memory.json"))
            self._store: Optional[JsonStore] = JsonStore(db_path)
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
    def _json_arrays(text: str) -> List[list]:
        """Return every top-level JSON array found in ``text``, in order."""
        text = (text or "")[:MAX_SCANNED_REPLY_CHARS]
        decoder = json.JSONDecoder()
        arrays: List[list] = []
        pos = 0
        failed_decodes = 0
        while (start := text.find("[", pos)) != -1:
            try:
                data, pos = decoder.raw_decode(text, start)
            except (RecursionError, ValueError):
                failed_decodes += 1
                if failed_decodes >= MAX_FAILED_DECODES:
                    break
                # skip a run of opening brackets, keeping its last one as a start
                pos = max(start + 1, len(text) - len(text[start:].lstrip("[")) - 1)
                continue
            arrays.append(data)
        return arrays

    @classmethod
    def _parse_facts(cls, text: str) -> List[str]:
        """Return the fact strings of every JSON array in the LLM response.

        Anything outside the arrays (commentary, code fences) is ignored, and a
        reply without a JSON array yields no facts, so prose is never stored.
        Items without a letter or digit are dropped, and repeats are removed.
        """
        facts: List[str] = []
        for array in cls._json_arrays(text):
            for item in array:
                if isinstance(item, str) and any(c.isalnum() for c in item):
                    fact = item.strip()
                    if fact not in facts:
                        facts.append(fact)
        return facts

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
            LOG.warning(f"EntityMemory: extraction skipped, so no fact was stored "
                        f"({exc}). Raise request_timeout if the server is slow.")
            return
        if reply.strip() and not self._json_arrays(reply):
            LOG.warning("EntityMemory: extraction reply holds no JSON array, so no "
                        "fact was stored. A custom extraction_prompt must ask for "
                        "a JSON array of strings.")
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

    # ---------------------------------------------------------------- lifecycle
    def close(self) -> None:
        """Release the persistence backend, if any."""
        if self._store is not None and hasattr(self._store, "close"):
            self._store.close()

    def __enter__(self) -> "EntityMemory":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
