# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Shared base class for *retrieval* memory plugins.

A retrieval memory stores each conversation exchange and, before every turn,
recalls the most relevant prior exchanges and folds them into the context. This
base factors out everything that is independent of *how* documents are stored
and searched — short-term history, deterministic ids, the query builder, the
context renderer, and the five ``inject_mode`` strategies — so a concrete
backend only implements two hooks:

- :meth:`_store_document` — persist one document.
- :meth:`_query_backend` — return ranked :class:`~ovos_memory_plugins.common.MemoryHit`.

Concrete subclasses: :class:`~ovos_memory_plugins.local_rag.LocalRAGMemory`
(semantic, embeddings + vector DB) and
:class:`~ovos_memory_plugins.lexical.LexicalMemory` (keyword, SQLite FTS5). Both
expose the public ``search()`` that lets
:class:`~ovos_memory_plugins.composite.CompositeMemory` fuse them.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import (
    AgentContextManager,
    AgentMessage,
    MessageRole,
    ToolCall,
)
from ovos_utils.log import LOG

from ovos_memory_plugins.common import MemoryHit

DEFAULT_HEADER = (
    "Use the following recalled context to answer the user's question. "
    "If the answer is not in the context, rely on the conversation instead."
)
# {system}, {header} and {context} for inject_mode="system_prompt"
DEFAULT_SYSTEM_PROMPT_TEMPLATE = "{system}\n\n{header}\n\nContext:\n{context}"
# {header}, {context} and {utterance} for inject_mode="user"
DEFAULT_USER_TEMPLATE = "{header}\n\nContext:\n{context}\n\nQuestion: {utterance}"

_VALID_MODES = {"system", "developer", "system_prompt", "user", "tool"}


class BaseRetrievalMemory(AgentContextManager, abc.ABC):
    """Abstract retrieval memory: history + recall + injection, storage-agnostic.

    Shared configuration keys
    -------------------------
    retrieval : dict
        ``max_num_results`` (int, default 5), ``min_score`` (float|None),
        ``query_mode`` (``"utterance"``|``"history"``), ``query_history_turns`` (int).
    context : dict
        ``header``, ``chunk_prefix``, ``chunk_separator``, ``include_sources``,
        ``tool_name`` — control how recalled chunks are rendered.
    inject_mode : str  (default ``"system"``)
        One of ``system``, ``developer``, ``system_prompt``, ``user``, ``tool``.
    system_prompt : str
        Optional persona system prompt.
    system_prompt_template / user_template : str
        Templates for the ``system_prompt`` / ``user`` inject modes.
    max_history : int  (default 10)
        Recent verbatim messages retained per session.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.max_history: int = int(self.config.get("max_history", 10))

        retrieval = self.config.get("retrieval", {})
        self.max_num_results: int = int(retrieval.get("max_num_results", 5))
        self.min_score: Optional[float] = retrieval.get("min_score")
        self.query_mode: str = retrieval.get("query_mode", "utterance")
        self.query_history_turns: int = int(retrieval.get("query_history_turns", 3))

        ctx = self.config.get("context", {})
        self.header: str = ctx.get("header", DEFAULT_HEADER)
        self.chunk_prefix: str = ctx.get("chunk_prefix", "- ")
        self.chunk_separator: str = ctx.get("chunk_separator", "\n\n")
        self.include_sources: bool = bool(ctx.get("include_sources", False))
        self.tool_name: str = ctx.get("tool_name", "search_memory")

        self.inject_mode: str = self.config.get("inject_mode", "system")
        self.system_prompt_template: str = self.config.get(
            "system_prompt_template", DEFAULT_SYSTEM_PROMPT_TEMPLATE)
        self.user_template: str = self.config.get("user_template", DEFAULT_USER_TEMPLATE)

        if self.inject_mode not in _VALID_MODES:
            raise ValueError(f"inject_mode must be one of {sorted(_VALID_MODES)}, "
                             f"got {self.inject_mode!r}")

        self.session2history: Dict[str, List[AgentMessage]] = {}
        # monotonic per-session counters → stable, deterministic ids
        self._doc_counter: Dict[str, int] = {}
        self._tool_call_counter: Dict[str, int] = {}

    # ----------------------------------------------------------- storage hooks
    @abc.abstractmethod
    def _store_document(self, doc_id: str, text: str, session_id: str,
                        metadata: Dict[str, Any]) -> None:
        """Persist one document. ``metadata`` carries ``content`` and ``session_id``."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _query_backend(self, query: str, top_k: int) -> List[MemoryHit]:
        """Return up to ``top_k`` ranked hits for ``query`` (no min_score filtering)."""
        raise NotImplementedError()

    # ------------------------------------------------------------------ history
    def get_history(self, session_id: str) -> List[AgentMessage]:
        """Return the retained short-term history for ``session_id``."""
        return list(self.session2history.get(session_id, []))

    def _next_doc_id(self, session_id: str) -> str:
        """Return a stable, deterministic document id for ``session_id``."""
        n = self._doc_counter.get(session_id, 0)
        self._doc_counter[session_id] = n + 1
        return f"{session_id}_{n}"

    def store(self, text: str, session_id: str,
              metadata: Optional[Dict[str, Any]] = None) -> None:
        """Embed/persist one document via the backend, swallowing failures.

        A storage error must never break the conversation, so backend exceptions
        are logged and dropped.
        """
        text = (text or "").strip()
        if not text:
            return
        meta = {"content": text, "session_id": session_id, **(metadata or {})}
        try:
            self._store_document(self._next_doc_id(session_id), text, session_id, meta)
        except Exception as e:
            LOG.error(f"{type(self).__name__}: failed to store document: {e}")

    def update_history(self, new_messages: List[AgentMessage], session_id: str) -> None:
        """Append ``new_messages`` to history and persist completed exchanges.

        Each assistant message paired with the preceding user message is stored as
        a retrievable ``"Q: ...\\nA: ..."`` document. History is truncated to
        ``max_history``.
        """
        history = self.session2history.setdefault(session_id, [])
        for msg in new_messages:
            if msg.role == MessageRole.ASSISTANT and msg.content:
                prior_user = next(
                    (m.content for m in reversed(history) if m.role == MessageRole.USER), "")
                doc = f"Q: {prior_user}\nA: {msg.content}".strip()
                self.store(doc, session_id)
            history.append(msg)
        if self.max_history:
            self.session2history[session_id] = history[-self.max_history:]

    # ---------------------------------------------------------------- retrieval
    def _build_query(self, utterance: str, session_id: str) -> str:
        """Build the search query per ``query_mode`` (optionally folding in history)."""
        if self.query_mode != "history" or not self.query_history_turns:
            return utterance
        prior_user = [m.content for m in self.get_history(session_id)
                      if m.role == MessageRole.USER][-self.query_history_turns:]
        return " ".join([*prior_user, utterance]).strip()

    def search(self, query: str, session_id: Optional[str] = None,
               top_k: Optional[int] = None) -> List[MemoryHit]:
        """Return ranked hits for ``query``, filtered by ``min_score``.

        ``top_k`` defaults to ``max_num_results``. Backend errors degrade to an
        empty list (the turn proceeds without recalled context).
        """
        k = top_k if top_k is not None else self.max_num_results
        try:
            hits = self._query_backend(query, k)
        except Exception as e:
            LOG.error(f"{type(self).__name__}: search failed ({e}); proceeding without context")
            return []
        if self.min_score is not None:
            hits = [h for h in hits if h.score >= self.min_score]
        return hits[:k]

    def _format_context(self, hits: List[MemoryHit]) -> str:
        """Render retrieved hits into a single context block string."""
        lines = []
        for hit in hits:
            prefix = self.chunk_prefix
            if self.include_sources and hit.source:
                prefix = f"{self.chunk_prefix}[{hit.source}] "
            lines.append(f"{prefix}{hit.content}")
        return self.chunk_separator.join(lines)

    def _next_tool_call_id(self, session_id: str) -> str:
        """Return a stable, deterministic tool-call id for ``session_id``."""
        n = self._tool_call_counter.get(session_id, 0)
        self._tool_call_counter[session_id] = n + 1
        return f"memrag_{session_id}_{n}"

    # ------------------------------------------------------------------ context
    def build_conversation_context(self, utterance: str, session_id: str) -> List[AgentMessage]:
        """Assemble the augmented context per the configured inject strategy.

        The first message MAY be a system message (``self.system_prompt``); the
        LAST message is always the user utterance (AgentContextManager contract).
        """
        hits = self.search(self._build_query(utterance, session_id), session_id=session_id)
        context = self._format_context(hits) if hits else ""
        history = self.get_history(session_id)
        return self.assemble_context(utterance, session_id, context, history)

    def assemble_context(self, utterance: str, session_id: str, context: str,
                         history: List[AgentMessage],
                         extra_system: Optional[List[AgentMessage]] = None) -> List[AgentMessage]:
        """Build the message list for a formatted ``context`` block per ``inject_mode``.

        Shared by concrete retrievers and reused by
        :class:`~ovos_memory_plugins.composite.CompositeMemory` (which passes a
        pre-fused context block and ``extra_system`` contributions from non-retriever
        members). ``extra_system`` messages are inserted right after the base system
        prompt and before the recalled-context block.
        """
        base_system = self.system_prompt
        messages: List[AgentMessage] = []

        if context and self.inject_mode == "system_prompt":
            combined = self.system_prompt_template.format(
                system=base_system, header=self.header, context=context).strip()
            messages.append(AgentMessage(role=MessageRole.SYSTEM, content=combined))
            messages.extend(extra_system or [])
        else:
            if base_system:
                messages.append(AgentMessage(role=MessageRole.SYSTEM, content=base_system))
            messages.extend(extra_system or [])
            if context and self.inject_mode in ("system", "developer"):
                role = MessageRole.DEVELOPER if self.inject_mode == "developer" else MessageRole.SYSTEM
                block = f"{self.header}\n\nContext:\n{context}"
                messages.append(AgentMessage(role=role, content=block))

        # the current utterance is appended last; drop any dangling trailing user
        # turn so it isn't duplicated, or left unanswered before a synthetic tool turn
        while history and history[-1].role == MessageRole.USER:
            history = history[:-1]
        messages.extend(history)

        # inject_mode="tool": a synthetic search tool-call + result, just before the
        # user turn. Assistant-with-tool_calls precedes its TOOL result (provider
        # ordering invariant); the user utterance stays last (context-manager contract).
        if context and self.inject_mode == "tool":
            call_id = self._next_tool_call_id(session_id)
            messages.append(AgentMessage(
                role=MessageRole.ASSISTANT, content="",
                tool_calls=[ToolCall(id=call_id, name=self.tool_name,
                                     arguments={"query": utterance.strip()})]))
            messages.append(AgentMessage(
                role=MessageRole.TOOL, content=context,
                tool_call_id=call_id, name=self.tool_name))

        if context and self.inject_mode == "user":
            content = self.user_template.format(
                header=self.header, context=context, utterance=utterance.strip())
            messages.append(AgentMessage(role=MessageRole.USER, content=content))
        else:
            messages.append(AgentMessage(role=MessageRole.USER, content=utterance.strip()))

        return messages
