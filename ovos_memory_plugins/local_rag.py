# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Fully-local, in-process Retrieval-Augmented Generation memory for OVOS personas.

``LocalRAGMemory`` is an :class:`AgentContextManager` that runs RAG entirely
in-process: it loads an OVOS text-embeddings plugin and an ``EmbeddingsDB``
plugin directly (no HTTP, no cloud key). Every assistant exchange is embedded
and stored in the vector database; before each turn the most relevant prior
exchanges are retrieved and folded into the conversation context via a
configurable ``inject_mode``. The persona's chat engine then generates the
answer — RAG composes with any chat backend instead of owning the round-trip.

A known-good fully-offline stack is ``ovos-gguf-embeddings-plugin`` (text
embeddings) + ``ovos-chromadb-embeddings-plugin`` (vector store); install the
``local-rag`` extra to pull both.

Configuration (persona JSON block, keyed by the plugin name)::

    {
      "name": "kb-assistant",
      "solvers": ["ovos-gguf-chat-plugin"],
      "memory_module": "ovos-memory-plugin-local-rag",
      "ovos-memory-plugin-local-rag": {
        "embeddings_plugin": "ovos-gguf-embeddings-plugin",
        "embeddings_config": {"model": "labse"},
        "embeddings_db_plugin": "ovos-chromadb-embeddings-plugin",
        "embeddings_db_config": {"path": "~/.local/share/ovos/local_rag_db"},
        "collection": "ovos_local_rag",

        "retrieval": {
          "max_num_results": 5,
          "min_score": null,            # drop hits below this score (null = keep all)
          "query_mode": "utterance",    # "utterance" | "history"
          "query_history_turns": 3      # turns folded into the query when query_mode="history"
        },

        "context": {
          "header": "Use the following context to answer ...",
          "chunk_prefix": "- ",
          "chunk_separator": "\\n\\n",
          "include_sources": false,      # prefix each chunk with its stored id
          "tool_name": "search_memory"   # name used for inject_mode="tool"
        },

        "inject_mode": "system",        # system | system_prompt | developer | user | tool
        "system_prompt": "You are a helpful assistant.",
        "max_history": 10
      }
    }

Injection strategies (``inject_mode``):

- ``system`` (default) — keep the persona's ``system_prompt`` as its own message
  and add the retrieved context as a **separate** system message before the user
  turn. Keeps the base system prompt stable/cacheable.
- ``developer`` — same, but the context goes in a ``developer``-role message.
- ``system_prompt`` — fold context into the persona's system prompt (one combined
  system message) via ``system_prompt_template``.
- ``user`` — prepend the context to the final user message via ``user_template``.
- ``tool`` — present the context as a tool-call result: a synthetic assistant
  ``tool_calls`` turn (a ``search_memory`` call for the query) followed by a
  ``MessageRole.TOOL`` message carrying the chunks, just before the user
  utterance. Requires a brain/contract with tool-call support
  (``ovos-plugin-manager`` TOOL role).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ovos_plugin_manager.embeddings import (
    load_text_embeddings_plugin,
    load_embeddings_db_plugin,
)
from ovos_plugin_manager.templates.agents import (
    AgentContextManager,
    AgentMessage,
    MessageRole,
    ToolCall,
)
from ovos_utils.log import LOG

DEFAULT_HEADER = (
    "Use the following recalled context to answer the user's question. "
    "If the answer is not in the context, rely on the conversation instead."
)
# {system} and {context} for inject_mode="system_prompt"
DEFAULT_SYSTEM_PROMPT_TEMPLATE = "{system}\n\n{header}\n\nContext:\n{context}"
# {context} and {utterance} for inject_mode="user"
DEFAULT_USER_TEMPLATE = "{header}\n\nContext:\n{context}\n\nQuestion: {utterance}"

DEFAULT_EMBEDDINGS_PLUGIN = "ovos-gguf-embeddings-plugin"
DEFAULT_EMBEDDINGS_DB_PLUGIN = "ovos-chromadb-embeddings-plugin"

_VALID_MODES = {"system", "developer", "system_prompt", "user", "tool"}


def _load_embeddings_db(plugin_name: str, config: Dict[str, Any]):
    """Instantiate an EmbeddingsDB plugin.

    EmbeddingsDB plugins have heterogeneous constructors: the base template and
    ovos-qdrant take ``config=``, while ovos-chromadb takes a positional
    ``path``. Adapt to whichever the plugin accepts (mirrors ovos-persona-server).
    """
    import inspect

    db_cls = load_embeddings_db_plugin(plugin_name)
    init_params = inspect.signature(db_cls.__init__).parameters
    if "config" in init_params:
        return db_cls(config=config)
    if "path" in init_params and config.get("path"):
        return db_cls(path=config["path"])
    return db_cls()


class LocalRAGMemory(AgentContextManager):
    """In-process RAG persona memory backed by local embeddings + vector store.

    Loads a text-embeddings plugin and an ``EmbeddingsDB`` plugin via
    ``ovos_plugin_manager`` and uses them directly — no network round-trip, so it
    works fully offline. Stores each exchange as a retrievable document and folds
    the top-k relevant prior exchanges into context per ``inject_mode``.

    Configuration keys
    ------------------
    embeddings_plugin : str  (default: ``"ovos-gguf-embeddings-plugin"``)
        Entry-point name of the ``opm.embeddings.text`` plugin.
    embeddings_config : dict
        Config forwarded to the text-embeddings plugin constructor.
    embeddings_db_plugin : str  (default: ``"ovos-chromadb-embeddings-plugin"``)
        Entry-point name of the ``opm.embeddings`` (``EmbeddingsDB``) plugin.
    embeddings_db_config : dict
        Config forwarded to the EmbeddingsDB plugin constructor (e.g. ``path``).
    collection : str  (default: ``"ovos_local_rag"``)
        Collection / vector-store name inside the EmbeddingsDB.
    retrieval : dict
        ``max_num_results`` (int, default 5), ``min_score`` (float|None),
        ``query_mode`` (``"utterance"``|``"history"``), ``query_history_turns`` (int).
    context : dict
        ``header``, ``chunk_prefix``, ``chunk_separator``, ``include_sources``,
        ``tool_name`` — control how retrieved chunks are rendered.
    inject_mode : str  (default: ``"system"``)
        One of ``system``, ``developer``, ``system_prompt``, ``user``, ``tool``.
    system_prompt : str
        Optional persona system prompt.
    system_prompt_template / user_template : str
        Templates for the ``system_prompt`` / ``user`` inject modes.
    max_history : int  (default: 10)
        Recent verbatim messages retained per session.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.embeddings_plugin: str = self.config.get(
            "embeddings_plugin", DEFAULT_EMBEDDINGS_PLUGIN)
        self.embeddings_config: Dict[str, Any] = self.config.get("embeddings_config", {})
        self.embeddings_db_plugin: str = self.config.get(
            "embeddings_db_plugin", DEFAULT_EMBEDDINGS_DB_PLUGIN)
        self.embeddings_db_config: Dict[str, Any] = dict(
            self.config.get("embeddings_db_config", {}))
        self.collection: str = self.config.get("collection", "ovos_local_rag")
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

        # injectable backends (also makes unit testing with stubs trivial)
        self._embedder = self.config.get("_embedder")
        self._db = self.config.get("_db")

        self.session2history: Dict[str, List[AgentMessage]] = {}
        # monotonic per-session counters → stable, deterministic ids
        self._doc_counter: Dict[str, int] = {}
        self._tool_call_counter: Dict[str, int] = {}

    # ------------------------------------------------------------------ backends
    @property
    def embedder(self):
        """The text-embeddings backend, loaded lazily on first use."""
        if self._embedder is None:
            plugin_cls = load_text_embeddings_plugin(self.embeddings_plugin)
            self._embedder = plugin_cls(self.embeddings_config)
            LOG.debug(f"LocalRAGMemory: loaded embeddings plugin {self.embeddings_plugin}")
        return self._embedder

    @property
    def db(self):
        """The EmbeddingsDB backend, loaded lazily on first use (collection ensured)."""
        if self._db is None:
            self._db = _load_embeddings_db(self.embeddings_db_plugin, self.embeddings_db_config)
            LOG.debug(f"LocalRAGMemory: loaded EmbeddingsDB plugin {self.embeddings_db_plugin}")
        try:
            self._db.create_collection(self.collection)
        except Exception as e:
            LOG.debug(f"LocalRAGMemory: create_collection({self.collection}) no-op: {e}")
        return self._db

    def _embed(self, text: str) -> List[float]:
        """Return the embedding for ``text`` as a plain ``list[float]``."""
        return [float(x) for x in self.embedder.get_embeddings(text)]

    # ------------------------------------------------------------------ history
    def get_history(self, session_id: str) -> List[AgentMessage]:
        """Return the retained short-term history for ``session_id``."""
        return list(self.session2history.get(session_id, []))

    def _next_doc_id(self, session_id: str) -> str:
        """Return a stable, deterministic document id for ``session_id``."""
        n = self._doc_counter.get(session_id, 0)
        self._doc_counter[session_id] = n + 1
        return f"{session_id}_{n}"

    def _store_exchange(self, user_text: str, assistant_text: str, session_id: str) -> None:
        """Embed and persist a single Q/A exchange in the vector store."""
        doc = f"Q: {user_text}\nA: {assistant_text}".strip()
        if not doc:
            return
        try:
            embedding = self._embed(doc)
            key = self._next_doc_id(session_id)
            self.db.add_embeddings(
                key, embedding,
                metadata={"content": doc, "session_id": session_id},
                collection_name=self.collection,
            )
        except Exception as e:
            LOG.error(f"LocalRAGMemory: failed to store exchange: {e}")

    def update_history(self, new_messages: List[AgentMessage], session_id: str) -> None:
        """Append ``new_messages`` to history and persist completed exchanges.

        Each assistant message paired with the preceding user message is embedded
        and stored as a retrievable document. History is truncated to
        ``max_history``.
        """
        history = self.session2history.setdefault(session_id, [])
        for msg in new_messages:
            if msg.role == MessageRole.ASSISTANT and msg.content:
                prior_user = next(
                    (m.content for m in reversed(history) if m.role == MessageRole.USER), "")
                self._store_exchange(prior_user, msg.content, session_id)
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

    def _search(self, query: str) -> List[Tuple[str, str, float]]:
        """Search the vector store; return ``(content, source_id, score)`` per hit.

        EmbeddingsDB query results carry a distance (lower = closer); for the
        cosine space used by the reference stack ``score = 1 - distance``.
        """
        try:
            query_vec = self._embed(query)
            raw = self.db.query(
                query_vec, top_k=self.max_num_results,
                return_metadata=True, collection_name=self.collection,
            )
        except Exception as e:
            LOG.error(f"LocalRAGMemory: search failed ({e}); proceeding without context")
            return []

        hits: List[Tuple[str, str, float]] = []
        for item in raw:
            key, distance = item[0], item[1]
            metadata = item[2] if len(item) > 2 else {}
            content = (metadata or {}).get("content")
            if not content:
                continue
            score = 1.0 - float(distance)
            if self.min_score is not None and score < self.min_score:
                continue
            hits.append((content, key, score))
        return hits

    def _format_context(self, hits: List[Tuple[str, str, float]]) -> str:
        """Render retrieved hits into a single context block string."""
        lines = []
        for content, source_id, _score in hits:
            prefix = self.chunk_prefix
            if self.include_sources and source_id:
                prefix = f"{self.chunk_prefix}[{source_id}] "
            lines.append(f"{prefix}{content}")
        return self.chunk_separator.join(lines)

    def _next_tool_call_id(self, session_id: str) -> str:
        """Return a stable, deterministic tool-call id for ``session_id``."""
        n = self._tool_call_counter.get(session_id, 0)
        self._tool_call_counter[session_id] = n + 1
        return f"memrag_{session_id}_{n}"

    # ------------------------------------------------------------------ context
    def build_conversation_context(self, utterance: str, session_id: str) -> List[AgentMessage]:
        """Assemble the augmented context per the configured strategies.

        The first message MAY be a system message (``self.system_prompt``); the
        LAST message is always the user utterance (AgentContextManager contract).
        """
        hits = self._search(self._build_query(utterance, session_id))
        context = self._format_context(hits) if hits else ""

        base_system = self.system_prompt
        messages: List[AgentMessage] = []

        if context and self.inject_mode == "system_prompt":
            combined = self.system_prompt_template.format(
                system=base_system, header=self.header, context=context).strip()
            messages.append(AgentMessage(role=MessageRole.SYSTEM, content=combined))
        else:
            if base_system:
                messages.append(AgentMessage(role=MessageRole.SYSTEM, content=base_system))
            if context and self.inject_mode in ("system", "developer"):
                role = MessageRole.DEVELOPER if self.inject_mode == "developer" else MessageRole.SYSTEM
                block = f"{self.header}\n\nContext:\n{context}"
                messages.append(AgentMessage(role=role, content=block))

        messages.extend(self.get_history(session_id))

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
