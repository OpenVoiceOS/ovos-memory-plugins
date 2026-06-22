# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Fully-local, in-process Retrieval-Augmented Generation memory for OVOS personas.

``LocalRAGMemory`` is a :class:`~ovos_memory_plugins.base.BaseRetrievalMemory`
that runs RAG entirely in-process: it loads an OVOS text-embeddings plugin and an
``EmbeddingsDB`` plugin directly (no HTTP, no cloud key). Every assistant exchange
is embedded and stored in the vector database; before each turn the most relevant
prior exchanges are retrieved and folded into the conversation context via a
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

Injection strategies, retrieval knobs, and history handling are documented on
:class:`~ovos_memory_plugins.base.BaseRetrievalMemory`. This class adds only the
embeddings + vector-store specifics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ovos_plugin_manager.embeddings import (
    load_text_embeddings_plugin,
    load_embeddings_db_plugin,
)
from ovos_utils.log import LOG

from ovos_memory_plugins.base import BaseRetrievalMemory
from ovos_memory_plugins.common import MemoryHit

DEFAULT_EMBEDDINGS_PLUGIN = "ovos-gguf-embeddings-plugin"
DEFAULT_EMBEDDINGS_DB_PLUGIN = "ovos-chromadb-embeddings-plugin"


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


class LocalRAGMemory(BaseRetrievalMemory):
    """In-process semantic RAG memory backed by local embeddings + vector store.

    Loads a text-embeddings plugin and an ``EmbeddingsDB`` plugin via
    ``ovos_plugin_manager`` and uses them directly — no network round-trip, so it
    works fully offline. Stores each exchange as a retrievable document and folds
    the top-k relevant prior exchanges into context per ``inject_mode``.

    Embeddings-specific configuration keys
    --------------------------------------
    embeddings_plugin : str  (default ``"ovos-gguf-embeddings-plugin"``)
        Entry-point name of the ``opm.embeddings.text`` plugin.
    embeddings_config : dict
        Config forwarded to the text-embeddings plugin constructor.
    embeddings_db_plugin : str  (default ``"ovos-chromadb-embeddings-plugin"``)
        Entry-point name of the ``opm.embeddings`` (``EmbeddingsDB``) plugin.
    embeddings_db_config : dict
        Config forwarded to the EmbeddingsDB plugin constructor (e.g. ``path``).
    collection : str  (default ``"ovos_local_rag"``)
        Collection / vector-store name inside the EmbeddingsDB.

    See :class:`~ovos_memory_plugins.base.BaseRetrievalMemory` for the shared
    ``retrieval`` / ``context`` / ``inject_mode`` / ``max_history`` keys.
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

        # injectable backends (also makes unit testing with stubs trivial)
        self._embedder = self.config.get("_embedder")
        self._db = self.config.get("_db")

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

    # ----------------------------------------------------------- storage hooks
    def _store_document(self, doc_id: str, text: str, session_id: str,
                        metadata: Dict[str, Any]) -> None:
        """Embed and persist a single document in the vector store."""
        embedding = self._embed(text)
        self.db.add_embeddings(
            doc_id, embedding, metadata=metadata, collection_name=self.collection)

    def _query_backend(self, query: str, top_k: int) -> List[MemoryHit]:
        """Embed ``query`` and return the top-k nearest documents as hits.

        EmbeddingsDB query results carry a distance (lower = closer); for the
        cosine space used by the reference stack ``score = 1 - distance``.
        """
        query_vec = self._embed(query)
        raw = self.db.query(
            query_vec, top_k=top_k, return_metadata=True, collection_name=self.collection)

        hits: List[MemoryHit] = []
        for item in raw:
            key, distance = item[0], item[1]
            metadata = item[2] if len(item) > 2 else {}
            content = (metadata or {}).get("content")
            if not content:
                continue
            hits.append(MemoryHit(
                content=content, source=key, score=1.0 - float(distance),
                metadata={"session_id": (metadata or {}).get("session_id")}))
        return hits

    # ----------------------------------------------------------- back-compat
    def _search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, str, float]]:
        """Legacy tuple-shaped search: ``[(content, source_id, score), ...]``.

        Retained for callers/tests predating the :class:`MemoryHit` API; new code
        should use :meth:`search`.
        """
        return [(h.content, h.source, h.score) for h in self.search(query, top_k=top_k)]
