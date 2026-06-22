# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Fully-local keyword/lexical memory backed by SQLite FTS5.

``LexicalMemory`` is a :class:`~ovos_memory_plugins.base.BaseRetrievalMemory`
that recalls prior exchanges by **keyword** match rather than semantics. It uses
SQLite's built-in FTS5 full-text index with its ``bm25()`` ranking function — so
it needs **no extra dependencies** (``sqlite3`` is stdlib and FTS5 ships with the
CPython-bundled SQLite) and persists to a single ``.db`` file.

On its own it is a lightweight long-term recall backend. Its real value is as the
*complement* to :class:`~ovos_memory_plugins.local_rag.LocalRAGMemory`: keyword
recall catches exact terms (names, codes, rare words) that dense embeddings miss,
while semantics catches paraphrases keywords miss. Combine the two through
:class:`~ovos_memory_plugins.composite.CompositeMemory` for hybrid search.

Configuration (persona JSON block)::

    {
      "memory_module": "ovos-memory-plugin-lexical",
      "ovos-memory-plugin-lexical": {
        "db_path": "~/.local/share/ovos/lexical_memory.db",
        "table": "lexical_memory",
        "retrieval": {"max_num_results": 5, "min_score": null},
        "inject_mode": "system",
        "system_prompt": "You are a helpful assistant."
      }
    }

Note: lexical ``score`` is derived from BM25 and is **not** on the 0..1 scale of
semantic cosine similarity — set ``retrieval.min_score`` accordingly, and prefer
rank-based fusion (RRF) when combining with other retrievers.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from ovos_utils.log import LOG

from ovos_memory_plugins.base import BaseRetrievalMemory
from ovos_memory_plugins.common import MemoryHit

DEFAULT_DB_PATH = "~/.local/share/ovos/lexical_memory.db"
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)  # unicode-aware alphanumeric runs


def _safe_identifier(name: str) -> str:
    """Sanitize a table name to a safe SQL identifier (alnum + underscore)."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


class LexicalMemory(BaseRetrievalMemory):
    """Keyword recall over stored exchanges via SQLite FTS5 + BM25.

    Lexical-specific configuration keys
    -----------------------------------
    db_path : str  (default ``~/.local/share/ovos/lexical_memory.db``)
        Path to the SQLite database file. ``":memory:"`` for an ephemeral store.
    table : str  (default ``"lexical_memory"``)
        FTS5 virtual-table name.

    See :class:`~ovos_memory_plugins.base.BaseRetrievalMemory` for the shared
    ``retrieval`` / ``context`` / ``inject_mode`` / ``max_history`` keys.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.db_path: str = self.config.get("db_path", DEFAULT_DB_PATH)
        self.table: str = _safe_identifier(self.config.get("table", "lexical_memory"))
        # injectable connection for tests
        self._con: Optional[sqlite3.Connection] = self.config.get("_con")
        self.fts5_available: bool = True

    # ------------------------------------------------------------------ backend
    @property
    def con(self) -> sqlite3.Connection:
        """The SQLite connection, opened lazily with the FTS5 table ensured."""
        if self._con is None:
            if self.db_path != ":memory:":
                Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            path = self.db_path if self.db_path == ":memory:" else str(Path(self.db_path).expanduser())
            self._con = sqlite3.connect(path, check_same_thread=False)
            self._ensure_table(self._con)
        return self._con

    def _ensure_table(self, con: sqlite3.Connection) -> None:
        """Create the FTS5 virtual table; flag if this SQLite build lacks FTS5."""
        try:
            con.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table} "
                f"USING fts5(doc_id UNINDEXED, session_id UNINDEXED, content)")
            con.commit()
        except sqlite3.OperationalError as e:
            self.fts5_available = False
            LOG.error(f"LexicalMemory: FTS5 unavailable in this SQLite build ({e}); "
                      f"keyword recall disabled")

    # ----------------------------------------------------------- storage hooks
    def _store_document(self, doc_id: str, text: str, session_id: str,
                        metadata: Dict[str, Any]) -> None:
        if not self.fts5_available:
            return
        self.con.execute(
            f"INSERT INTO {self.table}(doc_id, session_id, content) VALUES (?, ?, ?)",
            (doc_id, session_id, text))
        self.con.commit()

    def _query_backend(self, query: str, top_k: int) -> List[MemoryHit]:
        if not self.fts5_available:
            return []
        match = self._build_match(query)
        if not match:
            return []
        # bm25() returns lower (more negative) for better matches; negate for a
        # positive "higher = better" score.
        rows = self.con.execute(
            f"SELECT doc_id, content, bm25({self.table}) AS rank "
            f"FROM {self.table} WHERE {self.table} MATCH ? ORDER BY rank LIMIT ?",
            (match, top_k)).fetchall()
        return [MemoryHit(content=content, source=doc_id, score=-float(rank))
                for doc_id, content, rank in rows]

    @staticmethod
    def _build_match(query: str) -> str:
        """Turn free text into a safe FTS5 MATCH expression (alnum terms OR'd).

        Tokenizing to bare terms avoids FTS5 syntax errors on punctuation/quotes
        and treats the query as "any of these words".
        """
        terms = _WORD_RE.findall(query.lower())
        return " OR ".join(terms)
