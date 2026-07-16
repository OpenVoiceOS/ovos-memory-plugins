# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Shared persistence backends for memory plugins.

Both :class:`~ovos_memory_plugins.longterm.LongTermMemory` and
:class:`~ovos_memory_plugins.entity.EntityMemory` persist a per-``session_id``
record to disk. This module holds the storage primitives they share so neither
plugin has to reach into the other's internals:

- :class:`JsonStore` — a plain JSON file keyed by ``session_id``. Writes are
  **atomic** (temp file in the same directory + :func:`os.replace`) and
  **serialized** by a lock, so a crash mid-write or two concurrent writers can
  never truncate or interleave the file. A file that fails to parse on load is
  **quarantined** (renamed to ``<name>.corrupt``) with a logged error rather than
  silently discarded, so no data is lost without a trace.
- :class:`SqliteStore` — a transactional SQLite table keyed by ``session_id``.
  Already atomic per statement; adds an explicit :meth:`~SqliteStore.close`.

Both expose the same ``load(session_id) -> dict`` / ``save(session_id, record)``
contract plus a ``close()`` lifecycle hook and context-manager support.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict

from ovos_utils.log import LOG


class JsonStore:
    """Atomic, thread-safe JSON file store keyed by ``session_id``.

    A single JSON object maps each ``session_id`` to its record. Every ``save``
    does a read-modify-write of the whole file guarded by a per-instance lock and
    committed with an atomic rename, so a killed process or a disk-full error
    leaves the previous good file intact instead of a truncated one. Concurrent
    writers within a process are serialized by the lock.
    """

    def __init__(self, db_path: str):
        self.path = Path(db_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # guards the read-modify-write cycle so concurrent savers never lose or
        # interleave updates (the whole file is rewritten on every save)
        self._lock = threading.Lock()
        if not self.path.exists():
            self._atomic_write({})

    # ------------------------------------------------------------------ helpers
    def _quarantine(self, exc: Exception) -> None:
        """Preserve an unreadable file as ``<name>.corrupt`` and log the error.

        Never silently reset to ``{}`` — a corrupt store is a data-loss event the
        operator must be able to see and recover from.
        """
        corrupt = self.path.with_name(self.path.name + ".corrupt")
        try:
            os.replace(self.path, corrupt)
            LOG.error(f"JsonStore: {self.path} is corrupt ({exc}); "
                      f"preserved as {corrupt}, starting fresh")
        except OSError as move_exc:
            LOG.error(f"JsonStore: {self.path} is corrupt ({exc}) and could not be "
                      f"quarantined ({move_exc}); starting fresh")

    def _read_all(self) -> Dict:
        """Return the whole file as a dict, quarantining a corrupt file first."""
        try:
            return json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except ValueError as exc:  # malformed JSON
            self._quarantine(exc)
            return {}
        except OSError as exc:  # unreadable for some other reason
            LOG.error(f"JsonStore: cannot read {self.path} ({exc}); starting fresh")
            return {}

    def _atomic_write(self, data: Dict) -> None:
        """Write ``data`` to a temp file in the same dir then atomically replace.

        The temp file shares the destination directory so :func:`os.replace` is a
        same-filesystem atomic rename. If anything fails before the replace, the
        temp file is removed and the destination is left untouched.
        """
        fd, tmp = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -------------------------------------------------------------------- store
    def load(self, session_id: str) -> Dict:
        with self._lock:
            return self._read_all().get(session_id, {})

    def save(self, session_id: str, record: Dict) -> None:
        with self._lock:
            data = self._read_all()
            data[session_id] = record
            self._atomic_write(data)

    def close(self) -> None:
        """No OS resources to release; present for a uniform lifecycle."""

    def __enter__(self) -> "JsonStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class SqliteStore:
    """Transactional SQLite store keyed by ``session_id``."""

    def __init__(self, db_path: str):
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path), check_same_thread=False)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS sessions "
            "(session_id TEXT PRIMARY KEY, summary TEXT, recent TEXT, exchange_count INTEGER, updated_at REAL)"
        )
        self.con.commit()

    def load(self, session_id: str) -> Dict:
        cur = self.con.execute(
            "SELECT summary, recent, exchange_count, updated_at FROM sessions WHERE session_id=?",
            (session_id,)
        )
        row = cur.fetchone()
        if not row:
            return {}
        return {
            "summary": row[0],
            "recent": json.loads(row[1]) if row[1] else [],
            "exchange_count": row[2],
            "updated_at": row[3],
        }

    def save(self, session_id: str, record: Dict) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO sessions (session_id, summary, recent, exchange_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                record.get("summary", ""),
                json.dumps(record.get("recent", [])),
                record.get("exchange_count", 0),
                record.get("updated_at", time.time()),
            )
        )
        self.con.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection (idempotent)."""
        if self.con is not None:
            try:
                self.con.close()
            finally:
                self.con = None  # type: ignore[assignment]

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
