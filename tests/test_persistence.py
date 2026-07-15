"""Regression tests for the shared persistence backends.

Covers the atomicity, concurrency, corruption-preservation and lifecycle
guarantees of :mod:`ovos_memory_plugins.persistence`.
"""
import json
import os
import sqlite3
import threading
from unittest.mock import patch

import pytest

from ovos_memory_plugins.persistence import JsonStore, SqliteStore


# ---------------------------------------------------------------------------
# Shared-module promotion
# ---------------------------------------------------------------------------

def test_entity_and_longterm_share_the_same_store_class():
    """entity/longterm must reuse the shared class, not a private per-plugin copy."""
    from ovos_memory_plugins import entity, longterm
    assert entity.JsonStore is JsonStore
    # longterm keeps a back-compat alias pointing at the shared class
    assert longterm._JsonStore is JsonStore
    assert longterm._SqliteStore is SqliteStore


# ---------------------------------------------------------------------------
# Concurrent writers
# ---------------------------------------------------------------------------

def test_concurrent_writers_do_not_lose_entries(tmp_path):
    store = JsonStore(str(tmp_path / "mem.json"))
    n = 40

    def writer(i):
        store.save(f"session-{i}", {"value": i})

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = json.loads((tmp_path / "mem.json").read_text())
    assert len(data) == n
    for i in range(n):
        assert data[f"session-{i}"] == {"value": i}


def test_concurrent_writers_updating_same_session_stay_consistent(tmp_path):
    store = JsonStore(str(tmp_path / "mem.json"))

    def writer(i):
        for _ in range(20):
            store.save("shared", {"value": i})

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # file must always be valid JSON with a coherent record (last writer wins)
    data = json.loads((tmp_path / "mem.json").read_text())
    assert set(data.keys()) == {"shared"}
    assert data["shared"]["value"] in range(8)


# ---------------------------------------------------------------------------
# Kill-mid-write never truncates the main file
# ---------------------------------------------------------------------------

def test_failed_replace_leaves_previous_file_intact(tmp_path):
    path = tmp_path / "mem.json"
    store = JsonStore(str(path))
    store.save("s1", {"value": "good"})
    good = path.read_text()

    # simulate a crash exactly at the atomic rename
    with patch("ovos_memory_plugins.persistence.os.replace",
               side_effect=OSError("boom")):
        with pytest.raises(OSError):
            store.save("s2", {"value": "new"})

    # main file must be byte-for-byte the previous good version, never truncated
    assert path.read_text() == good
    assert json.loads(path.read_text()) == {"s1": {"value": "good"}}


def test_failed_write_cleans_up_temp_files(tmp_path):
    path = tmp_path / "mem.json"
    store = JsonStore(str(path))
    store.save("s1", {"value": "good"})

    with patch("ovos_memory_plugins.persistence.os.replace",
               side_effect=OSError("boom")):
        with pytest.raises(OSError):
            store.save("s2", {"value": "new"})

    # no leftover .tmp files in the directory
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_temp_file_is_created_in_same_directory(tmp_path):
    """os.replace is only atomic on the same filesystem → temp must be co-located."""
    path = tmp_path / "sub" / "mem.json"
    store = JsonStore(str(path))
    seen_dirs = []
    real_replace = os.replace

    def spy(src, dst):
        seen_dirs.append(os.path.dirname(os.path.abspath(src)))
        return real_replace(src, dst)

    with patch("ovos_memory_plugins.persistence.os.replace", side_effect=spy):
        store.save("s1", {"value": 1})

    assert seen_dirs
    assert seen_dirs[-1] == str(path.parent)


# ---------------------------------------------------------------------------
# Corrupt-file load preserves the original and logs
# ---------------------------------------------------------------------------

def test_corrupt_file_is_preserved_not_reset(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text("{ this is not valid json ")
    corrupt = tmp_path / "mem.json.corrupt"

    with patch("ovos_memory_plugins.persistence.LOG") as log:
        store = JsonStore(str(path))
        # loading a corrupt file must not raise and must yield an empty record
        assert store.load("anything") == {}
        assert log.error.called

    # original bytes preserved under .corrupt, not silently discarded
    assert corrupt.exists()
    assert corrupt.read_text() == "{ this is not valid json "


def test_write_after_corruption_starts_fresh_and_persists(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text("not json")

    store = JsonStore(str(path))
    store.save("s1", {"value": 1})

    # after quarantine, a fresh valid store exists with the new record
    data = json.loads(path.read_text())
    assert data == {"s1": {"value": 1}}
    assert (tmp_path / "mem.json.corrupt").exists()


def test_valid_file_is_never_quarantined(tmp_path):
    path = tmp_path / "mem.json"
    store = JsonStore(str(path))
    store.save("s1", {"value": 1})
    store.load("s1")
    assert not (tmp_path / "mem.json.corrupt").exists()


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_json_store_round_trip(tmp_path):
    store = JsonStore(str(tmp_path / "mem.json"))
    store.save("s1", {"summary": "hi", "recent": [1, 2]})
    assert store.load("s1") == {"summary": "hi", "recent": [1, 2]}
    assert store.load("missing") == {}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_sqlite_store_close_closes_connection(tmp_path):
    store = SqliteStore(str(tmp_path / "mem.db"))
    store.save("s1", {"summary": "hi"})
    con = store.con
    store.close()
    assert store.con is None
    import sqlite3
    with pytest.raises(sqlite3.ProgrammingError):
        con.execute("SELECT 1")


def test_sqlite_store_close_is_idempotent(tmp_path):
    store = SqliteStore(str(tmp_path / "mem.db"))
    store.close()
    store.close()  # must not raise


def test_sqlite_store_context_manager(tmp_path):
    with SqliteStore(str(tmp_path / "mem.db")) as store:
        store.save("s1", {"summary": "hi"})
        con = store.con
    assert store.con is None
    with pytest.raises(sqlite3.ProgrammingError):
        con.execute("SELECT 1")


def test_json_store_context_manager(tmp_path):
    with JsonStore(str(tmp_path / "mem.json")) as store:
        store.save("s1", {"value": 1})
    assert store.load("s1") == {"value": 1}
