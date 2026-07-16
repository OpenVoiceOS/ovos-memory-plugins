"""Regression tests for the ``close()`` lifecycle across memory plugins."""
import sqlite3

import pytest

from ovos_memory_plugins.longterm import LongTermMemory
from ovos_memory_plugins.entity import EntityMemory
from ovos_memory_plugins.lexical import LexicalMemory


def test_longterm_sqlite_close(tmp_path):
    mem = LongTermMemory(config={"backend": "sqlite",
                                 "db_path": str(tmp_path / "lt.db")})
    con = mem._store.con
    mem.close()
    with pytest.raises(sqlite3.ProgrammingError):
        con.execute("SELECT 1")


def test_longterm_json_close_is_noop(tmp_path):
    mem = LongTermMemory(config={"backend": "json",
                                 "db_path": str(tmp_path / "lt.json")})
    mem.close()  # JSON store has no OS resources; must not raise


def test_longterm_context_manager(tmp_path):
    with LongTermMemory(config={"backend": "sqlite",
                                "db_path": str(tmp_path / "lt.db")}) as mem:
        con = mem._store.con
    with pytest.raises(sqlite3.ProgrammingError):
        con.execute("SELECT 1")


def test_entity_close(tmp_path):
    mem = EntityMemory(config={"backend": "json",
                               "db_path": str(tmp_path / "e.json")})
    mem.close()  # must not raise


def test_lexical_close(tmp_path):
    mem = LexicalMemory(config={"db_path": str(tmp_path / "lex.db")})
    _ = mem.con  # force lazy open
    con = mem._con
    mem.close()
    assert mem._con is None
    with pytest.raises(sqlite3.ProgrammingError):
        con.execute("SELECT 1")


def test_lexical_close_before_open(tmp_path):
    mem = LexicalMemory(config={"db_path": str(tmp_path / "lex.db")})
    mem.close()  # never opened; must not raise


def test_composite_close_forwards_to_members(tmp_path):
    called = {}

    class _FakeMember:
        def close(self):
            called["closed"] = True

    from ovos_memory_plugins.composite import CompositeMemory
    mem = CompositeMemory.__new__(CompositeMemory)
    mem.members = [("fake", _FakeMember(), 1.0)]
    mem.close()
    assert called.get("closed") is True
