"""OVOS memory plugins — local-first conversational memory for personas.

Backends (all ``opm.agents.memory``):

- :class:`~ovos_memory_plugins.longterm.LongTermMemory` — rolling LLM summarization.
- :class:`~ovos_memory_plugins.local_rag.LocalRAGMemory` — in-process semantic RAG.
- :class:`~ovos_memory_plugins.lexical.LexicalMemory` — SQLite FTS5 keyword recall.
- :class:`~ovos_memory_plugins.recency.RecencyMemory` — sliding short-term buffer.
- :class:`~ovos_memory_plugins.entity.EntityMemory` — durable user-fact extraction.
- :class:`~ovos_memory_plugins.composite.CompositeMemory` — ensemble orchestrator.
"""
from ovos_memory_plugins.base import BaseRetrievalMemory
from ovos_memory_plugins.common import MemoryHit
from ovos_memory_plugins.composite import CompositeMemory
from ovos_memory_plugins.entity import EntityMemory
from ovos_memory_plugins.lexical import LexicalMemory
from ovos_memory_plugins.local_rag import LocalRAGMemory
from ovos_memory_plugins.longterm import LongTermMemory
from ovos_memory_plugins.recency import RecencyMemory

__all__ = [
    "BaseRetrievalMemory",
    "MemoryHit",
    "CompositeMemory",
    "EntityMemory",
    "LexicalMemory",
    "LocalRAGMemory",
    "LongTermMemory",
    "RecencyMemory",
]
