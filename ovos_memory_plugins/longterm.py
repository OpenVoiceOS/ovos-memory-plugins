"""
Long-term memory plugin for OVOS personas.

Wraps a short-term in-memory store; every ``summarize_every`` exchanges it
summarizes older turns into a rolling text summary via an OpenAI-compatible
chat endpoint, then persists the rolling summary + the recent window to disk
(JSON or SQLite).

Entry point: ``ovos-memory-plugin-longterm``  (``opm.agents.memory``)
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ovos_plugin_manager.templates.agents import AgentContextManager, AgentMessage, MessageRole
from ovos_utils.log import LOG

from ovos_memory_plugins._llm import chat_complete, resolve_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _messages_to_text(messages: List[AgentMessage]) -> str:
    """Flatten a list of AgentMessages into a readable transcript."""
    lines = []
    for m in messages:
        role = m.role.value if hasattr(m.role, "value") else str(m.role)
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence backends
# ---------------------------------------------------------------------------

class _JsonStore:
    """Simple JSON file store keyed by session_id."""

    def __init__(self, db_path: str):
        self.path = Path(db_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}")

    def load(self, session_id: str) -> Dict:
        try:
            data = json.loads(self.path.read_text())
            return data.get(session_id, {})
        except Exception:
            return {}

    def save(self, session_id: str, record: Dict):
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            data = {}
        data[session_id] = record
        self.path.write_text(json.dumps(data, indent=2))


class _SqliteStore:
    """SQLite store keyed by session_id."""

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

    def save(self, session_id: str, record: Dict):
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


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class LongTermMemory(AgentContextManager):
    """
    Long-term memory plugin that summarizes older conversation turns via an
    OpenAI-compatible chat endpoint and persists the rolling summary.

    Configuration keys
    ------------------
    api_url : str
        Base URL of the OpenAI-compatible server (e.g. ``http://192.168.1.200:8000/v1``).
    model : str
        Model name to send in chat/completions requests.
    summarize_every : int  (default: 6)
        Number of full exchanges (user + assistant pairs) to accumulate in the
        recent window before triggering a summarization pass.  One exchange = 2
        messages.  So the default triggers after 6 turns (12 messages).
    max_summary_tokens : int  (default: 256)
        ``max_tokens`` cap passed to the summarisation request.
    recent_window : int  (default: 4)
        Number of recent exchanges to keep verbatim after summarization.
    backend : str  (default: ``"json"``)
        Persistence backend — ``"json"`` or ``"sqlite"``.
    db_path : str
        Path to the JSON file or SQLite database.
        Default: ``~/.local/share/ovos/longterm_memory.json`` (or ``.db``).
    system_prompt : str
        Optional persona system prompt prepended to every context.
    request_timeout : int  (default: 30)
        HTTP request timeout in seconds.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.api_url: str = self.config.get("api_url", "http://192.168.1.200:8000/v1")
        self.model: str = self.config.get("model", "")
        self.summarize_every: int = int(self.config.get("summarize_every", 6))
        self.max_summary_tokens: int = int(self.config.get("max_summary_tokens", 256))
        self.recent_window: int = int(self.config.get("recent_window", 4))
        self.request_timeout: int = int(self.config.get("request_timeout", 30))

        backend = self.config.get("backend", "json").lower()
        default_ext = ".db" if backend == "sqlite" else ".json"
        default_path = f"~/.local/share/ovos/longterm_memory{default_ext}"
        db_path = self.config.get("db_path", default_path)

        self._store: Union[_SqliteStore, _JsonStore]
        if backend == "sqlite":
            self._store = _SqliteStore(db_path)
        else:
            self._store = _JsonStore(db_path)

        # in-memory cache: session_id → {summary, recent, exchange_count}
        self._cache: Dict[str, Dict] = {}

        if not self.model:
            self._resolve_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model(self):
        """Auto-detect the first available model from the endpoint."""
        model = resolve_model(self.api_url)
        if model:
            self.model = model

    def _load_session(self, session_id: str) -> Dict:
        if session_id not in self._cache:
            record = self._store.load(session_id)
            self._cache[session_id] = {
                "summary": record.get("summary", ""),
                "recent": [
                    AgentMessage(role=MessageRole(m["role"]), content=m["content"])
                    for m in (record.get("recent") or [])
                ],
                "exchange_count": record.get("exchange_count", 0),
            }
        return self._cache[session_id]

    def _persist_session(self, session_id: str):
        rec = self._cache[session_id]
        self._store.save(session_id, {
            "summary": rec["summary"],
            "recent": [{"role": m.role.value if hasattr(m.role, "value") else str(m.role),
                        "content": m.content}
                       for m in rec["recent"]],
            "exchange_count": rec["exchange_count"],
            "updated_at": time.time(),
        })

    def _maybe_summarize(self, session_id: str):
        """Trigger summarization if the exchange counter threshold is reached."""
        rec = self._cache[session_id]
        if rec["exchange_count"] < self.summarize_every:
            return

        to_summarize = rec["recent"][:- (self.recent_window * 2)] if self.recent_window > 0 else rec["recent"]
        if not to_summarize:
            rec["exchange_count"] = 0
            return

        transcript = _messages_to_text(to_summarize)
        previous_summary = rec["summary"]

        prompt_parts = []
        if previous_summary:
            prompt_parts.append(f"Previous summary:\n{previous_summary}\n")
        prompt_parts.append(f"New conversation:\n{transcript}\n")
        prompt_parts.append("Write a concise updated summary of what was discussed. Be brief.")

        prompt = "\n".join(prompt_parts)

        try:
            new_summary = chat_complete(
                api_url=self.api_url,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_summary_tokens,
                timeout=self.request_timeout,
            )
            rec["summary"] = new_summary
            # keep only recent window verbatim
            if self.recent_window > 0:
                rec["recent"] = rec["recent"][-(self.recent_window * 2):]
            else:
                rec["recent"] = []
            rec["exchange_count"] = 0
            LOG.info(f"LongTermMemory: summarized session '{session_id}' ({len(to_summarize)} messages → summary)")
        except Exception as exc:
            LOG.error(f"LongTermMemory: summarization failed for session '{session_id}': {exc}")

    # ------------------------------------------------------------------
    # AgentContextManager interface
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> List[AgentMessage]:
        rec = self._load_session(session_id)
        return list(rec["recent"])

    def update_history(self, new_messages: List[AgentMessage], session_id: str):
        rec = self._load_session(session_id)
        recent = rec["recent"]

        if recent:
            last = recent[-1]
            first_new = new_messages[0]
            # drop hanging user messages
            if first_new.role == MessageRole.USER and last.role == MessageRole.USER:
                recent.pop()
            # merge consecutive assistant messages
            if first_new.role == MessageRole.ASSISTANT and last.role == MessageRole.ASSISTANT:
                new_messages[0].content = last.content + "\n" + first_new.content
                recent.pop()

        recent.extend(new_messages)

        # count exchanges: each assistant message = one exchange
        for m in new_messages:
            if m.role == MessageRole.ASSISTANT:
                rec["exchange_count"] += 1

        self._maybe_summarize(session_id)
        self._persist_session(session_id)

    def build_conversation_context(self, utterance: str, session_id: str) -> List[AgentMessage]:
        rec = self._load_session(session_id)
        context: List[AgentMessage] = []

        # System prompt (may include rolling summary)
        system_parts = []
        if self.system_prompt.strip():
            system_parts.append(self.system_prompt.strip())
        if rec["summary"]:
            system_parts.append(f"[Conversation summary so far]\n{rec['summary']}")
        if system_parts:
            context.append(AgentMessage(role=MessageRole.SYSTEM, content="\n\n".join(system_parts)))

        # Recent verbatim history
        history = list(rec["recent"])
        # drop trailing unmatched user messages
        while history and history[-1].role == MessageRole.USER:
            history.pop()
        context.extend(history)

        # Current utterance
        context.append(AgentMessage(role=MessageRole.USER, content=utterance.strip()))
        return context
