"""
RAG memory plugin for OVOS personas.

Stores every exchange as a document via an OpenAI-compatible ``/v1/files``
endpoint and retrieves the most relevant prior exchanges via
``/v1/embeddings`` cosine similarity before each response.

Entry point: ``ovos-memory-plugin-rag``  (``opm.agents.memory``)

Endpoint contract (mirrors ovos-persona-server PR #11 / feat/rag)
-----------------------------------------------------------------
POST /v1/embeddings
    body: {"input": <str>}
    response: {"data": [{"embedding": [float, ...]}]}

POST /v1/files
    multipart/form-data: file=<content bytes>, purpose=<str>
    response: {"id": <str>, ...}

GET /v1/files/<file_id>/content
    response: raw text

POST /v1/vector_stores/<collection>/files
    body: {"file_id": <str>}

POST /v1/vector_stores/<collection>/search
    body: {"query": <str>, "k": <int>}
    response: {"data": [{"content": <str>, "score": <float>}]}

When vector_stores endpoints are unavailable the plugin falls back to
in-process cosine similarity against locally cached embeddings.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from ovos_plugin_manager.templates.agents import AgentContextManager, AgentMessage, MessageRole
from ovos_utils.log import LOG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed(api_url: str, text: str, timeout: int = 30) -> Optional[List[float]]:
    url = api_url.rstrip("/") + "/embeddings"
    try:
        resp = requests.post(url, json={"input": text}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception as exc:
        LOG.warning(f"RAGMemory: embed failed: {exc}")
        return None


def _upload_file(api_url: str, content: str, filename: str, timeout: int = 30) -> Optional[str]:
    """Upload text content to /v1/files; returns file_id or None."""
    url = api_url.rstrip("/") + "/files"
    try:
        resp = requests.post(
            url,
            files={"file": (filename, content.encode(), "text/plain")},
            data={"purpose": "memory"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("id")
    except Exception as exc:
        LOG.warning(f"RAGMemory: file upload failed: {exc}")
        return None


def _attach_to_store(api_url: str, collection: str, file_id: str, timeout: int = 30):
    """Attach a file to a named vector store."""
    url = api_url.rstrip("/") + f"/vector_stores/{collection}/files"
    try:
        resp = requests.post(url, json={"file_id": file_id}, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        LOG.warning(f"RAGMemory: attach to store failed: {exc}")


def _vector_search(api_url: str, collection: str, query: str, k: int,
                   timeout: int = 30) -> List[Tuple[str, float]]:
    """Search via /v1/vector_stores/<collection>/search."""
    url = api_url.rstrip("/") + f"/vector_stores/{collection}/search"
    try:
        resp = requests.post(url, json={"query": query, "k": k}, timeout=timeout)
        resp.raise_for_status()
        items = resp.json().get("data", [])
        return [(item["content"], float(item.get("score", 0.0))) for item in items]
    except Exception as exc:
        LOG.debug(f"RAGMemory: vector_stores search unavailable ({exc}); using fallback")
        return []


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class RAGMemory(AgentContextManager):
    """
    RAG-backed memory plugin.  Stores every exchange via the files API, uses
    embeddings to find the most relevant prior turns, and injects them as
    context messages before the current utterance.

    Configuration keys
    ------------------
    api_url : str
        Base URL of the OpenAI-compatible server.
    collection : str  (default: ``"ovos_memory"``)
        Name of the vector store / collection to use.
    top_k : int  (default: 3)
        Maximum number of similar documents to retrieve per query.
    min_score : float  (default: 0.0)
        Minimum cosine similarity to include a retrieved document.
    system_prompt : str
        Optional persona system prompt.
    request_timeout : int  (default: 30)
        HTTP request timeout in seconds.
    inject_as_system : bool  (default: False)
        When True, retrieved context is injected as a single SYSTEM message
        instead of individual ASSISTANT messages.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.api_url: str = self.config.get("api_url", "http://192.168.1.200:8000/v1")
        self.collection: str = self.config.get("collection", "ovos_memory")
        self.top_k: int = int(self.config.get("top_k", 3))
        self.min_score: float = float(self.config.get("min_score", 0.0))
        self.request_timeout: int = int(self.config.get("request_timeout", 30))
        self.inject_as_system: bool = bool(self.config.get("inject_as_system", False))

        # in-process fallback store: list of (embedding, text)
        self._local_store: List[Tuple[List[float], str]] = []

        # per-session recent window for plain history
        self._recent: Dict[str, List[AgentMessage]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _local_search(self, query_vec: List[float], k: int) -> List[Tuple[str, float]]:
        if not self._local_store:
            return []
        scored = [(text, _cosine(query_vec, emb)) for emb, text in self._local_store]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _store_document(self, text: str, session_id: str):
        """Embed + upload + attach. Falls back gracefully on any error."""
        try:
            emb = _embed(self.api_url, text, timeout=self.request_timeout)
            if emb:
                self._local_store.append((emb, text))  # fallback copy always kept
        except Exception as exc:
            LOG.warning(f"RAGMemory: embed failed in _store_document: {exc}")

        try:
            file_id = _upload_file(
                self.api_url, text,
                filename=f"memory_{session_id}_{int(time.time())}.txt",
                timeout=self.request_timeout,
            )
            if file_id:
                _attach_to_store(self.api_url, self.collection, file_id, timeout=self.request_timeout)
        except Exception as exc:
            LOG.warning(f"RAGMemory: upload/attach failed in _store_document: {exc}")

    def _retrieve(self, utterance: str) -> List[Tuple[str, float]]:
        """Return top-k (text, score) pairs above min_score."""
        # Try server-side vector search first
        try:
            results = _vector_search(self.api_url, self.collection, utterance,
                                     self.top_k, timeout=self.request_timeout)
        except Exception as exc:
            LOG.debug(f"RAGMemory: vector_search raised unexpectedly: {exc}")
            results = []

        # Fall back to local cosine if server-side returned nothing
        if not results:
            try:
                query_vec = _embed(self.api_url, utterance, timeout=self.request_timeout)
                if query_vec:
                    results = self._local_search(query_vec, self.top_k)
            except Exception as exc:
                LOG.debug(f"RAGMemory: local fallback embed failed: {exc}")

        return [(text, score) for text, score in results if score >= self.min_score]

    # ------------------------------------------------------------------
    # AgentContextManager interface
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> List[AgentMessage]:
        return list(self._recent.get(session_id, []))

    def update_history(self, new_messages: List[AgentMessage], session_id: str):
        if session_id not in self._recent:
            self._recent[session_id] = []

        recent = self._recent[session_id]

        # Store each assistant response as a retrievable document
        for msg in new_messages:
            if msg.role == MessageRole.ASSISTANT:
                # store exchange pair if we have the user turn
                if recent and recent[-1].role == MessageRole.USER:
                    doc = f"Q: {recent[-1].content}\nA: {msg.content}"
                else:
                    doc = msg.content
                self._store_document(doc, session_id)
            recent.append(msg)

        # Keep at most 10 recent messages in RAM
        self._recent[session_id] = recent[-10:]

    def build_conversation_context(self, utterance: str, session_id: str) -> List[AgentMessage]:
        context: List[AgentMessage] = []

        # System prompt
        if self.system_prompt.strip():
            context.append(AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt.strip()))

        # Retrieved similar past exchanges
        retrieved = self._retrieve(utterance)
        if retrieved:
            if self.inject_as_system:
                snippets = "\n\n".join(f"[Relevant past exchange]\n{text}" for text, _ in retrieved)
                context.append(AgentMessage(role=MessageRole.SYSTEM, content=snippets))
            else:
                for text, score in retrieved:
                    context.append(
                        AgentMessage(role=MessageRole.ASSISTANT,
                                     content=f"[Recalled — similarity {score:.2f}]\n{text}")
                    )

        # Recent verbatim history (excluding trailing user turns to avoid duplication)
        history = list(self._recent.get(session_id, []))
        while history and history[-1].role == MessageRole.USER:
            history.pop()
        context.extend(history)

        # Current utterance
        context.append(AgentMessage(role=MessageRole.USER, content=utterance.strip()))
        return context
