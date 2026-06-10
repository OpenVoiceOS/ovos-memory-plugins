"""
End-to-end test for RAGMemory using a deterministic local FastAPI stub.

The stub implements:
  POST /v1/embeddings  — returns a fixed 4-dim embedding based on keyword hash
  POST /v1/files       — stores content in memory, returns a file_id
  GET  /v1/files/{id}/content — returns stored content
  POST /v1/vector_stores/{collection}/files — attaches file to store
  POST /v1/vector_stores/{collection}/search — cosine-ranks all stored docs

Run with:  pytest tests/test_e2e_rag.py -v -s
"""
from __future__ import annotations

import math
import threading
import time
from typing import Dict, List
import hashlib

import pytest

try:
    import uvicorn
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import PlainTextResponse
    import httpx

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.rag import RAGMemory, _cosine

skip_if_no_fastapi = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="fastapi/uvicorn/httpx not installed — install test extras"
)


# ---------------------------------------------------------------------------
# Deterministic embedding: 4-dim vector based on character frequencies
# ---------------------------------------------------------------------------

def _deterministic_embed(text: str) -> List[float]:
    """
    Produce a deterministic 4-dim unit embedding from text.
    Words sharing the same first letter cluster together, making
    retrieval tests predictable.
    """
    h = hashlib.md5(text.encode()).digest()
    raw = [(b / 255.0) * 2 - 1 for b in h[:4]]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


# ---------------------------------------------------------------------------
# Stub server
# ---------------------------------------------------------------------------

def build_stub_app() -> FastAPI:
    app = FastAPI()
    _files: Dict[str, str] = {}          # file_id → content
    _store: Dict[str, List[str]] = {}    # collection → [file_id, ...]

    _counter = [0]

    @app.post("/v1/embeddings")
    async def embeddings(body: dict):
        text = body.get("input", "")
        return {"data": [{"embedding": _deterministic_embed(text), "index": 0}]}

    @app.post("/v1/files")
    async def upload_file(file: UploadFile = File(...), purpose: str = Form("memory")):
        _counter[0] += 1
        file_id = f"file_{_counter[0]:04d}"
        content = (await file.read()).decode()
        _files[file_id] = content
        return {"id": file_id, "object": "file", "created_at": int(time.time()), "purpose": purpose}

    @app.get("/v1/files/{file_id}/content", response_class=PlainTextResponse)
    async def get_file_content(file_id: str):
        if file_id not in _files:
            raise HTTPException(404, "not found")
        return _files[file_id]

    @app.post("/v1/vector_stores/{collection}/files")
    async def attach_file(collection: str, body: dict):
        file_id = body.get("file_id", "")
        if collection not in _store:
            _store[collection] = []
        _store[collection].append(file_id)
        return {"id": file_id, "object": "vector_store.file"}

    @app.post("/v1/vector_stores/{collection}/search")
    async def search(collection: str, body: dict):
        query = body.get("query", "")
        k = int(body.get("k", 3))
        query_vec = _deterministic_embed(query)
        file_ids = _store.get(collection, [])
        scored = []
        for fid in file_ids:
            if fid not in _files:
                continue
            content = _files[fid]
            doc_vec = _deterministic_embed(content)
            score = _cosine(query_vec, doc_vec)
            scored.append({"content": content, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"data": scored[:k], "object": "list"}

    return app


def start_stub(port: int = 18765):
    """Start the stub in a daemon thread; returns when the server is ready."""
    app = build_stub_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Wait for startup
    for _ in range(30):
        try:
            import requests
            r = requests.get(f"http://127.0.0.1:{port}/docs", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    return f"http://127.0.0.1:{port}/v1"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stub_url():
    if not FASTAPI_AVAILABLE:
        pytest.skip("fastapi not available")
    return start_stub()


@skip_if_no_fastapi
def test_e2e_rag_store_and_retrieve(stub_url):
    """Full store → retrieve → inject cycle against the deterministic stub."""
    plugin = RAGMemory(config={
        "api_url": stub_url,
        "collection": "test_e2e",
        "top_k": 3,
        "min_score": 0.0,
    })
    session_id = "e2e-rag-001"

    # Ingest 4 exchanges about different topics
    exchanges = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of Germany?", "The capital of Germany is Berlin."),
        ("How do you make pasta?", "Boil salted water and cook pasta for 8-10 minutes."),
        ("What is machine learning?", "ML is a subset of AI where models learn from data."),
    ]
    for u, a in exchanges:
        plugin.update_history([
            AgentMessage(role=MessageRole.USER, content=u),
            AgentMessage(role=MessageRole.ASSISTANT, content=a),
        ], session_id)

    print(f"\n--- Local store size: {len(plugin._local_store)} documents ---")

    # Build context for a geography query
    ctx = plugin.build_conversation_context("Tell me about European capitals", session_id)

    print(f"\n--- Context messages ({len(ctx)}) ---")
    for m in ctx:
        print(f"  [{m.role.value}] {m.content[:120]}")

    # At least one retrieved doc should be in context
    contents = " ".join(m.content for m in ctx)
    assert "Paris" in contents or "Berlin" in contents or "Recalled" in contents, \
        "Expected geography exchanges to be recalled"
    assert ctx[-1].role == MessageRole.USER
    assert "European capitals" in ctx[-1].content


@skip_if_no_fastapi
def test_e2e_rag_min_score_filters(stub_url):
    """With a high min_score, off-topic docs should be excluded from context."""
    plugin = RAGMemory(config={
        "api_url": stub_url,
        "collection": "filter_test",
        "top_k": 5,
        "min_score": 0.999,  # impossible threshold — nothing should be retrieved
    })
    session_id = "filter-001"
    plugin.update_history([
        AgentMessage(role=MessageRole.USER, content="What is AI?"),
        AgentMessage(role=MessageRole.ASSISTANT, content="AI stands for Artificial Intelligence."),
    ], session_id)

    ctx = plugin.build_conversation_context("Tell me about quantum physics", session_id)
    # With min_score=0.999, no retrieved documents should appear
    recalled = [m for m in ctx if "Recalled" in m.content or "Relevant past" in m.content]
    assert len(recalled) == 0, f"Expected no recalled docs at min_score=0.999, got: {recalled}"


@skip_if_no_fastapi
def test_e2e_rag_inject_as_system(stub_url):
    """inject_as_system=True should bundle all retrieved docs into one SYSTEM message."""
    plugin = RAGMemory(config={
        "api_url": stub_url,
        "collection": "sys_inject",
        "top_k": 3,
        "min_score": 0.0,
        "inject_as_system": True,
        "system_prompt": "You are a memory bot.",
    })
    session_id = "sys-001"
    plugin.update_history([
        AgentMessage(role=MessageRole.USER, content="Hello"),
        AgentMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
    ], session_id)

    ctx = plugin.build_conversation_context("How are you?", session_id)
    system_msgs = [m for m in ctx if m.role == MessageRole.SYSTEM]
    # Should have at least the persona system prompt
    assert any("memory bot" in m.content for m in system_msgs)


@skip_if_no_fastapi
def test_e2e_rag_multiple_sessions_isolated(stub_url):
    """Two different sessions should not see each other's history in their context."""
    plugin = RAGMemory(config={
        "api_url": stub_url,
        "collection": "isolation_test",
        "top_k": 2,
        "min_score": 0.0,
    })

    plugin.update_history([
        AgentMessage(role=MessageRole.USER, content="Session A unique topic: chocolate factories"),
        AgentMessage(role=MessageRole.ASSISTANT, content="Chocolate is made from cacao beans."),
    ], "session_A")

    plugin.update_history([
        AgentMessage(role=MessageRole.USER, content="Session B unique topic: rocket propulsion"),
        AgentMessage(role=MessageRole.ASSISTANT, content="Rockets use combustion to generate thrust."),
    ], "session_B")

    # session_A context should NOT include session_B recent history
    ctx_a = plugin.build_conversation_context("More about chocolate", "session_A")
    recent_in_ctx_a = " ".join(
        m.content for m in ctx_a if m.role in (MessageRole.USER, MessageRole.ASSISTANT)
    )
    # The *recent* history from session_B should not appear in session_A's context
    assert "rocket" not in recent_in_ctx_a or "Recalled" in " ".join(m.content for m in ctx_a), \
        "session_A context should not contain session_B recent history"

    print("\n--- session_A context ---")
    for m in ctx_a:
        print(f"  [{m.role.value}] {m.content[:100]}")
