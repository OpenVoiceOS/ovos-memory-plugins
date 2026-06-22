# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Minimal OpenAI-compatible chat client shared by the LLM-backed memory plugins.

Local-first: this talks to any OpenAI-compatible ``/v1`` endpoint (a local
llama.cpp / vLLM / Ollama server), never to a hosted provider with a baked-in
key. Used by :mod:`ovos_memory_plugins.longterm` (rolling summaries) and
:mod:`ovos_memory_plugins.entity` (fact extraction).
"""
from __future__ import annotations

from typing import Dict, List

import requests
from ovos_utils.log import LOG


def chat_complete(api_url: str, model: str, messages: List[Dict], max_tokens: int = 256,
                  timeout: int = 30) -> str:
    """Send a single chat-completion request and return the assistant text.

    Args:
        api_url: Base URL of the OpenAI-compatible server (``.../v1``).
        model: Model name to request.
        messages: OpenAI-style ``[{"role": ..., "content": ...}]`` messages.
        max_tokens: Cap on generated tokens.
        timeout: HTTP timeout in seconds.

    Returns:
        The assistant message content, stripped.
    """
    url = api_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def resolve_model(api_url: str, timeout: int = 10) -> str:
    """Return the first model id advertised by the endpoint, or ``""`` on failure.

    Queries ``GET /models`` and reads either the OpenAI ``data`` list or a bare
    ``models`` list. Never raises — logs and returns ``""`` so construction of a
    plugin never hard-fails just because the endpoint is offline.
    """
    try:
        url = api_url.rstrip("/") + "/models"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        models = body.get("data") or body.get("models", [])
        if models:
            model = models[0].get("id") or models[0].get("name", "")
            LOG.debug(f"resolve_model: auto-selected model '{model}'")
            return model
    except Exception as exc:
        LOG.warning(f"resolve_model: could not auto-detect model: {exc}")
    return ""
