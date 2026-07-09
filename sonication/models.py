"""Startup model discovery.

Queries each service's /v1/models and adopts the first model id as the one to
use (LLM, STT, and TTS), and reads the LLM context window (max_model_len) for
the context budget. Falls back to the configured defaults if a service is
unreachable, so a discovery failure never takes the server down.
"""
from typing import Optional

import httpx

from . import client, config


async def _first_model(base_url: str, api_key: str,
                       http_client: Optional[httpx.AsyncClient] = None
                       ) -> tuple[Optional[str], Optional[int]]:
    """Query /v1/models and return (first_model_id, max_model_len).

    Args:
        base_url: Service base URL.
        api_key: API key for auth.
        http_client: Optional httpx.AsyncClient. If None, a temporary client
                     is created and closed after this call.
    """
    c, close_after = client._make_client(http_client)
    try:
        resp = await c.get(
            f"{base_url}/v1/models", headers=config.bearer(api_key))
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or []
        if not data:
            return None, None
        return data[0].get("id"), data[0].get("max_model_len")
    finally:
        if close_after:
            await c.aclose()


async def discover() -> dict:
    info: dict = {}

    try:
        model_id, ctx = await _first_model(config.LLM_BASE_URL, config.LLM_API_KEY)
        if model_id:
            config.LLM_MODEL = model_id
        if ctx:
            config.LLM_CONTEXT_TOKENS = int(ctx)
        info["llm"] = {"model": config.LLM_MODEL, "context_tokens": config.LLM_CONTEXT_TOKENS}
    except Exception as exc:
        info["llm"] = {"error": str(exc), "model": config.LLM_MODEL}

    try:
        model_id, _ = await _first_model(config.STT_BASE_URL, config.STT_API_KEY)
        if model_id:
            config.STT_MODEL = model_id
        info["stt"] = {"model": config.STT_MODEL}
    except Exception as exc:
        info["stt"] = {"error": str(exc), "model": config.STT_MODEL}

    try:
        model_id, _ = await _first_model(config.TTS_BASE_URL, config.TTS_API_KEY)
        if model_id:
            config.TTS_MODEL = model_id
        info["tts"] = {"model": config.TTS_MODEL or "(none)"}
    except Exception as exc:
        info["tts"] = {"error": str(exc), "model": config.TTS_MODEL or "(none)"}

    return info
