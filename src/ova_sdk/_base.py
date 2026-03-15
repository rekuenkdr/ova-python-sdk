"""URL resolution, httpx client factory, and shared response helpers."""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import quote

import httpx

from ._errors import OVAAuthenticationError, OVARequestError


DEFAULT_BASE_URL = "http://localhost:5173"
DEFAULT_TIMEOUT = 120.0


def raise_for_status(response: httpx.Response) -> None:
    """Raise on non-200 for non-streaming responses (settings, info, transcribe).

    401 → OVAAuthenticationError, other → OVARequestError.
    """
    if response.status_code == 200:
        return
    if response.status_code == 401:
        raise OVAAuthenticationError("Invalid or missing API key")
    raise OVARequestError(response.status_code, response.text)


def raise_for_status_streaming(response: httpx.Response, body: bytes) -> None:
    """Raise on non-200 for streaming responses (tts, chat).

    The body must already be read/closed by the caller before passing here.
    401 → OVAAuthenticationError, other → OVARequestError.
    """
    if response.status_code == 200:
        return
    text = body.decode(errors="replace")
    if response.status_code == 401:
        raise OVAAuthenticationError("Invalid or missing API key")
    raise OVARequestError(response.status_code, text)


def _resolve_base_url(base_url: Optional[str]) -> str:
    url = base_url or os.environ.get("OVA_BASE_URL", DEFAULT_BASE_URL)
    return url.rstrip("/")


def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    key = api_key or os.environ.get("OVA_API_KEY", "") or None
    return key


def _build_headers(api_key: Optional[str]) -> dict:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def make_sync_client(
    base_url: str,
    api_key: Optional[str],
    timeout: float = DEFAULT_TIMEOUT,
) -> httpx.Client:
    return httpx.Client(
        base_url=base_url,
        headers=_build_headers(api_key),
        timeout=timeout,
    )


def make_async_client(
    base_url: str,
    api_key: Optional[str],
    timeout: float = DEFAULT_TIMEOUT,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url,
        headers=_build_headers(api_key),
        timeout=timeout,
    )


def ws_url(base_url: str, path: str, api_key: Optional[str]) -> str:
    """Build a WebSocket URL from HTTP base URL + path, with optional api_key query param."""
    url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    full = f"{url}{path}"
    if api_key:
        sep = "&" if "?" in full else "?"
        full += f"{sep}api_key={quote(api_key, safe='')}"
    return full
