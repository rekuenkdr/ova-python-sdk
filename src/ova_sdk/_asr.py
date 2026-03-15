"""ASR resource — streaming speech-to-text via WebSocket."""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from ._base import ws_url
from ._errors import OVAConnectionError


class ASRStream:
    """Sync WebSocket session for streaming ASR."""

    def __init__(self, ws):
        self._ws = ws
        self._partial = ""

    def send(self, audio: np.ndarray) -> None:
        """Send a float32 audio chunk."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        self._ws.send(audio.tobytes())

    def receive(self) -> dict:
        """Receive a message (partial or final transcript)."""
        msg = self._ws.recv()
        data = json.loads(msg)
        if "partial" in data:
            self._partial = data["partial"]
        return data

    def finish(self) -> str:
        """Signal end of audio and get final transcript."""
        self._ws.send(json.dumps({"action": "end"}))
        msg = self._ws.recv()
        data = json.loads(msg)
        return data.get("final", self._partial)

    def close(self):
        self._ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncASRStream:
    """Async WebSocket session for streaming ASR."""

    def __init__(self, ws):
        self._ws = ws
        self._partial = ""

    async def send(self, audio: np.ndarray) -> None:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        await self._ws.send(audio.tobytes())

    async def receive(self) -> dict:
        msg = await self._ws.recv()
        data = json.loads(msg)
        if "partial" in data:
            self._partial = data["partial"]
        return data

    async def finish(self) -> str:
        await self._ws.send(json.dumps({"action": "end"}))
        msg = await self._ws.recv()
        data = json.loads(msg)
        return data.get("final", self._partial)

    async def close(self):
        await self._ws.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class ASRResource:
    def __init__(self, base_url: str, api_key: Optional[str]):
        self._base_url = base_url
        self._api_key = api_key

    def stream(self) -> ASRStream:
        """Open a WebSocket connection for streaming ASR."""
        try:
            from websockets.sync.client import connect
        except ImportError:
            raise ImportError(
                "WebSocket ASR requires 'websockets'. "
                "Install with: pip install ova-sdk[asr]"
            )

        url = ws_url(self._base_url, "/v1/speech-to-text/stream", self._api_key)
        try:
            ws = connect(url)
        except Exception as e:
            raise OVAConnectionError(f"WebSocket connection failed: {e}") from e
        return ASRStream(ws)


class AsyncASRResource:
    def __init__(self, base_url: str, api_key: Optional[str]):
        self._base_url = base_url
        self._api_key = api_key

    async def stream(self) -> AsyncASRStream:
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            raise ImportError(
                "WebSocket ASR requires 'websockets'. "
                "Install with: pip install ova-sdk[asr]"
            )

        url = ws_url(self._base_url, "/v1/speech-to-text/stream", self._api_key)
        try:
            ws = await connect(url)
        except Exception as e:
            raise OVAConnectionError(f"WebSocket connection failed: {e}") from e
        return AsyncASRStream(ws)
