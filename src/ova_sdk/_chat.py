"""Chat resource — send text/audio through the LLM and get audio back."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Optional, Union

import httpx

from ._base import raise_for_status_streaming
from ._errors import OVAConnectionError, OVATimeoutError
from ._streaming import AsyncAudioStream, AudioStream


class ChatResource:
    def __init__(self, client: httpx.Client):
        self._client = client

    def send_text(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        image: Optional[Union[str, Path]] = None,
    ) -> AudioStream:
        """POST /v1/chat[/{voice}] — send text (+ optional image) through LLM, get audio.

        Args:
            text: Message text.
            voice: Optional voice name (sticky — changes session).
            language: Optional language code (sticky — changes session).
            image: Optional image (file path, Path, data URL, or base64).
        """
        path = f"/v1/chat/{voice}" if voice else "/v1/chat"
        params = {}
        if language is not None:
            params["language"] = language
        payload: dict = {"text": text}
        if image is not None:
            payload["image"] = _resolve_image(image)

        try:
            t0 = time.perf_counter()
            req = self._client.build_request("POST", path, json=payload, params=params)
            r = self._client.send(req, stream=True)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        if r.status_code != 200:
            body = r.read()
            r.close()
            raise_for_status_streaming(r, body)
        return AudioStream(r, _request_time=t0)

    def send_audio(
        self,
        audio: Union[str, Path, bytes],
        *,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> AudioStream:
        """POST /v1/chat[/{voice}]/audio — send WAV audio, get audio response.

        Args:
            audio: WAV audio as file path, Path, or bytes.
            voice: Optional voice name (sticky — changes session).
            language: Optional language code (sticky — changes session).
        """
        if isinstance(audio, (str, Path)):
            wav_bytes = Path(audio).read_bytes()
        else:
            wav_bytes = audio

        path = f"/v1/chat/{voice}/audio" if voice else "/v1/chat/audio"
        params = {}
        if language is not None:
            params["language"] = language
        headers = {"Content-Type": "audio/wav"}

        try:
            t0 = time.perf_counter()
            req = self._client.build_request("POST", path, content=wav_bytes, headers=headers, params=params)
            r = self._client.send(req, stream=True)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        if r.status_code != 200:
            body = r.read()
            r.close()
            raise_for_status_streaming(r, body)
        return AudioStream(r, _request_time=t0)


class AsyncChatResource:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def send_text(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        image: Optional[Union[str, Path]] = None,
    ) -> AsyncAudioStream:
        """POST /v1/chat[/{voice}] — send text (+ optional image) through LLM, get audio.

        Args:
            text: Message text.
            voice: Optional voice name (sticky — changes session).
            language: Optional language code (sticky — changes session).
            image: Optional image (file path, Path, data URL, or base64).
        """
        path = f"/v1/chat/{voice}" if voice else "/v1/chat"
        params = {}
        if language is not None:
            params["language"] = language
        payload: dict = {"text": text}
        if image is not None:
            payload["image"] = _resolve_image(image)

        try:
            t0 = time.perf_counter()
            req = self._client.build_request("POST", path, json=payload, params=params)
            r = await self._client.send(req, stream=True)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        if r.status_code != 200:
            body = await r.aread()
            await r.aclose()
            raise_for_status_streaming(r, body)
        return AsyncAudioStream(r, _request_time=t0)

    async def send_audio(
        self,
        audio: Union[str, Path, bytes],
        *,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> AsyncAudioStream:
        """POST /v1/chat[/{voice}]/audio — send WAV audio, get audio response.

        Args:
            audio: WAV audio as file path, Path, or bytes.
            voice: Optional voice name (sticky — changes session).
            language: Optional language code (sticky — changes session).
        """
        if isinstance(audio, (str, Path)):
            wav_bytes = Path(audio).read_bytes()
        else:
            wav_bytes = audio

        path = f"/v1/chat/{voice}/audio" if voice else "/v1/chat/audio"
        params = {}
        if language is not None:
            params["language"] = language
        headers = {"Content-Type": "audio/wav"}

        try:
            t0 = time.perf_counter()
            req = self._client.build_request("POST", path, content=wav_bytes, headers=headers, params=params)
            r = await self._client.send(req, stream=True)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        if r.status_code != 200:
            body = await r.aread()
            await r.aclose()
            raise_for_status_streaming(r, body)
        return AsyncAudioStream(r, _request_time=t0)


def _resolve_image(image: Union[str, Path]) -> str:
    """Convert image path to data URL or pass through if already a data URL / base64 string."""
    if isinstance(image, Path):
        if not image.is_file():
            raise ValueError(f"Image file not found: {image}")
        return _encode_image_file(image)

    s = str(image)

    # Data URLs pass through
    if s.startswith("data:"):
        return s

    # Try as file path
    p = Path(s)
    if p.is_file():
        return _encode_image_file(p)

    # If it looks like a path (has separators or extension), reject it
    if "/" in s or "\\" in s or (s.startswith(".") and len(s) > 1):
        raise ValueError(f"Image file not found: {s}")

    # Assume raw base64
    return s


def _encode_image_file(p: Path) -> str:
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode()
    suffix = p.suffix.lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif"}
    content_type = mime.get(suffix.lstrip("."), "image/jpeg")
    return f"data:{content_type};base64,{b64}"
