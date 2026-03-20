"""OVA (sync) and AsyncOVA (async) client classes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

import httpx

from ._asr import ASRResource, AsyncASRResource
from ._duplex import AsyncDuplexResource, DuplexResource
from ._base import (
    DEFAULT_TIMEOUT,
    _resolve_api_key,
    _resolve_base_url,
    make_async_client,
    make_sync_client,
    raise_for_status,
)
from ._chat import AsyncChatResource, ChatResource
from ._dialogue import AsyncDialogueResource, DialogueResource
from ._errors import OVAAuthenticationError, OVAConnectionError, OVAServerNotReady, OVATimeoutError
from ._models import Info, Transcription
from ._settings import AsyncSettingsResource, SettingsResource
from ._tts import AsyncTTSResource, TTSResource


class OVA:
    """Synchronous OVA client.

    Usage::

        client = OVA()
        client.wait_until_ready()
        audio = client.tts.generate("Hello world")
        audio.save("output.wav")
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self._base_url = _resolve_base_url(base_url)
        self._api_key = _resolve_api_key(api_key)
        self._http = make_sync_client(self._base_url, self._api_key, timeout)

        self.tts = TTSResource(self._http)
        self.chat = ChatResource(self._http)
        self.dialogue = DialogueResource(self._http)
        self.settings = SettingsResource(self._http)
        self.asr = ASRResource(self._base_url, self._api_key)
        self.duplex = DuplexResource(self._base_url, self._api_key)

    def info(self) -> Info:
        """GET /v1/info — pipeline configuration."""
        try:
            r = self._http.get("/v1/info")
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        raise_for_status(r)
        return Info.model_validate(r.json())

    def languages(self) -> list[str]:
        """Available language codes for the active TTS engine."""
        return self.settings.get().languages

    def voices(self, language: str | None = None) -> list[str]:
        """Available voice IDs. Optionally filter by language."""
        settings = self.settings.get()
        if language:
            return [v.id for v in settings.voices.get(language, [])]
        seen: set[str] = set()
        result: list[str] = []
        for lang_voices in settings.voices.values():
            for v in lang_voices:
                if v.id not in seen:
                    seen.add(v.id)
                    result.append(v.id)
        return result

    def ready(self) -> bool:
        """GET /v1/health — True if server is warmed up.

        Raises OVAAuthenticationError immediately on 401.
        """
        try:
            r = self._http.get("/v1/health")
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        if r.status_code == 401:
            raise OVAAuthenticationError("Invalid or missing API key")
        return r.status_code == 200

    def wait_until_ready(self, timeout: float = 120, poll_interval: float = 2) -> None:
        """Block until the server is ready or timeout is reached."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ready():
                return
            time.sleep(poll_interval)
        raise OVAServerNotReady(f"Server not ready after {timeout}s")

    def transcribe(
        self,
        audio: Union[str, Path, bytes],
        *,
        language: Optional[str] = None,
    ) -> Transcription:
        """POST /v1/speech-to-text — speech-to-text (standalone, no LLM).

        Args:
            audio: WAV audio as file path, Path, or bytes.
            language: Optional language code override for ASR (e.g., 'es', 'en').
        """
        if isinstance(audio, (str, Path)):
            wav_bytes = Path(audio).read_bytes()
        else:
            wav_bytes = audio

        params = {}
        if language is not None:
            params["language"] = language

        try:
            r = self._http.post("/v1/speech-to-text", content=wav_bytes, params=params)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        raise_for_status(r)
        return Transcription.model_validate(r.json())

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncOVA:
    """Asynchronous OVA client.

    Usage::

        async with AsyncOVA() as client:
            await client.wait_until_ready()
            audio = await client.tts.generate("Hello world")
            await audio.save("output.wav")
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self._base_url = _resolve_base_url(base_url)
        self._api_key = _resolve_api_key(api_key)
        self._http = make_async_client(self._base_url, self._api_key, timeout)

        self.tts = AsyncTTSResource(self._http)
        self.chat = AsyncChatResource(self._http)
        self.dialogue = AsyncDialogueResource(self._http)
        self.settings = AsyncSettingsResource(self._http)
        self.asr = AsyncASRResource(self._base_url, self._api_key)
        self.duplex = AsyncDuplexResource(self._base_url, self._api_key)

    async def info(self) -> Info:
        try:
            r = await self._http.get("/v1/info")
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        raise_for_status(r)
        return Info.model_validate(r.json())

    async def languages(self) -> list[str]:
        """Available language codes for the active TTS engine."""
        settings = await self.settings.get()
        return settings.languages

    async def voices(self, language: str | None = None) -> list[str]:
        """Available voice IDs. Optionally filter by language."""
        settings = await self.settings.get()
        if language:
            return [v.id for v in settings.voices.get(language, [])]
        seen: set[str] = set()
        result: list[str] = []
        for lang_voices in settings.voices.values():
            for v in lang_voices:
                if v.id not in seen:
                    seen.add(v.id)
                    result.append(v.id)
        return result

    async def ready(self) -> bool:
        """GET /v1/health — True if server is warmed up.

        Raises OVAAuthenticationError immediately on 401.
        """
        try:
            r = await self._http.get("/v1/health")
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        if r.status_code == 401:
            raise OVAAuthenticationError("Invalid or missing API key")
        return r.status_code == 200

    async def wait_until_ready(self, timeout: float = 120, poll_interval: float = 2) -> None:
        import asyncio

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.ready():
                return
            await asyncio.sleep(poll_interval)
        raise OVAServerNotReady(f"Server not ready after {timeout}s")

    async def transcribe(
        self,
        audio: Union[str, Path, bytes],
        *,
        language: Optional[str] = None,
    ) -> Transcription:
        """POST /v1/speech-to-text — speech-to-text (standalone, no LLM).

        Args:
            audio: WAV audio as file path, Path, or bytes.
            language: Optional language code override for ASR (e.g., 'es', 'en').
        """
        if isinstance(audio, (str, Path)):
            wav_bytes = Path(audio).read_bytes()
        else:
            wav_bytes = audio

        params = {}
        if language is not None:
            params["language"] = language

        try:
            r = await self._http.post("/v1/speech-to-text", content=wav_bytes, params=params)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        raise_for_status(r)
        return Transcription.model_validate(r.json())

    async def close(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
