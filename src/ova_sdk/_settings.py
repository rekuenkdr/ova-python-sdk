"""Settings resource for OVA SDK."""

from __future__ import annotations

from typing import Optional

import httpx

from ._base import raise_for_status
from ._errors import OVAConnectionError, OVATimeoutError
from ._models import ReloadPromptResponse, Settings, SettingsUpdateResponse


class SettingsResource:
    def __init__(self, client: httpx.Client):
        self._client = client

    def get(self) -> Settings:
        """GET /v1/settings — current config and available profiles."""
        try:
            r = self._client.get("/v1/settings")
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        raise_for_status(r)
        return Settings.model_validate(r.json())

    def update(
        self,
        *,
        language: Optional[str] = None,
        tts_engine: Optional[str] = None,
        voice: Optional[str] = None,
        stream_format: Optional[str] = None,
    ) -> SettingsUpdateResponse:
        """POST /v1/settings — update server settings."""
        payload = {}
        if language is not None:
            payload["language"] = language
        if tts_engine is not None:
            payload["tts_engine"] = tts_engine
        if voice is not None:
            payload["voice"] = voice
        if stream_format is not None:
            payload["stream_format"] = stream_format

        try:
            r = self._client.post("/v1/settings", json=payload)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        raise_for_status(r)
        return SettingsUpdateResponse.model_validate(r.json())

    def reload_prompt(
        self,
        *,
        language: Optional[str] = None,
        profile: Optional[str] = None,
        prompt: Optional[str] = None,
        clear_history: bool = False,
    ) -> ReloadPromptResponse:
        """POST /v1/settings/prompt — update system prompt."""
        payload = {}
        if language is not None:
            payload["language"] = language
        if profile is not None:
            payload["profile"] = profile
        if prompt is not None:
            payload["prompt"] = prompt
        if clear_history:
            payload["clear_history"] = True

        try:
            r = self._client.post("/v1/settings/prompt", json=payload)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        raise_for_status(r)
        return ReloadPromptResponse.model_validate(r.json())


class AsyncSettingsResource:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def get(self) -> Settings:
        try:
            r = await self._client.get("/v1/settings")
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        raise_for_status(r)
        return Settings.model_validate(r.json())

    async def update(
        self,
        *,
        language: Optional[str] = None,
        tts_engine: Optional[str] = None,
        voice: Optional[str] = None,
        stream_format: Optional[str] = None,
    ) -> SettingsUpdateResponse:
        payload = {}
        if language is not None:
            payload["language"] = language
        if tts_engine is not None:
            payload["tts_engine"] = tts_engine
        if voice is not None:
            payload["voice"] = voice
        if stream_format is not None:
            payload["stream_format"] = stream_format

        try:
            r = await self._client.post("/v1/settings", json=payload)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        raise_for_status(r)
        return SettingsUpdateResponse.model_validate(r.json())

    async def reload_prompt(
        self,
        *,
        language: Optional[str] = None,
        profile: Optional[str] = None,
        prompt: Optional[str] = None,
        clear_history: bool = False,
    ) -> ReloadPromptResponse:
        payload = {}
        if language is not None:
            payload["language"] = language
        if profile is not None:
            payload["profile"] = profile
        if prompt is not None:
            payload["prompt"] = prompt
        if clear_history:
            payload["clear_history"] = True

        try:
            r = await self._client.post("/v1/settings/prompt", json=payload)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        raise_for_status(r)
        return ReloadPromptResponse.model_validate(r.json())
