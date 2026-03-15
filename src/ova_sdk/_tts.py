"""TTS resource — pure text-to-speech (no LLM)."""

from __future__ import annotations

import time
from typing import Iterator

import httpx

from ._base import raise_for_status_streaming
from ._errors import OVAConnectionError, OVATimeoutError
from ._models import BatchTTSItem, BatchTTSResult
from ._streaming import AsyncAudioStream, AudioStream


class TTSResource:
    def __init__(self, client: httpx.Client):
        self._client = client

    def generate(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AudioStream:
        """POST /v1/text-to-speech[/{voice}] — synthesize text exactly as given.

        Args:
            text: Text to synthesize.
            voice: Optional voice name (non-sticky, per-request only).
            language: Optional language code (non-sticky, per-request only).
        """
        path = f"/v1/text-to-speech/{voice}" if voice else "/v1/text-to-speech"
        params = {}
        if language is not None:
            params["language"] = language
        payload: dict = {"text": text}

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

    def batch_generate(
        self,
        items: list[dict | BatchTTSItem],
    ) -> list[BatchTTSResult]:
        """POST /v1/text-to-speech/batch — synthesize multiple texts in a single batched call.

        Args:
            items: List of dicts or BatchTTSItem with keys: text, voice (optional), language (optional).

        Returns:
            List of BatchTTSResult sorted by index.
        """
        payload = {
            "items": [
                item.model_dump() if isinstance(item, BatchTTSItem) else item
                for item in items
            ]
        }

        try:
            r = self._client.post("/v1/text-to-speech/batch", json=payload, timeout=300)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        if r.status_code != 200:
            raise_for_status_streaming(r, r.content)

        results = []
        for line in r.text.strip().split("\n"):
            if line.strip():
                import json
                results.append(BatchTTSResult.model_validate(json.loads(line)))
        results.sort(key=lambda r: r.index)
        return results

    def batch_stream(
        self,
        items: list[dict | BatchTTSItem],
    ) -> Iterator[BatchTTSResult]:
        """POST /v1/text-to-speech/batch — stream results as they complete.

        Args:
            items: List of dicts or BatchTTSItem with keys: text, voice (optional), language (optional).

        Yields:
            BatchTTSResult for each completed item.
        """
        payload = {
            "items": [
                item.model_dump() if isinstance(item, BatchTTSItem) else item
                for item in items
            ]
        }

        try:
            req = self._client.build_request("POST", "/v1/text-to-speech/batch", json=payload)
            r = self._client.send(req, stream=True, timeout=300)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        if r.status_code != 200:
            body = r.read()
            r.close()
            raise_for_status_streaming(r, body)

        try:
            import json
            for line in r.iter_lines():
                if line.strip():
                    yield BatchTTSResult.model_validate(json.loads(line))
        finally:
            r.close()


class AsyncTTSResource:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def generate(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncAudioStream:
        """POST /v1/text-to-speech[/{voice}] — synthesize text exactly as given.

        Args:
            text: Text to synthesize.
            voice: Optional voice name (non-sticky, per-request only).
            language: Optional language code (non-sticky, per-request only).
        """
        path = f"/v1/text-to-speech/{voice}" if voice else "/v1/text-to-speech"
        params = {}
        if language is not None:
            params["language"] = language
        payload: dict = {"text": text}

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

    async def batch_generate(
        self,
        items: list[dict | BatchTTSItem],
    ) -> list[BatchTTSResult]:
        """POST /v1/text-to-speech/batch — synthesize multiple texts in a single batched call.

        Args:
            items: List of dicts or BatchTTSItem with keys: text, voice (optional), language (optional).

        Returns:
            List of BatchTTSResult sorted by index.
        """
        payload = {
            "items": [
                item.model_dump() if isinstance(item, BatchTTSItem) else item
                for item in items
            ]
        }

        try:
            r = await self._client.post("/v1/text-to-speech/batch", json=payload, timeout=300)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        if r.status_code != 200:
            raise_for_status_streaming(r, r.content)

        results = []
        for line in r.text.strip().split("\n"):
            if line.strip():
                import json
                results.append(BatchTTSResult.model_validate(json.loads(line)))
        results.sort(key=lambda r: r.index)
        return results

    async def batch_stream(
        self,
        items: list[dict | BatchTTSItem],
    ):
        """POST /v1/text-to-speech/batch — async stream results as they complete.

        Args:
            items: List of dicts or BatchTTSItem with keys: text, voice (optional), language (optional).

        Yields:
            BatchTTSResult for each completed item.
        """
        payload = {
            "items": [
                item.model_dump() if isinstance(item, BatchTTSItem) else item
                for item in items
            ]
        }

        try:
            req = self._client.build_request("POST", "/v1/text-to-speech/batch", json=payload)
            r = await self._client.send(req, stream=True, timeout=300)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        if r.status_code != 200:
            body = await r.aread()
            await r.aclose()
            raise_for_status_streaming(r, body)

        try:
            import json
            async for line in r.aiter_lines():
                if line.strip():
                    yield BatchTTSResult.model_validate(json.loads(line))
        finally:
            await r.aclose()
