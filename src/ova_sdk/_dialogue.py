"""Dialogue resource — multi-speaker text-to-dialogue via batch TTS endpoint."""

from __future__ import annotations

import json
from typing import Sequence

import httpx

from ._base import raise_for_status_streaming
from ._errors import OVAConnectionError, OVATimeoutError
from ._models import BatchTTSResult, DialogueBatchResult, DialogueInput


def _build_batch_items(
    inputs: Sequence[DialogueInput | dict],
    language: str | None,
) -> list[dict]:
    items = []
    for inp in inputs:
        if isinstance(inp, DialogueInput):
            item = {"text": inp.text, "voice": inp.voice_id}
        else:
            item = {"text": inp["text"], "voice": inp["voice_id"]}
        if language is not None:
            item["language"] = language
        items.append(item)
    return items


def _parse_batch_response(text: str) -> DialogueBatchResult:
    results = []
    for line in text.strip().split("\n"):
        if line.strip():
            results.append(BatchTTSResult.model_validate(json.loads(line)))
    results.sort(key=lambda r: r.index)
    return DialogueBatchResult(results)


class DialogueResource:
    def __init__(self, client: httpx.Client):
        self._client = client

    def generate(
        self,
        inputs: Sequence[DialogueInput | dict],
        *,
        language: str | None = None,
    ) -> DialogueBatchResult:
        """Dialogue synthesis via POST /v1/text-to-speech/batch.

        Args:
            inputs: List of DialogueInput or dicts with ``text`` and ``voice_id``.
            language: Optional language code (non-sticky, per-request only).

        Returns:
            DialogueBatchResult with concatenated audio and per-segment access.
        """
        return self.batch_generate(inputs, language=language)

    def batch_generate(
        self,
        inputs: Sequence[DialogueInput | dict],
        *,
        language: str | None = None,
    ) -> DialogueBatchResult:
        """Batch dialogue synthesis via POST /v1/text-to-speech/batch.

        Sends all dialogue segments in a single batched request, which processes
        all lines through the transformer simultaneously for better throughput.

        Args:
            inputs: List of DialogueInput or dicts with ``text`` and ``voice_id``.
            language: Optional language code (non-sticky, per-request only).

        Returns:
            DialogueBatchResult with concatenated audio and per-segment access.
        """
        payload = {"items": _build_batch_items(inputs, language)}

        try:
            r = self._client.post("/v1/text-to-speech/batch", json=payload, timeout=300)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        if r.status_code != 200:
            raise_for_status_streaming(r, r.content)

        return _parse_batch_response(r.text)


class AsyncDialogueResource:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def generate(
        self,
        inputs: Sequence[DialogueInput | dict],
        *,
        language: str | None = None,
    ) -> DialogueBatchResult:
        """Dialogue synthesis via POST /v1/text-to-speech/batch.

        Args:
            inputs: List of DialogueInput or dicts with ``text`` and ``voice_id``.
            language: Optional language code (non-sticky, per-request only).

        Returns:
            DialogueBatchResult with concatenated audio and per-segment access.
        """
        return await self.batch_generate(inputs, language=language)

    async def batch_generate(
        self,
        inputs: Sequence[DialogueInput | dict],
        *,
        language: str | None = None,
    ) -> DialogueBatchResult:
        """Batch dialogue synthesis via POST /v1/text-to-speech/batch.

        Sends all dialogue segments in a single batched request, which processes
        all lines through the transformer simultaneously for better throughput.

        Args:
            inputs: List of DialogueInput or dicts with ``text`` and ``voice_id``.
            language: Optional language code (non-sticky, per-request only).

        Returns:
            DialogueBatchResult with concatenated audio and per-segment access.
        """
        payload = {"items": _build_batch_items(inputs, language)}

        try:
            r = await self._client.post("/v1/text-to-speech/batch", json=payload, timeout=300)
        except httpx.ConnectError as e:
            raise OVAConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise OVATimeoutError(str(e)) from e
        if r.status_code != 200:
            raise_for_status_streaming(r, r.content)

        return _parse_batch_response(r.text)
