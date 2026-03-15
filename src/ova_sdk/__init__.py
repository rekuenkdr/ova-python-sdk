"""OVA Python SDK — programmatic access to the OVA voice assistant."""

__version__ = "0.1.0"

from ._asr import ASRStream, AsyncASRStream
from ._audio import play, save
from ._client import AsyncOVA, OVA
from ._errors import (
    OVAAuthenticationError,
    OVAConnectionError,
    OVAError,
    OVARequestError,
    OVAServerNotReady,
    OVATimeoutError,
)
from ._models import BatchTTSItem, BatchTTSResult, DialogueBatchResult, DialogueInput, Info, Settings, Transcription
from ._streaming import AsyncAudioStream, AudioStream

__all__ = [
    "OVA",
    "AsyncOVA",
    "AudioStream",
    "AsyncAudioStream",
    "ASRStream",
    "AsyncASRStream",
    "BatchTTSItem",
    "BatchTTSResult",
    "DialogueBatchResult",
    "DialogueInput",
    "Info",
    "Settings",
    "Transcription",
    "OVAError",
    "OVAAuthenticationError",
    "OVAConnectionError",
    "OVAServerNotReady",
    "OVARequestError",
    "OVATimeoutError",
    "play",
    "save",
]
