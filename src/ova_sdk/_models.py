"""Pydantic models mirroring OVA server JSON responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class VoiceInfo(BaseModel):
    id: str
    name: str


class Info(BaseModel):
    voice: Optional[str] = None
    tts_engine: str
    language: str
    supports_streaming: bool
    pcm_prebuffer_samples: int
    early_tts_decode: bool
    frontend_settings_disabled: bool = False
    multimodal_disabled: bool = False


class CurrentSettings(BaseModel):
    language: str
    tts_engine: str
    voice: Optional[str] = None
    stream_format: str
    system_prompt: str


class ProfileInfo(BaseModel):
    prompt: str


class Settings(BaseModel):
    current: CurrentSettings
    profiles: Dict[str, Dict[str, ProfileInfo]] = {}
    default_prompts: Dict[str, str] = {}
    languages: List[str] = []
    voices: Dict[str, List[VoiceInfo]] = {}


class SettingsUpdateResponse(BaseModel):
    restart_required: bool = False
    message: Optional[str] = None
    error: Optional[str] = None
    success: Optional[bool] = None


class ReloadPromptResponse(BaseModel):
    success: bool
    prompt: Optional[str] = None


class DialogueInput(BaseModel):
    """A single dialogue segment: text + voice_id."""
    text: str
    voice_id: str


class Transcription(BaseModel):
    """Response from POST /v1/speech-to-text."""
    text: str


class BatchTTSItem(BaseModel):
    """A single item in a batch TTS request."""
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None


class BatchTTSResult(BaseModel):
    """A single result from a batch TTS response (one NDJSON line)."""
    index: int
    status: str  # "ok" or "error"
    voice: Optional[str] = None
    language: Optional[str] = None
    audio_b64: Optional[str] = None
    error: Optional[str] = None

    def to_bytes(self) -> bytes:
        """Decode audio_b64 to raw WAV bytes."""
        import base64

        if self.audio_b64 is None:
            return b""
        return base64.b64decode(self.audio_b64)

    def save(self, path: str) -> None:
        """Save audio to a WAV file."""
        data = self.to_bytes()
        if data:
            with open(path, "wb") as f:
                f.write(data)

    def play(self) -> None:
        """Play audio using the SDK's play utility."""
        from ._audio import play

        data = self.to_bytes()
        if data:
            play(data)


class DialogueBatchResult:
    """Result from dialogue.batch_generate() — concatenated audio with per-segment access."""

    def __init__(self, segments: list[BatchTTSResult]):
        self.segments = segments

    def to_bytes(self) -> bytes:
        """Concatenate all segment WAVs into a single WAV file."""
        import io
        import struct
        import wave

        pcm_parts = []
        sample_rate = None
        channels = None
        sampwidth = None

        for seg in self.segments:
            wav_bytes = seg.to_bytes()
            if not wav_bytes:
                continue
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                if sample_rate is None:
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                pcm_parts.append(wf.readframes(wf.getnframes()))

        if not pcm_parts or sample_rate is None:
            return b""

        pcm_data = b"".join(pcm_parts)
        byte_rate = sample_rate * channels * sampwidth
        block_align = channels * sampwidth
        bits = sampwidth * 8
        data_size = len(pcm_data)
        file_size = data_size + 36

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", file_size, b"WAVE",
            b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits,
            b"data", data_size,
        )
        return header + pcm_data

    def save(self, path: str) -> None:
        """Save concatenated dialogue audio to a WAV file."""
        data = self.to_bytes()
        if data:
            with open(path, "wb") as f:
                f.write(data)

    def play(self) -> None:
        """Play concatenated audio using sounddevice."""
        from ._audio import play

        data = self.to_bytes()
        if data:
            play(data)
