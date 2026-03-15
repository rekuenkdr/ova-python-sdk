"""Unit tests for _models.py — Pydantic model parsing."""

import base64
import struct

from ova_sdk._models import (
    BatchTTSItem,
    BatchTTSResult,
    DialogueBatchResult,
    DialogueInput,
    Info,
    Settings,
    Transcription,
)
from ova_sdk._streaming import _build_wav


class TestInfo:
    def test_parse_minimal(self):
        info = Info(
            tts_engine="qwen3",
            language="en",
            supports_streaming=True,
            pcm_prebuffer_samples=9600,
            early_tts_decode=True,
        )
        assert info.tts_engine == "qwen3"
        assert info.voice is None

    def test_parse_with_extra_fields(self):
        """Extra fields from newer server versions should be ignored."""
        data = {
            "voice": "myvoice",
            "tts_engine": "qwen3",
            "language": "en",
            "supports_streaming": True,
            "pcm_prebuffer_samples": 9600,
            "early_tts_decode": True,
            "future_field": "should be ignored",
        }
        info = Info.model_validate(data)
        assert info.voice == "myvoice"
        assert info.language == "en"

    def test_defaults(self):
        info = Info(
            tts_engine="kokoro",
            language="es",
            supports_streaming=False,
            pcm_prebuffer_samples=0,
            early_tts_decode=False,
        )
        assert info.frontend_settings_disabled is False
        assert info.multimodal_disabled is False


class TestSettings:
    def test_parse_nested(self):
        data = {
            "current": {
                "language": "en",
                "tts_engine": "qwen3",
                "voice": "myvoice",
                "stream_format": "pcm",
                "system_prompt": "You are a helpful assistant.",
            },
            "profiles": {
                "en": {
                    "myvoice": {"prompt": "A friendly voice."},
                },
            },
            "default_prompts": {"en": "Default prompt."},
            "languages": ["en", "es"],
            "voices": {
                "en": [{"id": "myvoice", "name": "My Voice"}],
            },
        }
        settings = Settings.model_validate(data)
        assert settings.current.language == "en"
        assert settings.current.voice == "myvoice"
        assert len(settings.languages) == 2
        assert settings.voices["en"][0].id == "myvoice"
        assert settings.profiles["en"]["myvoice"].prompt == "A friendly voice."


class TestBatchTTSResult:
    def test_to_bytes_decodes_base64(self):
        # Create a real small WAV and encode it
        pcm = b"\x00\x01" * 50
        wav = _build_wav(pcm, sr=24000)
        b64 = base64.b64encode(wav).decode()

        result = BatchTTSResult(index=0, status="ok", audio_b64=b64)
        decoded = result.to_bytes()
        assert decoded == wav
        assert decoded[:4] == b"RIFF"

    def test_to_bytes_none_audio(self):
        result = BatchTTSResult(index=0, status="error", error="failed")
        assert result.to_bytes() == b""


class TestDialogueBatchResult:
    def test_to_bytes_concatenates(self):
        pcm_a = b"\x01\x00" * 100  # 200 bytes
        pcm_b = b"\x02\x00" * 100  # 200 bytes
        wav_a = _build_wav(pcm_a, sr=24000)
        wav_b = _build_wav(pcm_b, sr=24000)
        b64_a = base64.b64encode(wav_a).decode()
        b64_b = base64.b64encode(wav_b).decode()

        segments = [
            BatchTTSResult(index=0, status="ok", audio_b64=b64_a),
            BatchTTSResult(index=1, status="ok", audio_b64=b64_b),
        ]
        dialogue = DialogueBatchResult(segments)
        combined = dialogue.to_bytes()

        assert combined[:4] == b"RIFF"
        # Total PCM should be 400 bytes
        data_size = struct.unpack_from("<I", combined, 40)[0]
        assert data_size == 400

    def test_empty_segments(self):
        dialogue = DialogueBatchResult([])
        assert dialogue.to_bytes() == b""


class TestTranscription:
    def test_parse(self):
        t = Transcription(text="Hello world")
        assert t.text == "Hello world"


class TestDialogueInput:
    def test_fields(self):
        d = DialogueInput(text="Hi", voice_id="speaker_a")
        assert d.text == "Hi"
        assert d.voice_id == "speaker_a"


class TestBatchTTSItem:
    def test_defaults(self):
        item = BatchTTSItem(text="Test")
        assert item.voice is None
        assert item.language is None
