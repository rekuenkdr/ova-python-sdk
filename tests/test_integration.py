"""Integration tests — require a running OVA server.

Run with: pytest -m integration
"""

from pathlib import Path

import pytest

from ova_sdk import OVA, OVAConnectionError, OVAServerNotReady
from ova_sdk._models import Info, Settings

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "tests"


@pytest.fixture(scope="module", autouse=True)
def _output_dir():
    """Ensure the output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def client():
    """Create a client and verify the server is reachable."""
    c = OVA()
    try:
        c.wait_until_ready(timeout=10, poll_interval=1)
    except (OVAServerNotReady, OVAConnectionError):
        pytest.skip("OVA server not available")
    yield c
    c.close()


@pytest.fixture(scope="module")
def voice(client):
    """Discover the first available voice, skip if none."""
    voices = client.voices()
    if not voices:
        pytest.skip("No voice profiles available")
    return voices[0]


@pytest.mark.integration
class TestHealth:
    def test_ready(self, client):
        assert client.ready() is True

    def test_info(self, client):
        info = client.info()
        assert isinstance(info, Info)
        assert info.tts_engine in ("qwen3", "kokoro")
        assert isinstance(info.language, str)

    def test_languages(self, client):
        langs = client.languages()
        assert isinstance(langs, list)
        assert len(langs) > 0

    def test_voices(self, client):
        voices = client.voices()
        assert isinstance(voices, list)


@pytest.mark.integration
class TestSettings:
    def test_get(self, client):
        settings = client.settings.get()
        assert isinstance(settings, Settings)
        assert isinstance(settings.current.language, str)


@pytest.mark.integration
class TestTTS:
    def test_generate(self, client, voice):
        audio = client.tts.generate("Test.", voice=voice)
        wav = audio.to_bytes()
        assert len(wav) > 44
        assert wav[:4] == b"RIFF"
        audio_path = OUTPUT_DIR / "tts_generate.wav"
        with open(audio_path, "wb") as f:
            f.write(wav)

    def test_invalid_voice_error(self, client):
        from ova_sdk import OVARequestError

        with pytest.raises(OVARequestError):
            audio = client.tts.generate("Test.", voice="nonexistent_voice_xyz_99")
            audio.to_bytes()


@pytest.mark.integration
class TestTranscription:
    def test_transcribe(self, client, voice):
        audio = client.tts.generate("Hello world.", voice=voice)
        wav = audio.to_bytes()
        audio_path = OUTPUT_DIR / "transcribe_input.wav"
        with open(audio_path, "wb") as f:
            f.write(wav)
        result = client.transcribe(wav)
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        text_path = OUTPUT_DIR / "transcribe_output.txt"
        text_path.write_text(result.text)
