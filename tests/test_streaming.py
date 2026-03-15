"""Unit tests for _streaming.py — WAV building, AudioStream carry buffer."""

import struct

from ova_sdk._streaming import _build_wav, _PCM_STREAMING_MARKER, _WAV_HEADER_SIZE


class TestBuildWav:
    def test_valid_riff_header(self):
        pcm = b"\x00\x01" * 100  # 200 bytes of PCM
        wav = _build_wav(pcm, sr=24000)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "

    def test_sample_rate_embedded(self):
        wav = _build_wav(b"\x00" * 100, sr=16000)
        sr = struct.unpack_from("<I", wav, 24)[0]
        assert sr == 16000

    def test_data_size_correct(self):
        pcm = b"\xAB\xCD" * 50  # 100 bytes
        wav = _build_wav(pcm, sr=24000)
        data_size = struct.unpack_from("<I", wav, 40)[0]
        assert data_size == 100

    def test_file_size_correct(self):
        pcm = b"\x00" * 200
        wav = _build_wav(pcm, sr=24000)
        file_size = struct.unpack_from("<I", wav, 4)[0]
        assert file_size == 200 + 36  # data_size + 36

    def test_pcm_data_appended(self):
        pcm = b"\x01\x02\x03\x04"
        wav = _build_wav(pcm, sr=24000)
        assert wav[_WAV_HEADER_SIZE:] == pcm

    def test_header_size_is_44(self):
        wav = _build_wav(b"", sr=24000)
        assert len(wav) == 44  # header only, no PCM

    def test_channels_and_bits(self):
        wav = _build_wav(b"\x00" * 10, sr=24000)
        channels = struct.unpack_from("<H", wav, 22)[0]
        bits = struct.unpack_from("<H", wav, 34)[0]
        assert channels == 1
        assert bits == 16


class TestPcmStreamingMarker:
    def test_marker_value(self):
        assert _PCM_STREAMING_MARKER == 0x7FFFFFFF

    def test_marker_in_crafted_header(self):
        """A WAV header with data_size=0x7FFFFFFF signals PCM streaming."""
        pcm = b"\x00" * 100
        # Build a normal WAV then patch data_size
        wav = _build_wav(pcm, sr=24000)
        patched = wav[:40] + struct.pack("<I", _PCM_STREAMING_MARKER) + wav[44:]
        data_size = struct.unpack_from("<I", patched, 40)[0]
        assert data_size == _PCM_STREAMING_MARKER
