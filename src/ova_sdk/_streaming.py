"""AudioStream / AsyncAudioStream — wraps httpx streaming responses."""

from __future__ import annotations

import io
import struct
import time
from typing import Iterator, Optional

import httpx


_WAV_HEADER_SIZE = 44
_PCM_STREAMING_MARKER = 0x7FFFFFFF


class AudioStream:
    """Wraps a streaming HTTP response carrying audio data.

    Supports three modes (auto-detected from headers + WAV header):
    - PCM streaming: WAV header with data_size=0x7FFFFFFF, then raw int16 chunks
    - WAV streaming: each chunk is a complete WAV file
    - Non-streaming: single complete WAV (has Content-Length)
    """

    def __init__(self, response: httpx.Response, *, _request_time: float | None = None):
        self._response = response
        self._consumed = False
        self._raw_chunks: Optional[list[bytes]] = None
        self._sample_rate: Optional[int] = None
        self._is_pcm_streaming: Optional[bool] = None
        self._t_request = _request_time
        self._t_first_chunk: float | None = None
        self._t_done: float | None = None

    @property
    def sample_rate(self) -> int:
        """Sample rate from the WAV header. Available after first iteration or to_bytes()."""
        if self._sample_rate is None:
            self.to_bytes()
        return self._sample_rate

    @property
    def ttfb(self) -> float | None:
        """Seconds from request to first audio byte. Available after iteration starts."""
        if self._t_request is None or self._t_first_chunk is None:
            return None
        return self._t_first_chunk - self._t_request

    @property
    def elapsed(self) -> float | None:
        """Seconds from request to stream completion. Available after full consumption."""
        if self._t_request is None or self._t_done is None:
            return None
        return self._t_done - self._t_request

    def __iter__(self) -> Iterator[bytes]:
        """Iterate raw bytes chunks as they arrive from the server.

        Maintains a carry buffer so every yielded chunk has even byte length
        (required by int16 PCM consumers like sounddevice).
        """
        if self._consumed:
            if self._raw_chunks is not None:
                yield from self._raw_chunks
                return
            raise RuntimeError("AudioStream already consumed and not cached")

        self._consumed = True
        self._raw_chunks = []
        carry = b""
        for raw in self._response.iter_bytes():
            if self._t_first_chunk is None:
                self._t_first_chunk = time.perf_counter()
            chunk = carry + raw
            carry = b""
            if len(chunk) % 2:
                carry = chunk[-1:]
                chunk = chunk[:-1]
            if chunk:
                self._raw_chunks.append(chunk)
                yield chunk
        if carry:
            self._raw_chunks.append(carry)
            yield carry
        self._t_done = time.perf_counter()

    def _ensure_consumed(self):
        if not self._consumed:
            self._raw_chunks = []
            for chunk in self._response.iter_bytes():
                if self._t_first_chunk is None:
                    self._t_first_chunk = time.perf_counter()
                self._raw_chunks.append(chunk)
            self._t_done = time.perf_counter()
            self._consumed = True

    def to_bytes(self) -> bytes:
        """Materialize entire stream as a valid WAV file.

        For PCM streaming, reconstructs proper WAV with correct data_size.
        For WAV streaming, concatenates all PCM data into single WAV.
        For non-streaming, returns as-is.
        """
        self._ensure_consumed()
        raw = b"".join(self._raw_chunks)

        if len(raw) < _WAV_HEADER_SIZE:
            return raw

        # Parse WAV header
        data_size = struct.unpack_from("<I", raw, 40)[0]
        sr = struct.unpack_from("<I", raw, 24)[0]
        self._sample_rate = sr

        if data_size == _PCM_STREAMING_MARKER:
            # PCM streaming: strip header, rewrite with correct size
            self._is_pcm_streaming = True
            pcm_data = raw[_WAV_HEADER_SIZE:]
            return _build_wav(pcm_data, sr)

        # Check if Content-Length was present (non-streaming single WAV)
        content_length = self._response.headers.get("content-length")
        if content_length:
            self._is_pcm_streaming = False
            return raw

        # WAV streaming: each chunk is a complete WAV, concatenate PCM data
        self._is_pcm_streaming = False
        pcm_parts = []
        pos = 0
        while pos + _WAV_HEADER_SIZE <= len(raw):
            if raw[pos:pos+4] != b"RIFF":
                pos += 1
                continue
            chunk_file_size = struct.unpack_from("<I", raw, pos + 4)[0] + 8
            if pos + chunk_file_size > len(raw):
                break
            chunk_data_size = struct.unpack_from("<I", raw, pos + 40)[0]
            sr = struct.unpack_from("<I", raw, pos + 24)[0]
            self._sample_rate = sr
            pcm_parts.append(raw[pos + _WAV_HEADER_SIZE: pos + _WAV_HEADER_SIZE + chunk_data_size])
            pos += chunk_file_size

        if pcm_parts:
            return _build_wav(b"".join(pcm_parts), self._sample_rate)
        return raw

    def save(self, path: str) -> None:
        """Save audio to a WAV file."""
        with open(path, "wb") as f:
            f.write(self.to_bytes())

    def play(self) -> None:
        """Play audio using sounddevice (requires optional dependency)."""
        from ._audio import play
        play(self)

    def close(self):
        self._response.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self._response.close()
        except Exception:
            pass


class AsyncAudioStream:
    """Async version of AudioStream."""

    def __init__(self, response: httpx.Response, *, _request_time: float | None = None):
        self._response = response
        self._consumed = False
        self._raw_chunks: Optional[list[bytes]] = None
        self._sample_rate: Optional[int] = None
        self._t_request = _request_time
        self._t_first_chunk: float | None = None
        self._t_done: float | None = None

    @property
    def ttfb(self) -> float | None:
        """Seconds from request to first audio byte. Available after iteration starts."""
        if self._t_request is None or self._t_first_chunk is None:
            return None
        return self._t_first_chunk - self._t_request

    @property
    def elapsed(self) -> float | None:
        """Seconds from request to stream completion. Available after full consumption."""
        if self._t_request is None or self._t_done is None:
            return None
        return self._t_done - self._t_request

    async def __aiter__(self):
        """Iterate raw bytes chunks as they arrive from the server.

        Maintains a carry buffer so every yielded chunk has even byte length
        (required by int16 PCM consumers like sounddevice).
        """
        if self._consumed:
            if self._raw_chunks is not None:
                for chunk in self._raw_chunks:
                    yield chunk
                return
            raise RuntimeError("AsyncAudioStream already consumed and not cached")

        self._consumed = True
        self._raw_chunks = []
        carry = b""
        async for raw in self._response.aiter_bytes():
            if self._t_first_chunk is None:
                self._t_first_chunk = time.perf_counter()
            chunk = carry + raw
            carry = b""
            if len(chunk) % 2:
                carry = chunk[-1:]
                chunk = chunk[:-1]
            if chunk:
                self._raw_chunks.append(chunk)
                yield chunk
        if carry:
            self._raw_chunks.append(carry)
            yield carry
        self._t_done = time.perf_counter()

    async def _ensure_consumed(self):
        if not self._consumed:
            self._raw_chunks = []
            async for chunk in self._response.aiter_bytes():
                if self._t_first_chunk is None:
                    self._t_first_chunk = time.perf_counter()
                self._raw_chunks.append(chunk)
            self._t_done = time.perf_counter()
            self._consumed = True

    async def get_sample_rate(self) -> int:
        """Sample rate from the WAV header. Consumes the stream if not already done."""
        if self._sample_rate is None:
            await self.to_bytes()
        return self._sample_rate

    async def to_bytes(self) -> bytes:
        await self._ensure_consumed()
        raw = b"".join(self._raw_chunks)

        if len(raw) < _WAV_HEADER_SIZE:
            return raw

        data_size = struct.unpack_from("<I", raw, 40)[0]
        sr = struct.unpack_from("<I", raw, 24)[0]
        self._sample_rate = sr

        if data_size == _PCM_STREAMING_MARKER:
            pcm_data = raw[_WAV_HEADER_SIZE:]
            return _build_wav(pcm_data, sr)

        content_length = self._response.headers.get("content-length")
        if content_length:
            return raw

        pcm_parts = []
        pos = 0
        while pos + _WAV_HEADER_SIZE <= len(raw):
            if raw[pos:pos+4] != b"RIFF":
                pos += 1
                continue
            chunk_file_size = struct.unpack_from("<I", raw, pos + 4)[0] + 8
            if pos + chunk_file_size > len(raw):
                break
            chunk_data_size = struct.unpack_from("<I", raw, pos + 40)[0]
            sr = struct.unpack_from("<I", raw, pos + 24)[0]
            self._sample_rate = sr
            pcm_parts.append(raw[pos + _WAV_HEADER_SIZE: pos + _WAV_HEADER_SIZE + chunk_data_size])
            pos += chunk_file_size

        if pcm_parts:
            return _build_wav(b"".join(pcm_parts), self._sample_rate)
        return raw

    async def save(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(await self.to_bytes())

    async def play(self) -> None:
        """Play audio using sounddevice (requires optional dependency)."""
        from ._audio import play
        wav_bytes = await self.to_bytes()
        play(wav_bytes)

    async def close(self):
        await self._response.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


def _build_wav(pcm_data: bytes, sr: int, channels: int = 1, bits: int = 16) -> bytes:
    """Build a complete WAV file from raw PCM data."""
    byte_rate = sr * channels * (bits // 8)
    block_align = channels * (bits // 8)
    data_size = len(pcm_data)
    file_size = data_size + 36

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", file_size, b"WAVE",
        b"fmt ", 16, 1, channels, sr, byte_rate, block_align, bits,
        b"data", data_size,
    )
    return header + pcm_data
