"""Full-duplex voice conversation via WebSocket."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import quote

import numpy as np

from ._base import ws_url
from ._errors import OVAConnectionError


@dataclass
class DuplexEventHandler:
    """Callbacks for duplex session events. All fields are optional."""

    on_session_started: Optional[Callable[[int], None]] = None
    on_session_ended: Optional[Callable[[], None]] = None
    on_vad: Optional[Callable[[bool], None]] = None
    on_transcript: Optional[Callable[[str, bool], None]] = None
    on_bot_thinking: Optional[Callable[[], None]] = None
    on_bot_speaking: Optional[Callable[[], None]] = None
    on_bot_idle: Optional[Callable[[], None]] = None
    on_bot_interrupted: Optional[Callable[[], None]] = None
    on_error: Optional[Callable[[str], None]] = None
    on_audio: Optional[Callable[[bytes], None]] = None


def _duplex_ws_url(
    base_url: str,
    api_key: Optional[str],
    language: Optional[str] = None,
    voice: Optional[str] = None,
) -> str:
    """Build the /v1/duplex WebSocket URL with query parameters."""
    url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    full = f"{url}/v1/duplex"
    params: list[str] = []
    if language:
        params.append(f"language={quote(language, safe='')}")
    if voice:
        params.append(f"voice={quote(voice, safe='')}")
    if api_key:
        params.append(f"api_key={quote(api_key, safe='')}")
    if params:
        full += "?" + "&".join(params)
    return full


def _dispatch(handler: DuplexEventHandler, msg: dict) -> None:
    """Dispatch a parsed JSON message to the appropriate handler callback."""
    t = msg.get("type", "")
    if t == "session.started":
        if handler.on_session_started:
            handler.on_session_started(msg.get("sample_rate", 24000))
    elif t == "session.ended":
        if handler.on_session_ended:
            handler.on_session_ended()
    elif t == "vad":
        if handler.on_vad:
            handler.on_vad(msg.get("speech", False))
    elif t == "transcript":
        if handler.on_transcript:
            handler.on_transcript(msg.get("text", ""), msg.get("is_final", False))
    elif t == "bot.thinking":
        if handler.on_bot_thinking:
            handler.on_bot_thinking()
    elif t == "bot.speaking":
        if handler.on_bot_speaking:
            handler.on_bot_speaking()
    elif t == "bot.idle":
        if handler.on_bot_idle:
            handler.on_bot_idle()
    elif t == "bot.interrupted":
        if handler.on_bot_interrupted:
            handler.on_bot_interrupted()
    elif t == "error":
        if handler.on_error:
            handler.on_error(msg.get("message", "unknown error"))


# ---------------------------------------------------------------------------
# Sync session
# ---------------------------------------------------------------------------


class DuplexSession:
    """Synchronous full-duplex voice session over WebSocket."""

    def __init__(self, ws, handler: DuplexEventHandler):
        self._ws = ws
        self._handler = handler
        self._closed = threading.Event()
        self._tts_sample_rate: int = 24000

    # -- receive loop ------------------------------------------------------

    def run(self) -> None:
        """Blocking receive loop. Dispatches messages to handler callbacks.

        Returns when the server closes the connection or :meth:`close` is called.
        """
        try:
            for message in self._ws:
                if self._closed.is_set():
                    break
                if isinstance(message, bytes):
                    if self._handler.on_audio:
                        self._handler.on_audio(message)
                else:
                    data = json.loads(message)
                    if data.get("type") == "session.started":
                        self._tts_sample_rate = data.get("sample_rate", 24000)
                    _dispatch(self._handler, data)
                    if data.get("type") == "session.ended":
                        break
        except Exception:
            if not self._closed.is_set():
                raise

    # -- send helpers -------------------------------------------------------

    def send_audio(self, pcm) -> None:
        """Send a PCM int16 audio frame (bytes or ndarray)."""
        if isinstance(pcm, np.ndarray):
            pcm = pcm.astype(np.int16).tobytes()
        self._ws.send(pcm)

    def send_text(self, text: str, image: Optional[str] = None) -> None:
        """Send a text message (optionally with a base64 image)."""
        msg: dict = {"type": "session.text", "text": text}
        if image is not None:
            msg["image"] = image
        self._ws.send(json.dumps(msg))

    def send_image(self, data: str) -> None:
        """Send a base64-encoded image for the next voice turn."""
        self._ws.send(json.dumps({"type": "session.image", "image": data}))

    def send_config(
        self,
        language: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> None:
        """Update session configuration (language and/or voice)."""
        msg: dict = {"type": "session.config"}
        if language is not None:
            msg["language"] = language
        if voice is not None:
            msg["voice"] = voice
        self._ws.send(json.dumps(msg))

    def interrupt(self) -> None:
        """Request the bot to stop speaking."""
        self._ws.send(json.dumps({"type": "session.interrupt"}))

    def close(self) -> None:
        """Gracefully end the session."""
        self._closed.set()
        try:
            self._ws.send(json.dumps({"type": "session.end"}))
        except Exception:
            pass
        try:
            self._ws.close()
        except Exception:
            pass

    # -- convenience: run with local audio hardware -------------------------

    def run_with_audio(self, mic_rate: int = 16000) -> None:
        """Open mic + speaker and run the receive loop.

        Microphone audio is captured via sounddevice and sent to the server.
        Received audio is played through the default output device.
        Blocks until the session ends.
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "run_with_audio() requires 'sounddevice'. "
                "Install with: pip install sounddevice"
            )

        out_buf: list[bytes] = []
        out_lock = threading.Lock()

        # Patch on_audio to buffer for playback
        original_on_audio = self._handler.on_audio

        def _on_audio(pcm: bytes) -> None:
            with out_lock:
                out_buf.append(pcm)
            if original_on_audio:
                original_on_audio(pcm)

        self._handler.on_audio = _on_audio

        # Wait for session.started to learn sample rate
        original_on_started = self._handler.on_session_started
        started_event = threading.Event()

        def _on_started(sr: int) -> None:
            self._tts_sample_rate = sr
            started_event.set()
            if original_on_started:
                original_on_started(sr)

        self._handler.on_session_started = _on_started

        # Mic callback — sends audio to server
        def _mic_callback(indata, frames, time_info, status):
            if not self._closed.is_set():
                self.send_audio(indata.copy())

        # Speaker callback — plays received audio
        def _speaker_callback(outdata, frames, time_info, status):
            needed = frames * 2  # int16 = 2 bytes per sample
            chunk = b""
            with out_lock:
                while out_buf and len(chunk) < needed:
                    chunk += out_buf.pop(0)
            if len(chunk) >= needed:
                outdata[:] = np.frombuffer(chunk[:needed], dtype=np.int16).reshape(-1, 1)
                leftover = chunk[needed:]
                if leftover:
                    with out_lock:
                        out_buf.insert(0, leftover)
            else:
                # Pad with what we have + silence
                if chunk:
                    available = np.frombuffer(chunk, dtype=np.int16)
                    outdata[: len(available), 0] = available
                    outdata[len(available) :, 0] = 0
                else:
                    outdata[:] = 0

        mic_stream = sd.InputStream(
            samplerate=mic_rate,
            channels=1,
            dtype="int16",
            callback=_mic_callback,
            blocksize=4096,
        )
        mic_stream.start()

        # Start speaker after we know the TTS sample rate
        speaker_stream = None

        def _start_speaker():
            nonlocal speaker_stream
            started_event.wait(timeout=10)
            speaker_stream = sd.OutputStream(
                samplerate=self._tts_sample_rate,
                channels=1,
                dtype="int16",
                callback=_speaker_callback,
                blocksize=2048,
            )
            speaker_stream.start()

        speaker_thread = threading.Thread(target=_start_speaker, daemon=True)
        speaker_thread.start()

        try:
            self.run()
        finally:
            mic_stream.stop()
            mic_stream.close()
            if speaker_stream is not None:
                speaker_stream.stop()
                speaker_stream.close()
            # Restore original callbacks
            self._handler.on_audio = original_on_audio
            self._handler.on_session_started = original_on_started

    # -- context manager ----------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Async session
# ---------------------------------------------------------------------------


class AsyncDuplexSession:
    """Asynchronous full-duplex voice session over WebSocket."""

    def __init__(self, ws, handler: DuplexEventHandler):
        self._ws = ws
        self._handler = handler
        self._closed = False
        self._tts_sample_rate: int = 24000

    # -- receive loop ------------------------------------------------------

    async def run(self) -> None:
        """Awaitable receive loop. Dispatches messages to handler callbacks."""
        try:
            async for message in self._ws:
                if self._closed:
                    break
                if isinstance(message, bytes):
                    if self._handler.on_audio:
                        self._handler.on_audio(message)
                else:
                    data = json.loads(message)
                    if data.get("type") == "session.started":
                        self._tts_sample_rate = data.get("sample_rate", 24000)
                    _dispatch(self._handler, data)
                    if data.get("type") == "session.ended":
                        break
        except Exception:
            if not self._closed:
                raise

    # -- send helpers -------------------------------------------------------

    async def send_audio(self, pcm) -> None:
        if isinstance(pcm, np.ndarray):
            pcm = pcm.astype(np.int16).tobytes()
        await self._ws.send(pcm)

    async def send_text(self, text: str, image: Optional[str] = None) -> None:
        msg: dict = {"type": "session.text", "text": text}
        if image is not None:
            msg["image"] = image
        await self._ws.send(json.dumps(msg))

    async def send_image(self, data: str) -> None:
        await self._ws.send(json.dumps({"type": "session.image", "image": data}))

    async def send_config(
        self,
        language: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> None:
        msg: dict = {"type": "session.config"}
        if language is not None:
            msg["language"] = language
        if voice is not None:
            msg["voice"] = voice
        await self._ws.send(json.dumps(msg))

    async def interrupt(self) -> None:
        await self._ws.send(json.dumps({"type": "session.interrupt"}))

    async def close(self) -> None:
        self._closed = True
        try:
            await self._ws.send(json.dumps({"type": "session.end"}))
        except Exception:
            pass
        try:
            await self._ws.close()
        except Exception:
            pass

    # -- convenience: run with local audio hardware -------------------------

    async def run_with_audio(self, mic_rate: int = 16000) -> None:
        """Open mic + speaker and run the receive loop concurrently.

        Uses sounddevice for audio I/O and asyncio for concurrency.
        """
        import asyncio

        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "run_with_audio() requires 'sounddevice'. "
                "Install with: pip install sounddevice"
            )

        out_buf: list[bytes] = []
        out_lock = threading.Lock()
        loop = asyncio.get_running_loop()

        original_on_audio = self._handler.on_audio

        def _on_audio(pcm: bytes) -> None:
            with out_lock:
                out_buf.append(pcm)
            if original_on_audio:
                original_on_audio(pcm)

        self._handler.on_audio = _on_audio

        original_on_started = self._handler.on_session_started
        started_event = asyncio.Event()

        def _on_started(sr: int) -> None:
            self._tts_sample_rate = sr
            loop.call_soon_threadsafe(started_event.set)
            if original_on_started:
                original_on_started(sr)

        self._handler.on_session_started = _on_started

        # Mic sends via asyncio from the callback thread
        def _mic_callback(indata, frames, time_info, status):
            if not self._closed:
                data = indata.copy().tobytes()
                asyncio.run_coroutine_threadsafe(self._ws.send(data), loop)

        def _speaker_callback(outdata, frames, time_info, status):
            needed = frames * 2
            chunk = b""
            with out_lock:
                while out_buf and len(chunk) < needed:
                    chunk += out_buf.pop(0)
            if len(chunk) >= needed:
                outdata[:] = np.frombuffer(chunk[:needed], dtype=np.int16).reshape(-1, 1)
                leftover = chunk[needed:]
                if leftover:
                    with out_lock:
                        out_buf.insert(0, leftover)
            else:
                if chunk:
                    available = np.frombuffer(chunk, dtype=np.int16)
                    outdata[: len(available), 0] = available
                    outdata[len(available) :, 0] = 0
                else:
                    outdata[:] = 0

        mic_stream = sd.InputStream(
            samplerate=mic_rate,
            channels=1,
            dtype="int16",
            callback=_mic_callback,
            blocksize=4096,
        )
        mic_stream.start()

        await started_event.wait()
        speaker_stream = sd.OutputStream(
            samplerate=self._tts_sample_rate,
            channels=1,
            dtype="int16",
            callback=_speaker_callback,
            blocksize=2048,
        )
        speaker_stream.start()

        try:
            await self.run()
        finally:
            mic_stream.stop()
            mic_stream.close()
            speaker_stream.stop()
            speaker_stream.close()
            self._handler.on_audio = original_on_audio
            self._handler.on_session_started = original_on_started

    # -- context manager ----------------------------------------------------

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


# ---------------------------------------------------------------------------
# Resource factories
# ---------------------------------------------------------------------------


class DuplexResource:
    """Factory for sync duplex sessions."""

    def __init__(self, base_url: str, api_key: Optional[str]):
        self._base_url = base_url
        self._api_key = api_key

    def connect(
        self,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        handler: Optional[DuplexEventHandler] = None,
    ) -> DuplexSession:
        """Open a full-duplex WebSocket session."""
        try:
            from websockets.sync.client import connect
        except ImportError:
            raise ImportError(
                "Duplex sessions require 'websockets'. "
                "Install with: pip install ova-sdk[asr]"
            )

        url = _duplex_ws_url(self._base_url, self._api_key, language, voice)
        if handler is None:
            handler = DuplexEventHandler()
        try:
            ws = connect(url)
        except Exception as e:
            raise OVAConnectionError(f"WebSocket connection failed: {e}") from e
        return DuplexSession(ws, handler)


class AsyncDuplexResource:
    """Factory for async duplex sessions."""

    def __init__(self, base_url: str, api_key: Optional[str]):
        self._base_url = base_url
        self._api_key = api_key

    async def connect(
        self,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        handler: Optional[DuplexEventHandler] = None,
    ) -> AsyncDuplexSession:
        """Open a full-duplex WebSocket session."""
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            raise ImportError(
                "Duplex sessions require 'websockets'. "
                "Install with: pip install ova-sdk[asr]"
            )

        url = _duplex_ws_url(self._base_url, self._api_key, language, voice)
        if handler is None:
            handler = DuplexEventHandler()
        try:
            ws = await connect(url)
        except Exception as e:
            raise OVAConnectionError(f"WebSocket connection failed: {e}") from e
        return AsyncDuplexSession(ws, handler)
