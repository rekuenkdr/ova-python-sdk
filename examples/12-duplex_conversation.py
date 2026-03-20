"""Full-duplex hands-free voice conversation.

Unlike example 10 (interactive chat), which uses a push-to-talk loop
(record → ASR → chat → TTS → play), duplex mode runs everything
concurrently over a single WebSocket:

  mic audio (16kHz int16) ──► server VAD + ASR ──► LLM ──► TTS ──► speaker
                                   ▲                              │
                                   └── interrupt on barge-in ─────┘

The server handles voice-activity detection, turn-taking, and interruption.
The client streams mic audio in, tracks turn state, and manages playback —
stopping TTS immediately when the user barges in.

Requires: pip install ova-sdk[asr] sounddevice numpy
Usage:    python 12-duplex_conversation.py [--language en] [--voice af_heart]

Tip: use headphones to avoid echo feedback (there is no echo cancellation).
"""

import argparse
import signal
import sys
import threading
import time

import numpy as np
import sounddevice as sd

from ova_sdk import OVA, DuplexEventHandler


# ── Shared playback state ────────────────────────────────────

state = "idle"  # idle | recording | thinking | playing
accepting_audio = False

out_buf: list[bytes] = []
out_lock = threading.Lock()

speaker_stream: sd.OutputStream | None = None
speaker_ready = threading.Event()
tts_sample_rate = 24000

# Timing
_t_turn_end: float = 0.0   # when user finished speaking (final transcript)
_t_thinking: float = 0.0   # when bot.thinking received
_t_first_audio = False      # whether first audio chunk has been reported this turn


def clear_buffer():
    """Drop all queued TTS audio so playback stops immediately."""
    with out_lock:
        out_buf.clear()


# ── Event callbacks ──────────────────────────────────────────

def on_session_started(sample_rate: int) -> None:
    global tts_sample_rate
    tts_sample_rate = sample_rate
    speaker_ready.set()
    print(f"  Session started (TTS sample rate: {sample_rate} Hz)")
    print("  Speak freely — talk over the bot to interrupt.\n")


def on_session_ended() -> None:
    global state, accepting_audio
    accepting_audio = False
    clear_buffer()
    state = "idle"
    print("\n  Session ended.")


def on_vad(speech: bool) -> None:
    global state, accepting_audio, _t_turn_end
    if speech:
        if state == "playing":
            # Barge-in: user started speaking while bot is playing
            accepting_audio = False
            clear_buffer()
        state = "recording"
        print("  [listening]", flush=True)
    else:
        # User stopped speaking — this is the real end-of-speech moment
        _t_turn_end = time.monotonic()


def on_transcript(text: str, is_final: bool) -> None:
    global state
    if is_final:
        print(f"  You: {text}")
        state = "thinking"
    else:
        print(f"\r  ... {text}    ", end="", flush=True)


def on_bot_thinking() -> None:
    global state, _t_thinking
    _t_thinking = time.monotonic()
    state = "thinking"
    print("  Assistant is thinking...")


def on_bot_speaking() -> None:
    global state, accepting_audio, _t_first_audio
    accepting_audio = True
    _t_first_audio = False
    state = "playing"
    llm_ms = (time.monotonic() - _t_thinking) * 1000 if _t_thinking else 0
    print(f"  Assistant is speaking... (LLM: {llm_ms:.0f}ms)")


def on_bot_idle() -> None:
    global state, accepting_audio
    accepting_audio = False
    state = "idle"
    print("  [idle — waiting for you to speak]")


def on_bot_interrupted() -> None:
    global state, accepting_audio
    accepting_audio = False
    clear_buffer()
    state = "recording"
    print("  [interrupted]")


def on_audio(pcm: bytes) -> None:
    global _t_first_audio
    if accepting_audio:
        if not _t_first_audio:
            _t_first_audio = True
            if _t_turn_end:
                total_ms = (time.monotonic() - _t_turn_end) * 1000
                print(f"  First audio: {total_ms:.0f}ms after end of speech")
        with out_lock:
            out_buf.append(pcm)


def on_error(message: str) -> None:
    print(f"  Error: {message}", file=sys.stderr)


# ── Audio I/O ────────────────────────────────────────────────

def speaker_callback(outdata, frames, time_info, status):
    """Pull queued TTS audio into the speaker buffer."""
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
        if chunk:
            available = np.frombuffer(chunk, dtype=np.int16)
            outdata[: len(available), 0] = available
            outdata[len(available) :, 0] = 0
        else:
            outdata[:] = 0


def start_speaker():
    """Wait for TTS sample rate, then open the speaker stream."""
    global speaker_stream
    speaker_ready.wait(timeout=10)
    speaker_stream = sd.OutputStream(
        samplerate=tts_sample_rate,
        channels=1,
        dtype="int16",
        callback=speaker_callback,
        blocksize=2048,
    )
    speaker_stream.start()


# ── Main ─────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Full-duplex hands-free voice conversation")
parser.add_argument("--language", default=None, help="Override server language (e.g. en, es, fr)")
parser.add_argument("--voice", default=None, help="Override server voice (e.g. af_heart)")
args = parser.parse_args()

client = OVA()

print("Waiting for OVA server...")
client.wait_until_ready()

info = client.info()
LANGUAGE = args.language or info.language
VOICE = args.voice or info.voice

voices = client.voices(language=LANGUAGE)
if not voices:
    print(f"No voices found for '{LANGUAGE}'. Create a voice profile first — see QUICKSTART.md")
    sys.exit(1)
if VOICE not in voices:
    print(f"Voice '{VOICE}' not available for '{LANGUAGE}'. Available: {voices}")
    VOICE = voices[0]

# Clear conversation history from previous runs
client.settings.reload_prompt(clear_history=True)

# Warmup: prime torch.compile cache
print("Warming up torch.compile cache...")
_ = client.tts.generate("Warmup.", voice=VOICE, language=LANGUAGE).to_bytes()
print("Warmup done.\n")

print(f"Voice: {VOICE}, Language: {LANGUAGE}")
print("Opening duplex session... (Ctrl+C to quit)\n")

handler = DuplexEventHandler(
    on_session_started=on_session_started,
    on_session_ended=on_session_ended,
    on_vad=on_vad,
    on_transcript=on_transcript,
    on_bot_thinking=on_bot_thinking,
    on_bot_speaking=on_bot_speaking,
    on_bot_idle=on_bot_idle,
    on_bot_interrupted=on_bot_interrupted,
    on_audio=on_audio,
    on_error=on_error,
)

session = client.duplex.connect(
    language=LANGUAGE,
    voice=VOICE,
    handler=handler,
)

# Mic → server: stream PCM int16 at 16kHz
def mic_callback(indata, frames, time_info, status):
    if not session._closed.is_set():
        session.send_audio(indata.copy())

mic_stream = sd.InputStream(
    samplerate=16000,
    channels=1,
    dtype="int16",
    callback=mic_callback,
    blocksize=4096,
)
mic_stream.start()

# Speaker starts in a background thread (waits for session.started)
threading.Thread(target=start_speaker, daemon=True).start()


# ── Graceful shutdown on Ctrl+C ──────────────────────────────
def _sigint(sig, frame):
    print("\nClosing session...")
    session.close()


signal.signal(signal.SIGINT, _sigint)

try:
    session.run()
except KeyboardInterrupt:
    pass
finally:
    mic_stream.stop()
    mic_stream.close()
    if speaker_stream is not None:
        speaker_stream.stop()
        speaker_stream.close()
    client.close()
    print("Bye!")
