# OVA SDK

Python SDK for the [OVA](https://github.com/rekuenkdr/ova) voice assistant server — text-to-speech, chat, full-duplex voice conversation, transcription, and settings in a few lines of code.

## Installation

```bash
pip install ova-sdk
```

**Linux only** — audio playback requires the PortAudio system library:

```bash
sudo apt-get install libportaudio2   # Debian/Ubuntu
```

## Quick Start

```python
from ova_sdk import OVA

client = OVA()
client.wait_until_ready()

audio = client.chat.send_text("Tell me a joke")
audio.play()
```

> **Note:** You must create at least one voice profile before using TTS or chat features. See [QUICKSTART.md](../QUICKSTART.md) for setup instructions.

## What You Can Do

### Text-to-Speech

Turn text into speech. Override voice and language per request without changing session state.

```python
audio = client.tts.generate("Hello, world!")
audio.play()

# Use a specific voice for this request only
audio = client.tts.generate("Hola, mundo!", voice="myvoice", language="es")
audio.save("hola.wav")
```

### Chat

Send text (or an image) to an LLM and get a spoken answer back.

```python
audio = client.chat.send_text("What's the weather like today?")
audio.play()

# Send a voice recording instead
audio = client.chat.send_audio("question.wav")
audio.play()
```

### Speech-to-Text

Transcribe an audio file in one call.

```python
result = client.transcribe("recording.wav")
print(result.text)
```

### Streaming ASR

Real-time transcription over WebSocket — send audio chunks, get partial transcripts back.

```python
stream = client.asr.stream()
stream.send(audio_chunk)          # numpy float32 array
partial = stream.receive()        # partial transcript
final_text = stream.finish()      # finalize and get full text
stream.close()
```

### Full-Duplex Voice Conversation

Hands-free, real-time voice conversation over a single WebSocket. The server handles VAD, turn-taking, and interruption — the client streams mic audio in and plays bot audio out.

```python
from ova_sdk import OVA, DuplexEventHandler

client = OVA()
client.wait_until_ready()

handler = DuplexEventHandler(
    on_session_started=lambda sr: print(f"Session started ({sr} Hz)"),
    on_vad=lambda speech: print("[listening]" if speech else ""),
    on_transcript=lambda text, final: print(f"You: {text}" if final else ""),
    on_bot_thinking=lambda: print("Thinking..."),
    on_bot_speaking=lambda: print("Speaking..."),
    on_bot_idle=lambda: print("[idle]"),
    on_bot_interrupted=lambda: print("[interrupted]"),
    on_error=lambda msg: print(f"Error: {msg}"),
)

session = client.duplex.connect(language="en", voice="af_heart", handler=handler)
session.run_with_audio()  # blocks — streams mic in, plays audio out
```

For production use with proper barge-in handling (stopping TTS playback when the user interrupts), see [`examples/12-duplex_conversation.py`](examples/12-duplex_conversation.py) which manages audio I/O directly with state tracking and buffer clearing.

**Session methods:**

```python
session.send_text("Hello")           # Send text as if user spoke it
session.send_image(base64_data)      # Queue image for next turn
session.send_config(voice="other")   # Switch voice/language at runtime
session.interrupt()                  # Stop bot speech
session.close()                      # End session
```

### Batch TTS

Synthesize many texts in a single request.

```python
from ova_sdk.models import BatchTTSItem

items = [
    BatchTTSItem(text="First sentence"),
    BatchTTSItem(text="Second sentence"),
    BatchTTSItem(text="Third sentence", voice="myvoice"),
]
results = client.tts.batch_generate(items)

for r in results:
    r.play()
```

### Multi-Speaker Dialogue

Different voices in a conversation, batched into one request.

```python
from ova_sdk.models import DialogueInput

lines = [
    DialogueInput(text="How are you?", voice_id="speaker_a"),
    DialogueInput(text="I'm great, thanks!", voice_id="speaker_b"),
]
dialogue = client.dialogue.generate(lines)
dialogue.play()  # plays all segments in order
```

### Server Settings

Change voice, language, or TTS engine on the fly.

```python
settings = client.settings.get()
print(settings.current.voice)

client.settings.update(voice="myvoice", language="en")
```

## Server Endpoints

The SDK wraps these OVA server endpoints:

| SDK Method | HTTP | Endpoint | What it does |
|---|---|---|---|
| `client.tts.generate()` | POST | `/v1/text-to-speech` | Synthesize text to speech |
| `client.tts.batch_generate()` | POST | `/v1/text-to-speech/batch` | Batch synthesize multiple texts |
| `client.dialogue.generate()` | POST | `/v1/text-to-speech/batch` | Multi-speaker dialogue synthesis |
| `client.chat.send_text()` | POST | `/v1/chat` | Text → LLM → spoken response |
| `client.chat.send_audio()` | POST | `/v1/chat/audio` | Audio → LLM → spoken response |
| `client.transcribe()` | POST | `/v1/speech-to-text` | One-shot transcription |
| `client.asr.stream()` | WS | `/v1/speech-to-text/stream` | Real-time streaming ASR |
| `client.duplex.connect()` | WS | `/v1/duplex` | Full-duplex voice conversation |
| `client.info()` | GET | `/v1/info` | Server configuration |
| `client.settings.get()` | GET | `/v1/settings` | Current settings |
| `client.settings.update()` | POST | `/v1/settings` | Update settings |
| `client.settings.reload_prompt()` | POST | `/v1/settings/prompt` | Update system prompt |
| `client.ready()` | GET | `/v1/health` | Health check |

## Client Configuration

```python
from ova_sdk import OVA

client = OVA(
    base_url="http://localhost:5173",   # OVA server URL
    api_key="sk-...",                   # Optional API key
    timeout=120.0,                      # Request timeout in seconds
)
```

| Parameter  | Env Variable   | Default                  |
|------------|----------------|--------------------------|
| `base_url` | `OVA_BASE_URL` | `http://localhost:5173`  |
| `api_key`  | `OVA_API_KEY`  | `None`                   |
| `timeout`  | —              | `120.0`                  |

Constructor arguments take precedence over environment variables.

### Readiness Check

The server needs time to load models. Block until ready:

```python
client.wait_until_ready(timeout=120, poll_interval=2)
```

Or check without blocking:

```python
if client.ready():
    print("Server is up")
```

### Server Info

```python
info = client.info()
print(info.tts_engine)          # "qwen3"
print(info.language)            # "en"
print(info.voice)               # "myvoice"
print(info.supports_streaming)  # True
```

### Discovery

```python
languages = client.languages()    # ["en", "es", "fr"]
voices = client.voices()          # ["myvoice", ...]
voices = client.voices(language="en")  # filter by language
```

## Async Usage

Every method has an async counterpart via `AsyncOVA`:

```python
from ova_sdk import AsyncOVA

async with AsyncOVA() as client:
    await client.wait_until_ready()
    audio = await client.chat.send_text("Hello from async")
    await audio.save("output.wav")
```

Use `await audio.get_sample_rate()` instead of the sync `audio.sample_rate` property.

## AudioStream

All TTS and Chat methods return an `AudioStream` (or `AsyncAudioStream` for async). It supports:

- **`.play()`** — play the audio through your speakers
- **`.save(path)`** — save to a WAV file
- **`.to_bytes()`** — get the full WAV as bytes
- **Iteration** — `for chunk in audio:` yields raw byte chunks as they arrive
- **`.ttfb`** — time-to-first-byte in seconds (available after iteration starts)
- **`.elapsed`** — total elapsed time in seconds (available after full consumption)
- **`.sample_rate`** — audio sample rate (sync only; use `await audio.get_sample_rate()` for async)

Standalone utilities are also available:

```python
from ova_sdk import play, save

play(wav_bytes)
save(wav_bytes, "output.wav")
```

## Error Handling

```
OVAError
├── OVAAuthenticationError  # API key is missing or invalid (401)
├── OVAConnectionError      # Cannot reach the server
├── OVAServerNotReady       # Server still warming up (wait_until_ready timeout)
├── OVARequestError         # HTTP 4xx/5xx (has .status_code attribute)
└── OVATimeoutError         # Request timed out
```

All exceptions are importable from `ova_sdk`.

## Examples

Ready-to-run scripts in [`examples/`](examples/):

| Script | Description |
|--------|-------------|
| [`01-quickstart.py`](examples/01-quickstart.py) | Discover the server, pick a voice, and speak |
| [`02-tts_voices.py`](examples/02-tts_voices.py) | Same text in every voice for the current language |
| [`03-multilingual_tts.py`](examples/03-multilingual_tts.py) | Generate speech across multiple languages |
| [`04-streaming_playback.py`](examples/04-streaming_playback.py) | Real-time streaming playback with TTFB measurement |
| [`05-batch_tts.py`](examples/05-batch_tts.py) | Batch TTS: synthesize multiple texts in one call |
| [`06-text_to_dialogue.py`](examples/06-text_to_dialogue.py) | Multi-speaker dialogue via batch TTS |
| [`07-chat_conversation.py`](examples/07-chat_conversation.py) | Multi-turn chat with voice switching |
| [`08-settings_and_info.py`](examples/08-settings_and_info.py) | Server inspection, voice discovery, and configuration |
| [`09-transcribe_audio.py`](examples/09-transcribe_audio.py) | Standalone speech-to-text with text file output |
| [`10-interactive_chat.py`](examples/10-interactive_chat.py) | Interactive voice chat — speak into the mic, hear the reply |
| [`11-error_handling.py`](examples/11-error_handling.py) | Proper error handling patterns for every exception type |
| [`12-duplex_conversation.py`](examples/12-duplex_conversation.py) | Full-duplex voice conversation with barge-in and latency measurement |

## API Reference

### Client

| Method | Returns | Description |
|--------|---------|-------------|
| `OVA(*, base_url, api_key, timeout)` | `OVA` | Create sync client |
| `AsyncOVA(*, base_url, api_key, timeout)` | `AsyncOVA` | Create async client |
| `.info()` | `Info` | Pipeline configuration |
| `.languages()` | `list[str]` | Available language codes |
| `.voices(language=None)` | `list[str]` | Available voice IDs (optionally filtered by language) |
| `.ready()` | `bool` | True if server is warmed up |
| `.wait_until_ready(timeout=120, poll_interval=2)` | `None` | Block until ready |
| `.transcribe(audio, *, language=None)` | `Transcription` | Speech-to-text (one-shot, no LLM) |
| `.close()` | `None` | Release HTTP connection |

### TTS

| Method | Returns | Description |
|--------|---------|-------------|
| `.tts.generate(text, *, voice=None, language=None)` | `AudioStream` | Synthesize text to speech (non-sticky) |
| `.tts.batch_generate(items)` | `list[BatchTTSResult]` | Batch-synthesize multiple texts in one call |
| `.tts.batch_stream(items)` | `Iterator[BatchTTSResult]` | Stream batch results as they complete |

### Dialogue

| Method | Returns | Description |
|--------|---------|-------------|
| `.dialogue.generate(inputs, *, language=None)` | `DialogueBatchResult` | Multi-speaker batch synthesis |
| `.dialogue.batch_generate(inputs, *, language=None)` | `DialogueBatchResult` | Multi-speaker batch synthesis (alias) |

### Chat

| Method | Returns | Description |
|--------|---------|-------------|
| `.chat.send_text(text, *, voice=None, language=None, image=None)` | `AudioStream` | Text (+ optional image) through LLM → audio (sticky) |
| `.chat.send_audio(audio, *, voice=None, language=None)` | `AudioStream` | WAV audio through LLM → audio (sticky) |

`audio` accepts `str`, `Path`, or `bytes`. `image` accepts `str`, `Path`, or data URL.
`voice` and `language` are optional overrides. On TTS they're non-sticky (per-request only). On Chat they're sticky (change session state).

### Settings

| Method | Returns | Description |
|--------|---------|-------------|
| `.settings.get()` | `Settings` | Current config, available profiles, languages, voices |
| `.settings.update(*, language, tts_engine, voice, stream_format)` | `SettingsUpdateResponse` | Update server settings |
| `.settings.reload_prompt(*, language, profile, prompt, clear_history=False)` | `ReloadPromptResponse` | Update system prompt |

### ASR

| Method | Returns | Description |
|--------|---------|-------------|
| `.asr.stream()` | `ASRStream` | Open WebSocket for streaming ASR |

**ASRStream / AsyncASRStream:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.send(audio)` | `None` | Send float32 numpy audio chunk |
| `.receive()` | `dict` | Receive partial/final transcript |
| `.finish()` | `str` | End stream, get final transcript |
| `.close()` | `None` | Close WebSocket |

### Duplex

| Method | Returns | Description |
|--------|---------|-------------|
| `.duplex.connect(language=None, voice=None, handler=None)` | `DuplexSession` | Open full-duplex WebSocket session |

**DuplexSession / AsyncDuplexSession:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.run()` | `None` | Blocking receive loop (dispatches events to handler) |
| `.run_with_audio(mic_rate=16000)` | `None` | Convenience: open mic + speaker and run receive loop |
| `.send_audio(pcm)` | `None` | Send PCM int16 audio (bytes or ndarray) |
| `.send_text(text, image=None)` | `None` | Send text as if user spoke it |
| `.send_image(data)` | `None` | Queue base64 image for next turn |
| `.send_config(language=None, voice=None)` | `None` | Update language/voice at runtime |
| `.interrupt()` | `None` | Request the bot to stop speaking |
| `.close()` | `None` | Gracefully end the session |

**DuplexEventHandler callbacks (all optional):**

| Callback | Signature | When |
|----------|-----------|------|
| `on_session_started` | `(sample_rate: int)` | Session opened, TTS sample rate declared |
| `on_session_ended` | `()` | Session closed |
| `on_vad` | `(speech: bool)` | Voice activity detected / ended |
| `on_transcript` | `(text: str, is_final: bool)` | Partial or final ASR transcript |
| `on_bot_thinking` | `()` | LLM processing started |
| `on_bot_speaking` | `()` | TTS audio streaming started |
| `on_bot_idle` | `()` | Bot finished, waiting for user |
| `on_bot_interrupted` | `()` | Bot speech was interrupted |
| `on_audio` | `(pcm_bytes: bytes)` | Raw PCM int16 TTS audio chunk |
| `on_error` | `(message: str)` | Error occurred |

### AudioStream

| Member | Type | Description |
|--------|------|-------------|
| `__iter__` / `__aiter__` | `Iterator[bytes]` | Yield raw byte chunks |
| `.to_bytes()` | `bytes` | Materialize as complete WAV |
| `.save(path)` | `None` | Save WAV to file |
| `.play()` | `None` | Play via sounddevice |
| `.ttfb` | `float \| None` | Seconds to first audio byte |
| `.elapsed` | `float \| None` | Seconds to stream completion |
| `.sample_rate` | `int` | Sample rate (sync only) |
| `.get_sample_rate()` | `int` | Sample rate (async only) |

### Data Models

| Model | Fields |
|-------|--------|
| `Info` | `voice`, `tts_engine`, `language`, `supports_streaming`, `pcm_prebuffer_samples`, `early_tts_decode`, `frontend_settings_disabled`, `multimodal_disabled` |
| `Settings` | `current: CurrentSettings`, `profiles: dict`, `default_prompts: dict`, `languages: list[str]`, `voices: dict[str, list[VoiceInfo]]` |
| `CurrentSettings` | `language`, `tts_engine`, `voice`, `stream_format`, `system_prompt` |
| `VoiceInfo` | `id`, `name` |
| `ProfileInfo` | `prompt` |
| `SettingsUpdateResponse` | `restart_required`, `message`, `error`, `success` |
| `ReloadPromptResponse` | `success`, `prompt` |
| `Transcription` | `text` |
| `DialogueInput` | `text`, `voice_id` |
| `BatchTTSItem` | `text`, `voice`, `language` |
| `BatchTTSResult` | `index`, `status`, `voice`, `language`, `audio_b64`, `error` + `.to_bytes()`, `.save()`, `.play()` |
| `DialogueBatchResult` | `.segments: list[BatchTTSResult]` + `.to_bytes()`, `.save()`, `.play()` |

### Exceptions

| Exception | Description |
|-----------|-------------|
| `OVAError` | Base exception |
| `OVAAuthenticationError` | API key is missing or invalid (401) |
| `OVAConnectionError` | Cannot reach server |
| `OVAServerNotReady` | Server warming up |
| `OVARequestError` | HTTP error (`.status_code` attribute) |
| `OVATimeoutError` | Request timed out |

## Requirements

- Python >= 3.10
- Core: `httpx >= 0.25`, `pydantic >= 2.0`, `numpy >= 1.24`, `sounddevice >= 0.4`, `websockets >= 12.0`

## License

MIT
