# SDK Examples

Standalone scripts demonstrating OVA SDK usage, ordered by complexity.

## Prerequisites

```bash
pip install ova-sdk
sudo apt-get install libportaudio2  # Linux only — PortAudio for audio I/O
```

OVA server must be running. Output files are saved to `output/{example-name}/` automatically.

## Overview

| File | Description |
|------|-------------|
| `01-quickstart.py` | Discover server, pick a voice, generate speech |
| `02-tts_voices.py` | Same text in every voice for the current language |
| `03-multilingual_tts.py` | Same phrase across multiple languages |
| `04-streaming_playback.py` | Real-time TTS playback with TTFB measurement |
| `05-batch_tts.py` | Batch TTS: sequential vs. parallel synthesis comparison |
| `06-text_to_dialogue.py` | Multi-speaker dialogue with voice alternation |
| `07-chat_conversation.py` | Multi-turn chat with voice switching |
| `08-settings_and_info.py` | Server inspection, voice discovery, configuration |
| `09-transcribe_audio.py` | One-shot speech-to-text with text file output |
| `10-interactive_chat.py` | Mic → ASR → LLM → TTS → speaker loop |
| `11-error_handling.py` | SDK exception hierarchy |

## Examples

### 01-quickstart.py

Connects to the server, discovers available voices, picks the first available voice, and generates speech using per-request overrides. Run this first to verify your setup. Saves to `output/quickstart/`.

### 02-tts_voices.py

Renders one sentence across every available voice for English using per-request `voice=` overrides. Works with both Qwen3 and Kokoro engines — uses the discovery API (`client.voices()`) to enumerate voices automatically. Saves to `output/tts-voices/`.

### 03-multilingual_tts.py

Generates a greeting in English, Spanish, French, and German. Uses per-request `voice=` and `language=` overrides with the first available voice per language. Saves to `output/multilingual-tts/`.

### 04-streaming_playback.py

Plays TTS audio through your speakers as chunks arrive. Handles all server modes: PCM streaming (Qwen3+pcm), WAV streaming (Qwen3+wav), and non-streaming (Kokoro). Measures time-to-first-byte for benchmarking streaming latency. Uses the first available voice. Saves to `output/streaming-playback/`.

### 05-batch_tts.py

Compares sequential vs. batch TTS generation. Synthesizes multiple texts using `client.tts.batch_generate()` and shows timing differences. Uses the first available voice. Saves to `output/batch-tts/`.

### 06-text_to_dialogue.py

Multi-speaker dialogue generation with voice alternation two discovered voices. Compares sequential vs. batch dialogue generation via `client.dialogue.generate()`. Saves to `output/text-to-dialogue/`.

### 07-chat_conversation.py

Runs a multi-turn conversation, switching voices between turns via per-request `voice=` overrides on `send_text()`. Uses discovered voices, switching between turns. Saves to `output/chat-conversation/`.

### 08-settings_and_info.py

Walks through the full settings API: query server info, list all languages and voices, switch the active voice via `settings.update()`, inject a custom system prompt (pirate mode), and restore defaults. Generates TTS after each change to confirm it took effect. Saves to `output/settings-and-info/`.

### 09-transcribe_audio.py

Sends a WAV file to the standalone transcription endpoint and prints the recognized text. Demonstrates transcription from file path and bytes, language override (e.g., forcing Spanish ASR), and error handling for oversized input. Saves transcription results to `output/transcribe-audio/transcription.txt`.

### 10-interactive_chat.py

Full voice conversation loop in the terminal. Records from your mic, streams to ASR for real-time transcription, sends the transcript through the LLM, and plays TTS chunks through your speakers. Uses the first available voice. Press Enter to start/stop each turn.

### 11-error_handling.py

Triggers each SDK exception type: `OVAAuthenticationError`, `OVAConnectionError`, `OVAServerNotReady`, `OVARequestError` (with HTTP status), `OVATimeoutError`, and the `OVAError` base class. Run with and without the server to see different error paths.
