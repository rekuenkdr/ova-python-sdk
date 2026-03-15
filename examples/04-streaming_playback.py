"""Real-time streaming playback with TTFB measurement.

Plays audio chunks through the speakers as they arrive from the server,
rather than buffering the entire response first. Saves a copy to disk.
"""

import struct
import sys
import wave
from pathlib import Path

import sounddevice as sd

from ova_sdk import OVA

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "streaming-playback"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE = "en"
TEXT = "This is a streaming demo. Each audio chunk arrives as it's generated, giving you low-latency playback."

_WAV_HEADER_SIZE = 44

client = OVA()
client.wait_until_ready()

voices = client.voices(language=LANGUAGE)
if not voices:
    print(f"No voices found for '{LANGUAGE}'. Create a voice profile first — see QUICKSTART.md")
    sys.exit(1)
VOICE = voices[0]

# ── Warmup: prime torch.compile cache ───────────────────
print("Warming up torch.compile cache...")
_ = client.tts.generate("Warmup.", voice=VOICE, language=LANGUAGE).to_bytes()
print("Warmup done.\n")

audio = client.tts.generate(TEXT, voice=VOICE, language=LANGUAGE)
chunk_count = 0
stream = None
sr = None
all_pcm = bytearray()

for chunk in audio:
    chunk_count += 1

    if stream is None:
        # First chunk: parse WAV header, extract sample rate, open output
        if len(chunk) < _WAV_HEADER_SIZE:
            continue
        sr = struct.unpack_from("<I", chunk, 24)[0]
        stream = sd.RawOutputStream(samplerate=sr, channels=1, dtype="int16")
        stream.start()
        pcm = chunk[_WAV_HEADER_SIZE:]
    elif chunk[:4] == b"RIFF":
        # WAV streaming mode: each chunk is a complete WAV file
        pcm = chunk[_WAV_HEADER_SIZE:]
    else:
        pcm = chunk

    if pcm:
        stream.write(pcm)
        all_pcm.extend(pcm)

# Drain remaining buffered audio
if stream is not None:
    sd.sleep(int(stream.latency * 1000))
    stream.stop()
    stream.close()

print(f"TTFB:       {audio.ttfb:.3f}s")
print(f"Chunks:     {chunk_count}")
print(f"Total time: {audio.elapsed:.3f}s")

# Save collected PCM as WAV
if all_pcm and sr:
    wav_path = OUTPUT_DIR / "streaming_playback.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(all_pcm))
    print(f"Saved to {wav_path}")
