"""Interactive voice chat — speak into the mic, hear the assistant reply.

Mirrors the same pipeline as the web frontend:
  1. Record mic audio (16kHz mono float32)
  2. Stream chunks to ASR WebSocket for real-time transcription
  3. Send final transcript through chat -> streaming TTS response
  4. Play audio chunks in real-time as they arrive

Requires: pip install ova-sdk[asr] sounddevice numpy
"""

import queue
import struct
import sys
import threading

import numpy as np
import sounddevice as sd

from ova_sdk import OVA

LANGUAGE = "en"
SAMPLE_RATE = 16000
CHUNK_MS = 500
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
_WAV_HEADER_SIZE = 44


def record_and_stream_asr(client: OVA) -> str:
    """Record from mic, stream to ASR, return final transcript."""
    audio_q: queue.Queue[np.ndarray | None] = queue.Queue()
    partials: list[str] = []
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"  [mic] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    def sender(asr_stream):
        """Read chunks from queue, send to ASR, print partials."""
        while not stop_event.is_set():
            try:
                chunk = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                break
            asr_stream.send(chunk)
            resp = asr_stream.receive()
            if "partial" in resp and resp["partial"]:
                partials.append(resp["partial"])
                print(f"\r  Hearing: {resp['partial']}", end="", flush=True)

    asr_stream = client.asr.stream()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
        callback=audio_callback,
    )

    print("  Recording... (press Enter to stop)")
    stream.start()
    sender_thread = threading.Thread(target=sender, args=(asr_stream,), daemon=True)
    sender_thread.start()

    input()  # Block until Enter

    stream.stop()
    stream.close()
    stop_event.set()
    audio_q.put(None)
    sender_thread.join(timeout=2)

    print()  # Newline after partial transcripts
    transcript = asr_stream.finish()
    asr_stream.close()
    return transcript


def play_streaming_response(audio_stream) -> None:
    """Play audio chunks in real-time, print TTFB and stats."""
    out_stream = None
    chunk_count = 0

    try:
        for chunk in audio_stream:
            chunk_count += 1

            if out_stream is None:
                if len(chunk) < _WAV_HEADER_SIZE:
                    continue
                sr = struct.unpack_from("<I", chunk, 24)[0]
                out_stream = sd.RawOutputStream(samplerate=sr, channels=1, dtype="int16")
                out_stream.start()
                pcm = chunk[_WAV_HEADER_SIZE:]
            elif chunk[:4] == b"RIFF":
                # WAV streaming mode: each chunk is a complete WAV file
                pcm = chunk[_WAV_HEADER_SIZE:]
            else:
                pcm = chunk

            if pcm:
                out_stream.write(pcm)
    finally:
        if out_stream is not None:
            sd.sleep(int(out_stream.latency * 1000))
            out_stream.stop()
            out_stream.close()

    if audio_stream.ttfb is not None:
        print(f"  TTFB: {audio_stream.ttfb:.3f}s | Chunks: {chunk_count} | Total: {audio_stream.elapsed:.3f}s")
    else:
        print("  No audio received")


def main():
    client = OVA()

    print("Waiting for OVA server...")
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

    print("Ready! Start a voice conversation (Ctrl+C to quit)")
    print(f"Voice: {VOICE}, Language: {LANGUAGE}\n")

    try:
        while True:
            print("Press Enter to start recording...")
            input()

            transcript = record_and_stream_asr(client)
            if not transcript.strip():
                print("  (empty transcript, skipping)\n")
                continue

            print(f"  You: {transcript}")
            print("  Assistant is thinking...")

            with client.chat.send_text(transcript, voice=VOICE, language=LANGUAGE) as audio:
                play_streaming_response(audio)
            print()
    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        client.close()


if __name__ == "__main__":
    main()
