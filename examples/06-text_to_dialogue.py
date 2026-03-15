"""Multi-speaker dialogue generation via batch TTS.

Alternates between two voices for dialogue lines.
Compares sequential tts.generate() vs batch dialogue.generate() timing.
"""

import io
import sys
import time
import wave
from pathlib import Path

from ova_sdk import OVA, DialogueInput

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "text-to-dialogue"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE = "en"

with OVA() as client:
    client.wait_until_ready()

    voices = client.voices(language=LANGUAGE)
    if not voices:
        print(f"No voices found for '{LANGUAGE}'. Create a voice profile first — see QUICKSTART.md")
        sys.exit(1)
    VOICE_A = voices[0]
    VOICE_B = voices[1] if len(voices) >= 2 else voices[0]

    info = client.info()
    print(f"Engine: {info.tts_engine}, Language: {LANGUAGE}, Voices: {voices}\n")

    dialogue = [
        DialogueInput(text="Hey! How are you doing today?", voice_id=VOICE_A),
        DialogueInput(text="Pretty good, thanks for asking! And you?", voice_id=VOICE_B),
        DialogueInput(text="Fantastic! I've been working on some really interesting projects.", voice_id=VOICE_A),
        DialogueInput(text="That sounds amazing. Tell me more!", voice_id=VOICE_B),
    ]

    # ── Warmup: prime torch.compile cache ────
    print("Warming up torch.compile cache...")
    _ = client.tts.generate("Warmup.", voice=VOICE_A, language=LANGUAGE).to_bytes()
    if VOICE_B != VOICE_A:
        _ = client.tts.generate("Warmup.", voice=VOICE_B, language=LANGUAGE).to_bytes()
    print("Warmup done.\n")

    # ── Sequential: one item at a time ────────────────────────────
    print(f"\n{'='*60}")
    print(f"SEQUENTIAL — generating {len(dialogue)} items one at a time")
    print(f"{'='*60}\n")

    seq_wavs = []
    seq_times = []
    seq_start = time.perf_counter()

    for i, item in enumerate(dialogue):
        audio = client.tts.generate(item.text, voice=item.voice_id, language=LANGUAGE)
        wav_bytes = audio.to_bytes()
        seq_times.append(audio.elapsed)
        seq_wavs.append(wav_bytes)
        print(f"  Item {i}: {audio.elapsed:.3f}s  [{item.voice_id}] \"{item.text[:50]}\"")

    seq_total = time.perf_counter() - seq_start
    print(f"\n  Total: {seq_total:.3f}s | Avg: {sum(seq_times)/len(seq_times):.3f}s")

    # ── Batch: all items in a single call ─────────────────────────
    print(f"\n{'='*60}")
    print(f"BATCH — generating {len(dialogue)} items in a single call")
    print(f"{'='*60}\n")

    batch_start = time.perf_counter()
    result = client.dialogue.generate(dialogue, language=LANGUAGE)
    batch_total = time.perf_counter() - batch_start

    for seg in result.segments:
        status = "ok" if seg.status == "ok" else "FAIL"
        print(f"  [{status}] #{seg.index} [{seg.voice}] \"{dialogue[seg.index].text[:50]}\"")

    print(f"\n  Total: {batch_total:.3f}s (all {len(dialogue)} items)")

    # ── Comparison ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  Sequential total : {seq_total:.3f}s")
    print(f"  Batch total      : {batch_total:.3f}s")
    if batch_total > 0:
        speedup = seq_total / batch_total
        print(f"  Speedup          : {speedup:.1f}x")
    print()

    # ── Save results ─────────────────────────────────────────────
    # Sequential: manually concatenate WAVs
    if seq_wavs:
        with wave.open(io.BytesIO(seq_wavs[0]), "rb") as wf:
            params = wf.getparams()

        seq_path = OUTPUT_DIR / "dialogue_sequential.wav"
        with wave.open(str(seq_path), "wb") as out:
            out.setparams(params)
            for chunk in seq_wavs:
                with wave.open(io.BytesIO(chunk), "rb") as wf:
                    out.writeframes(wf.readframes(wf.getnframes()))

        print(f"Saved {seq_path} ({len(seq_wavs)} items concatenated)")

    # Batch: use built-in save
    batch_path = OUTPUT_DIR / "dialogue_batch.wav"
    result.save(str(batch_path))
    print(f"Saved {batch_path} ({len(result.segments)} segments)")
