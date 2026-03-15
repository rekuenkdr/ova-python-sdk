"""Batch TTS — synthesize multiple texts in a single call.

Uses client.tts.batch_generate() to process all items through the
transformer simultaneously, returning a list of BatchTTSResult objects.
Each result can be saved individually or concatenated into one WAV.

Compares sequential tts.generate() vs batch tts.batch_generate() timing.
"""

import io
import sys
import time
import wave
from pathlib import Path

from ova_sdk import OVA, BatchTTSItem

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "batch-tts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE = "en"

with OVA() as client:
    client.wait_until_ready()

    voices = client.voices(language=LANGUAGE)
    if not voices:
        print(f"No voices found for '{LANGUAGE}'. Create a voice profile first — see QUICKSTART.md")
        sys.exit(1)
    VOICE = voices[0]

    info = client.info()
    voices = client.voices(language=LANGUAGE)
    print(f"Engine: {info.tts_engine}, Language: {LANGUAGE}, Voices: {voices}\n")

    # ── Warmup: prime torch.compile cache ───────────────────
    print("Warming up torch.compile cache...")
    _ = client.tts.generate("Warmup.", voice=VOICE, language=LANGUAGE).to_bytes()
    print("Warmup done.\n")

    # Build a batch of items
    items = [
        BatchTTSItem(text="Welcome to the batch TTS demo.", voice=VOICE, language=LANGUAGE),
        BatchTTSItem(text="Each item is synthesized in parallel on the server.", voice=VOICE),
    ]

    # ── Sequential: one item at a time ────────────────────────────
    print(f"\n{'='*60}")
    print(f"SEQUENTIAL — generating {len(items)} items one at a time")
    print(f"{'='*60}\n")

    seq_wavs = []
    seq_times = []
    seq_start = time.perf_counter()

    for i, item in enumerate(items):
        audio = client.tts.generate(item.text, voice=item.voice, language=item.language)
        wav_bytes = audio.to_bytes()
        seq_times.append(audio.elapsed)
        seq_wavs.append(wav_bytes)
        print(f"  Item {i}: {audio.elapsed:.3f}s  \"{item.text[:50]}\"")

    seq_total = time.perf_counter() - seq_start
    print(f"\n  Total: {seq_total:.3f}s | Avg: {sum(seq_times)/len(seq_times):.3f}s")

    # ── Batch: all items in a single call ─────────────────────────
    print(f"\n{'='*60}")
    print(f"BATCH — generating {len(items)} items in a single call")
    print(f"{'='*60}\n")

    batch_start = time.perf_counter()
    results = client.tts.batch_generate(items)
    batch_total = time.perf_counter() - batch_start

    wav_chunks = []
    for r in results:
        status = "ok" if r.status == "ok" else "FAIL"
        print(f"  [{status}] #{r.index} voice={r.voice}: \"{items[r.index].text[:50]}\"")
        if r.status == "ok":
            r.save(str(OUTPUT_DIR / f"batch_{r.index}.wav"))
            wav_chunks.append(r.to_bytes())
        else:
            print(f"         Error: {r.error}")

    print(f"\n  Total: {batch_total:.3f}s (all {len(items)} items)")

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

    # ── Concatenate batch results into batch_all.wav ──────────────
    if wav_chunks:
        with wave.open(io.BytesIO(wav_chunks[0]), "rb") as wf:
            params = wf.getparams()

        all_wav_path = OUTPUT_DIR / "batch_all.wav"
        with wave.open(str(all_wav_path), "wb") as out:
            out.setparams(params)
            for chunk in wav_chunks:
                with wave.open(io.BytesIO(chunk), "rb") as wf:
                    out.writeframes(wf.readframes(wf.getnframes()))

        print(f"Saved {all_wav_path} ({len(wav_chunks)} items concatenated)")
        print(f"Also saved batch_0.wav .. batch_{len(wav_chunks)-1}.wav in {OUTPUT_DIR}/")
