"""Generate speech across multiple languages using per-request overrides."""

import sys
from pathlib import Path

from ova_sdk import OVA

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "multilingual-tts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHRASES = {
    "en": "Hello! I love voice technology.",
    "es": "¡Hola! Me encanta la tecnología de voz.",
    "fr": "Bonjour! J'adore la technologie vocale.",
    "de": "Hallo! Ich liebe Sprachtechnologie.",
}

client = OVA()
client.wait_until_ready()

warmup_voices = client.voices(language="en") or client.voices()
if not warmup_voices:
    print("No voices found. Create a voice profile first — see QUICKSTART.md")
    sys.exit(1)

info = client.info()
languages = client.languages()
print(f"Engine: {info.tts_engine}, Available languages: {languages}\n")

# ── Warmup: prime torch.compile cache ───────────────────
print("Warming up torch.compile cache...")
_ = client.tts.generate("Warmup.", voice=warmup_voices[0], language="en").to_bytes()
print("Warmup done.\n")

for lang, text in PHRASES.items():
    if lang not in languages:
        print(f"  [{lang}] skipped (not available)")
        continue

    voices = client.voices(language=lang)
    if not voices:
        print(f"  [{lang}] skipped (no voices)")
        continue

    voice = voices[0]

    print(f"  [{lang}] {voice}: {text[:50]}...")
    audio = client.tts.generate(text, voice=voice, language=lang)
    audio.save(str(OUTPUT_DIR / f"multilingual_{lang}.wav"))

print("\nDone!")
