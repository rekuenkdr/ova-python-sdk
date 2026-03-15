"""Quickstart: discover the server, pick a voice, speak."""

import sys
from pathlib import Path

from ova_sdk import OVA

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "quickstart"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE = "en"

client = OVA()
client.wait_until_ready()

voices = client.voices(language=LANGUAGE)
if not voices:
    print(f"No voices found for '{LANGUAGE}'. Create a voice profile first — see QUICKSTART.md")
    sys.exit(1)
VOICE = voices[0]

# 1. What engine is running?
info = client.info()
print(f"Engine:   {info.tts_engine}")
print(f"Current:  language={info.language}, voice={info.voice}")

# 2. What languages are available?
languages = client.languages()
print(f"Languages: {languages}")

# 3. What voices are available for English?
voices = client.voices(language=LANGUAGE)
print(f"Voices ({LANGUAGE}): {voices}")

# 4. Generate speech
print(f"\nUsing voice '{VOICE}'")

# ── Warmup: prime torch.compile cache ───────────────────
print("Warming up torch.compile cache...")
_ = client.tts.generate("Warmup.", voice=VOICE, language=LANGUAGE).to_bytes()
print("Warmup done.\n")

print(f"Generating with voice '{VOICE}'...")
audio = client.tts.generate(
    "Hello! This is a quick test of the OVA voice assistant.",
    voice=VOICE,
    language=LANGUAGE,
)
audio.save(str(OUTPUT_DIR / "quickstart.wav"))
print(f"Saved to {OUTPUT_DIR / 'quickstart.wav'}")
