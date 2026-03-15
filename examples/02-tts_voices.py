"""Same text in every voice for English."""

from pathlib import Path

from ova_sdk import OVA

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "tts-voices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE = "en"

client = OVA()
client.wait_until_ready()

info = client.info()
voices = client.voices(language=LANGUAGE)
print(f"Engine: {info.tts_engine}, Language: {LANGUAGE}, Voices: {len(voices)}\n")

TEXT = "The quick brown fox jumps over the lazy dog."

# ── Warmup: prime torch.compile cache ───────────────────
print("Warming up torch.compile cache...")
_ = client.tts.generate("Warmup.", voice=voices[0], language=LANGUAGE).to_bytes()
print("Warmup done.\n")

for voice in voices:
    print(f"  {voice}...")
    audio = client.tts.generate(TEXT, voice=voice, language=LANGUAGE)
    audio.save(str(OUTPUT_DIR / f"tts_{voice}.wav"))

print(f"\nDone! {len(voices)} files in {OUTPUT_DIR}/")
