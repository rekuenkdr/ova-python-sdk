"""Server inspection, voice discovery, and configuration."""

import sys
from pathlib import Path

from ova_sdk import OVA

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "settings-and-info"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE = "en"

client = OVA()
client.wait_until_ready()

all_voices = client.voices(language=LANGUAGE)
if not all_voices:
    print(f"No voices found for '{LANGUAGE}'. Create a voice profile first — see QUICKSTART.md")
    sys.exit(1)
VOICE = all_voices[0]

# ── Warmup: prime torch.compile cache ───────────────────
print("Warming up torch.compile cache...")
_ = client.tts.generate("Warmup.", voice=VOICE, language=LANGUAGE).to_bytes()
print("Warmup done.\n")

# -- Server info --
info = client.info()
print("=== Server Info ===")
print(f"  Engine:     {info.tts_engine}")
print(f"  Language:   {info.language}")
print(f"  Voice:      {info.voice}")
print(f"  Streaming:  {info.supports_streaming}")

# -- Available languages and voices --
print("\n=== Languages & Voices ===")
languages = client.languages()
for lang in languages:
    voices = client.voices(language=lang)
    print(f"  {lang}: {', '.join(voices) if voices else '(none)'}")

# -- Current settings --
print("\n=== Current Settings ===")
settings = client.settings.get()
print(f"  Language:  {settings.current.language}")
print(f"  Voice:     {settings.current.voice}")
print(f"  Prompt:    {settings.current.system_prompt[:80]}...")

# -- Switch voice --
voices = client.voices(language=LANGUAGE)
if VOICE in voices:
    print(f"\n=== Switch to {VOICE} ===")
    result = client.settings.update(voice=VOICE, language=LANGUAGE)
    print(f"  Restart required: {result.restart_required}")

    audio = client.tts.generate(
        f"Hello, I'm {VOICE}. The settings update worked.",
        voice=VOICE,
        language=LANGUAGE,
    )
    audio.save(str(OUTPUT_DIR / f"settings_{VOICE}.wav"))
    print(f"  Saved to {OUTPUT_DIR / f'settings_{VOICE}.wav'}")

# -- Custom prompt --
print("\n=== Custom Prompt: Pirate ===")
result = client.settings.reload_prompt(
    prompt="You are a pirate captain. Speak like a pirate. Keep it short.",
    clear_history=True,
)
print(f"  Success: {result.success}")

audio = client.chat.send_text(
    "What's your favorite hobby?",
    voice=VOICE,
    language=LANGUAGE,
)
audio.save(str(OUTPUT_DIR / "settings_pirate.wav"))
print("  Saved pirate response")

# -- Restore --
print("\n=== Restore ===")
client.settings.update(voice=info.voice, language=info.language)
client.settings.reload_prompt(clear_history=True)
print("  Restored defaults")
print("\nDone!")
