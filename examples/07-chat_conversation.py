"""Multi-turn chat with voice switching via send_text(voice=...)."""

import sys
from pathlib import Path

from ova_sdk import OVA

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "chat-conversation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE = "en"

client = OVA()
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

# Turn 1
print(f"Turn 1: {VOICE}")
audio = client.chat.send_text(
    "Hi! What's your name and what language are you speaking?",
    voice=VOICE,
    language=LANGUAGE,
)
audio.save(str(OUTPUT_DIR / "chat_turn1.wav"))
print("  Saved chat_turn1.wav")

# Turn 2: switch to another voice if available
other_voices = [v for v in voices if v != VOICE]
if other_voices:
    second_voice = other_voices[0]
    print(f"Turn 2: switching to {second_voice}")
    audio = client.chat.send_text("Now tell me a fun fact.", voice=second_voice)
    audio.save(str(OUTPUT_DIR / "chat_turn2.wav"))
    print("  Saved chat_turn2.wav")

# Turn 3: switch to yet another voice if available
if len(other_voices) > 1:
    third_voice = other_voices[1]
    print(f"Turn 3: switching to {third_voice}")
    audio = client.chat.send_text("One more fact, please.", voice=third_voice)
    audio.save(str(OUTPUT_DIR / "chat_turn3.wav"))
    print("  Saved chat_turn3.wav")

print("\nDone!")
