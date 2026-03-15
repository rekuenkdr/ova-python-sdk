"""Audio playback and save utilities."""

from __future__ import annotations

import io
import struct
import wave
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ._streaming import AudioStream


def play(source: Union["AudioStream", bytes]) -> None:
    """Play audio using sounddevice.

    Args:
        source: AudioStream or raw WAV bytes.

    Raises:
        ImportError: If sounddevice is not installed.
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError(
            "Audio playback requires 'sounddevice'. "
            "Install with: pip install sounddevice"
        )
    import numpy as np

    wav_bytes = source.to_bytes() if hasattr(source, "to_bytes") else source
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    sd.play(audio, samplerate=sr)
    sd.wait()


def save(source: Union["AudioStream", bytes], path: str) -> None:
    """Save audio to a WAV file.

    Args:
        source: AudioStream or raw WAV bytes.
        path: Output file path.
    """
    wav_bytes = source.to_bytes() if hasattr(source, "to_bytes") else source
    with open(path, "wb") as f:
        f.write(wav_bytes)
