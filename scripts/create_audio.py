#!/usr/bin/env python3
"""create_audio.py — Generate PCM audio files for pizza agent testing.

Usage:
    python scripts/create_audio.py "I'd like a large pepperoni pizza"
    python scripts/create_audio.py "medium with mushrooms and olives"
    python scripts/create_audio.py "That sounds great, total price?"

Generates mono 16-bit PCM WAV files in data/audio/ directory.
"""
import sys
import wave
import struct
from pathlib import Path

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit


def text_to_pcm(text: str) -> bytes:
    """Synthesize simple PCM audio from text using tone generation.

    This creates a simple tone-based representation. For real speech,
    you would use a TTS service to generate the audio.
    """
    import time

    samples = []
    duration_per_char = 0.05  # 50ms per character
    silence_duration = 0.1  # 100ms between words

    words = text.split()
    for i, word in enumerate(words):
        # Generate a simple tone for each character
        for char in word:
            freq = 200 + (ord(char) % 26) * 30  # Vary frequency by character
            num_samples = int(SAMPLE_RATE * duration_per_char)
            for t in range(num_samples):
                value = int(32767 * 0.3 * (0.5 + 0.5 * (t / num_samples)))
                samples.append(value)

        # Add silence between words
        if i < len(words) - 1:
            silence_samples = int(SAMPLE_RATE * silence_duration)
            samples.extend([0] * silence_samples)

    # Convert to bytes
    pcm_bytes = struct.pack(f'<{len(samples)}h', *samples)
    return pcm_bytes


def create_wav_file(text: str, filename: str) -> Path:
    """Create a WAV file from text."""
    pcm_bytes = text_to_pcm(text)
    output_path = Path("data/audio") / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

    print(f"Created: {output_path}")
    print(f"  Duration: {len(pcm_bytes) / (SAMPLE_RATE * NUM_CHANNELS * SAMPLE_WIDTH):.2f}s")
    print(f"  Text: {text}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/create_audio.py \"text to convert\"")
        print("  Generates mono 16-bit PCM WAV file in data/audio/")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    import hashlib
    hash_str = hashlib.md5(text.encode()).hexdigest()[:8]
    filename = f"pizza_input_{hash_str}.wav"

    create_wav_file(text, filename)