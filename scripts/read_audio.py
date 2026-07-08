#!/usr/bin/env python3
"""read_audio.py — Read and play PCM audio files.

Usage:
    python scripts/read_audio.py data/audio/pizza_input_abc123.wav
    python scripts/read_audio.py data/audio/llm_response_xyz789.wav

Plays the audio file using the system audio player.
"""
import sys
import subprocess
import tempfile
from pathlib import Path


def play_audio(filepath: str) -> None:
    """Play an audio file using the system audio player."""
    audio_path = Path(filepath)

    if not audio_path.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    # Get file extension
    ext = audio_path.suffix.lower()

    if ext == ".wav":
        # Use system player for WAV files
        import platform
        system = platform.system()

        if system == "Darwin":  # macOS
            subprocess.run(["afplay", str(audio_path)], check=True)
        elif system == "Windows":
            # Use PowerShell for Windows
            cmd = f'Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Application]::EnableVisualStyles(); $player = New-Object System.Media.SoundPlayer; $player.SoundLocation = "{audio_path}"; $player.PlaySync()'
            subprocess.run(["powershell", "-Command", cmd], check=True)
        else:  # Linux
            # Try pulseaudio, then aplay, then ffplay
            try:
                subprocess.run(["paplay", str(audio_path)], check=True)
            except FileNotFoundError:
                try:
                    subprocess.run(["aplay", str(audio_path)], check=True)
                except FileNotFoundError:
                    try:
                        subprocess.run(["ffplay", "-nodisp", "-autoexit", str(audio_path)], check=True)
                    except FileNotFoundError:
                        print("Error: No audio player found (need paplay, aplay, or ffplay)")
                        sys.exit(1)
    elif ext == ".pcm":
        # Convert PCM to WAV temporarily
        import wave
        import struct

        with open(str(audio_path), "rb") as f:
            pcm_data = f.read()

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with wave.open(str(tmp_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)
                # Convert PCM bytes to WAV format
                num_samples = len(pcm_data) // 2
                samples = struct.unpack(f'<{num_samples}h', pcm_data)
                wf.writeframes(struct.pack(f'<{num_samples}h', *samples))

            # Play the temporary WAV file
            play_audio(str(tmp_path))
            tmp_path.unlink()  # Clean up
    else:
        print(f"Error: Unsupported format: {ext}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/read_audio.py <audio_file.wav|pcm>")
        print("  Plays the audio file using the system audio player")
        sys.exit(1)

    play_audio(sys.argv[1])