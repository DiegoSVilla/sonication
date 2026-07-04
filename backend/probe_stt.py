"""TTS -> STT round-trip probe.

Synthesizes a known sentence with the configured TTS voice, then transcribes it
with the STT and reports the text and timings. Confirms both endpoints and the
audio format line up.

    python -m backend.probe_stt
    python -m backend.probe_stt "Custom sentence to speak and transcribe."
"""
import asyncio
import sys
import time

from . import clients, config


async def run(text: str) -> None:
    print(f"voice={config.TTS_VOICE!r}  tts_lang={config.TTS_LANGUAGE!r}  "
          f"stt_model={config.STT_MODEL!r}  stt_lang={config.STT_LANGUAGE!r}")
    print(f"\ninput text : {text!r}")

    t0 = time.perf_counter()
    wav = await clients.synthesize_wav(text, config.TTS_VOICE, config.TTS_LANGUAGE)
    t_tts = (time.perf_counter() - t0) * 1000
    print(f"TTS        : {len(wav)} bytes wav in {t_tts:.0f} ms")

    t1 = time.perf_counter()
    result = await clients.transcribe(wav, language=config.STT_LANGUAGE or None)
    t_stt = (time.perf_counter() - t1) * 1000
    print(f"STT        : {t_stt:.0f} ms  usage={result.get('usage')}")
    print(f"transcript : {result.get('text')!r}")


if __name__ == "__main__":
    sentence = " ".join(sys.argv[1:]) or (
        "The quick brown fox jumps over the lazy dog. Testing one two three four five."
    )
    asyncio.run(run(sentence))
