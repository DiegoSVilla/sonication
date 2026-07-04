"""Background endpoint pinger.

Runs one shared async loop that hits a cheap health route on each service every
few seconds, keeps a small rolling window of round-trip times, and exposes a
rounded representative value per service. This is the network-floor reference
shown next to the measured latencies (e.g. TTS TTFB 485ms with a 233ms ping
means ~250ms is real service time above the network).
"""
import asyncio
import time
from collections import deque
from statistics import median
from typing import Optional

from . import clients, config

PING_INTERVAL_S = 5.0
WINDOW = 5  # representative value is the median of the last WINDOW samples


def _targets() -> dict[str, str]:
    return {
        "llm": f"{config.LLM_BASE_URL}/health",
        "tts": f"{config.TTS_BASE_URL}/health",
        "stt": f"{config.STT_BASE_URL}/health",
    }


class EndpointMonitor:
    def __init__(self) -> None:
        self._samples: dict[str, deque] = {k: deque(maxlen=WINDOW) for k in _targets()}
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            targets = _targets()
            await asyncio.gather(*(self._ping(n, u) for n, u in targets.items()))
            await asyncio.sleep(PING_INTERVAL_S)

    async def _ping(self, name: str, url: str) -> None:
        # Goes through clients.ping so it honours the connection mode: pooled
        # (keeps connections warm) or new-connection (fresh handshake each time).
        key = {"llm": config.LLM_API_KEY, "tts": config.TTS_API_KEY,
               "stt": config.STT_API_KEY}.get(name, "")
        headers = config.bearer(key)
        t0 = time.perf_counter()
        try:
            status = await clients.ping(url, headers)
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            if status < 500:
                self._samples[name].append(rtt_ms)
        except Exception:
            pass  # drop the failed sample; median uses the successful ones

    def snapshot(self) -> dict[str, Optional[int]]:
        """Rounded median RTT per service over the last WINDOW samples."""
        return {
            name: (round(median(dq)) if dq else None)
            for name, dq in self._samples.items()
        }


monitor = EndpointMonitor()
