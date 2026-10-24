"""Loud-noise detection for the robot server: listens to Pupper's mic and reports bangs, shouts, alarms and sirens.

OPT IN: it is off every time the server starts, and the mic isn't opened for it until a person
asks Pupper to listen (the sound_alerts tool in lab/hackathon_alerts.py, which also decides what
each detection becomes: the screen, and a note to Gemini).

It shares the mic with the kiosk browser through PipeWire, so it can run while Gemini listens.
The detectors use the thresholds saved by the calibration tool (hearing/mic_lab.py) in
~/.config/gemini-pupper/hearing/calibration.json. Those are in dB above the room's own noise
floor, so they hold at any fixed mic gain. The gain must not move while listening, though, so the
source volume is held where it was when listening started (the kiosk browser is also started
with its auto gain control kept off the system volume, see robot/start.sh).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import time
from typing import Callable

import numpy as np

import classify
import detect as D
import meter as M
import store

log = logging.getLogger("hearing")
BLOCK = 2400            # 50 ms
WINDOW_S = 12.0
SENSITIVITY_OFFSET_DB = {"low": 5.0, "medium": 0.0, "high": -5.0}  # added to every calibrated threshold


def _wpctl(*args: str) -> str:
    env = {**os.environ, "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")}
    try:
        return subprocess.run(["wpctl", *args], capture_output=True, text=True, timeout=3, env=env).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def source_volume() -> float | None:
    m = re.search(r"Volume:\s*([\d.]+)", _wpctl("get-volume", "@DEFAULT_AUDIO_SOURCE@"))
    return float(m.group(1)) if m else None


class Hearing:
    def __init__(self, on_event: Callable[[dict], None], device: str = "pulse"):
        self.on_event = on_event
        self.device = device
        self.enabled = False
        self.stream = None
        self.task: asyncio.Task | None = None
        self.volume: float | None = None
        self.params: dict = {}
        self.calibrated_at: str | None = None
        self.offset_db = 0.0
        self.heard = 0
        self.hears: dict = {"label": None, "t": 0.0}   # YAMNet's top label right now (None: silence)

    def now(self) -> dict:
        """What Pupper hears right now, for the corner of the screen: {"label": ...} or label None."""
        fresh = self.enabled and time.time() - self.hears.get("t", 0.0) < 2.0
        return {"label": self.hears.get("label") if fresh else None, "confidence": self.hears.get("confidence", 0.0)}

    def status(self) -> dict:
        return {"enabled": self.enabled, "heard": self.heard, "source_volume": self.volume,
                "calibrated_at": self.calibrated_at, "sensitivity_offset_db": self.offset_db,
                "thresholds": {c: self.params[c]["threshold"] for c in D.CLASSES} if self.params else None}

    async def set_enabled(self, on: bool, sensitivity: str = "medium") -> tuple[bool, str]:
        """Start or stop listening. Returns (ok, what happened), for the tool's reply."""
        offset = SENSITIVITY_OFFSET_DB.get(sensitivity, 0.0)
        if not on:
            if self.enabled:
                await self.stop()
            return True, "the mic is no longer checked for loud sounds"
        if self.enabled and offset == self.offset_db:
            return True, self.describe()
        if self.enabled:
            await self.stop()
        self.offset_db = offset
        try:
            await self.start()
        except Exception as e:  # a missing mic must not crash the server
            log.exception("hearing failed to start")
            await self.stop()
            return False, f"couldn't open the mic for sound alerts: {type(e).__name__}: {e}"
        return True, self.describe()

    def describe(self) -> str:
        what = ", ".join(f"{c} +{self.params[c]['threshold']:g} dB" for c in D.CLASSES)
        return f"listening with the calibrated detector (over the room: {what})"

    async def start(self) -> None:
        import sounddevice as sd

        cal = store.load_calibration()
        self.params = D.merged_params(cal["params"] if cal else None)
        for c in D.CLASSES:
            self.params[c]["threshold"] = self.params[c]["threshold"] + self.offset_db
        self.calibrated_at = cal.get("saved_at") if cal else None
        self.volume = source_volume()
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def callback(indata, frames, t, status):
            loop.call_soon_threadsafe(q.put_nowait, indata[:, 0].copy())

        self.stream = sd.InputStream(device=self.device, samplerate=M.FS, channels=1, dtype="float32",
                                     blocksize=BLOCK, latency="high", callback=callback)
        self.stream.start()
        self.enabled = True
        self.task = loop.create_task(self._run(q))
        log.info("hearing on (source volume %s, thresholds %s, calibration %s)", self.volume,
                 self.status()["thresholds"], self.calibrated_at)

    async def stop(self) -> None:
        self.enabled = False
        if self.task:
            self.task.cancel()
            self.task = None
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        log.info("hearing off")

    async def _run(self, q: asyncio.Queue) -> None:
        yam = classify.SoundClassifier() if classify.available() else None   # sirens and alarms
        if yam is None:
            log.warning("YAMNet not installed (python3 hearing/classify.py --install): sirens and alarms use the meter")
        meter, recent = M.Meter(), []
        live = D.LiveDetector(self.params, classes=("bang", "shout", "knock") if yam else D.CLASSES,
                              soft_knocks=yam is not None)
        knocks = classify.KnockGate(yam) if yam else None   # meter knocks wait for YAMNet to confirm (KnockGate)
        blocks, next_check = 0, time.monotonic() + 2.0
        while True:
            block = await q.get()
            pcm = np.clip(np.round(block * 32768.0), -32768, 32767).astype(np.int16)
            f = meter.feed(pcm)
            blocks += 1
            if len(f["t"]):
                recent.append(f)
                if yam is not None:
                    view, evs = yam.feed(pcm, float(f["la"][-1] - f["floor_a"][-1]))
                    if view is not None:
                        evs = evs + knocks.ready()
                        now = view["now"]
                        self.hears = {"label": now[0] if now else None, "confidence": now[1] if now else 0.0,
                                      "t": time.time()}
                    for ev in evs:
                        self._heard(ev)
            if blocks % 2 == 0 and recent:
                w = M.concat_frames(recent)
                w = M.slice_frames(w, max(0, len(w["t"]) - int(WINDOW_S / D.DT)))
                recent = [w]
                win = M.slice_frames(w, max(0, len(w["t"]) - int(D.LiveDetector.WINDOW_S / D.DT)))
                for ev in live.update(win):
                    if ev["cls"] == "knock" and knocks is not None:
                        knocks.add(ev)
                    else:
                        self._heard(ev)
            if time.monotonic() > next_check:
                next_check = time.monotonic() + 2.0
                await self._hold_volume()

    def _heard(self, ev: dict) -> None:
        self.heard += 1
        log.info("heard %s: %.1f, %s", ev["cls"], ev["score"], ev["info"])
        try:
            self.on_event(ev)
        except Exception:  # a bug in what reacts must not stop the listening
            log.exception("hearing callback failed")

    async def _hold_volume(self) -> None:
        if self.volume is None:
            return
        now = await asyncio.to_thread(source_volume)
        if now is not None and abs(now - self.volume) > 0.005:
            log.warning("mic volume moved from %.2f to %.2f while listening; putting it back", self.volume, now)
            await asyncio.to_thread(_wpctl, "set-volume", "@DEFAULT_AUDIO_SOURCE@", f"{self.volume:.2f}")
