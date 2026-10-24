#!/usr/bin/env python3
"""Loud-noise calibration tool: see Pupper's mic signal live, record labelled trials, fit and save
the detector's thresholds, then check them on fresh sounds.

On the Pupper. Stop the kiosk first: its browser holds the mic. The robot stack isn't needed.
    python3 hearing/mic_lab.py                     # web UI on port 8001
    ssh -N -L 8001:localhost:8001 pi@pupper5.local # on the laptop, then open http://localhost:8001
    python3 hearing/mic_lab.py --bench             # CPU use of the signal chain on this machine
    python3 hearing/mic_lab.py --rescore           # re-score the latest session with the saved calibration
On a laptop, without the Pupper:
    python3 hearing/mic_lab.py --demo              # a synthetic scene on loop (claps, shout, talk, alarms)
    python3 hearing/mic_lab.py --wav file.wav      # a recording, on loop
    python3 hearing/mic_lab.py --device default    # the laptop's mic (UI work only: its levels mean nothing for Pupper)

The mic is read straight from ALSA (the Brio 101's hw device) at a fixed hardware gain. Going
through PipeWire would put the source volume in the path, and the kiosk browser's auto gain
control moves that volume by itself, so a calibration made through PipeWire would drift.

This tool never moves the robot. It listens only while it runs, and it writes audio to disk only
for the trials you record (see store.py for where).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np
from contextlib import asynccontextmanager
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classify  # noqa: E402
import detect as D  # noqa: E402
import meter as M  # noqa: E402
import store  # noqa: E402
import synth as Y  # noqa: E402

log = logging.getLogger("mic_lab")
BLOCK = 2400            # 50 ms per audio block (a multiple of the 10 ms hop)
PRE_S = 3.0             # pre-roll kept in front of every trial
RECENT_S = 12.0         # frames kept for the live detector and the UI
MAX_TRIAL_S = 120.0
SEND_COLS = ("t", "la", "laf", "lc_max", "floor_a", "floor_c", "peak", "clip", "onset_c", "tone_prom",
             "periodicity", "voice_ratio")
TRIAL_KINDS = ("positive", "negative", "quiet", "linearity", "spl", "other")


# ---------------------------------------------------------------------------- the mic
def find_input(name: str):
    import sounddevice as sd

    if name == "default":
        idx = sd.default.device[0]
        return idx, sd.query_devices(idx)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name.lower() in d["name"].lower():
            return i, d
    names = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    raise SystemExit(f"no input device matching {name!r}; inputs: {names}. Is the webcam plugged in?")


def alsa_card(device_name: str) -> int | None:
    m = re.search(r"\(hw:(\d+),", device_name)
    return int(m.group(1)) if m else None


def read_gain(card: int | None) -> dict | None:
    """The Brio's hardware mic gain (ALSA 'Mic Capture Volume'), raw and in dB."""
    if card is None:
        return None
    try:
        out = subprocess.run(["amixer", "-c", str(card), "cget", "name=Mic Capture Volume"],
                             capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    val = re.search(r": values=(\d+)", out)
    rng = re.search(r"min=(\d+),max=(\d+)", out)
    dbr = re.search(r"dBminmax-min=([-\d.]+)dB,max=([-\d.]+)dB", out)
    if not val:
        return None
    raw = int(val.group(1))
    g = {"raw": raw}
    if rng:
        g["min"], g["max"] = int(rng.group(1)), int(rng.group(2))
    if rng and dbr:
        lo, hi = float(dbr.group(1)), float(dbr.group(2))
        g["db"] = round(lo + (hi - lo) * (raw - g["min"]) / max(1, g["max"] - g["min"]), 2)
        g["db_min"], g["db_max"] = lo, hi
    return g


class MicSource:
    def __init__(self, device: str):
        self.index, self.info = find_input(device)
        self.name = self.info["name"]
        self.card = alsa_card(self.name)
        self.overflows = 0
        self.stream = None

    def start(self, loop: asyncio.AbstractEventLoop, q: asyncio.Queue) -> None:
        import sounddevice as sd

        def callback(indata, frames, t, status):
            if status.input_overflow:
                self.overflows += 1
            loop.call_soon_threadsafe(q.put_nowait, indata[:, 0].copy())

        try:
            self.stream = sd.InputStream(device=self.index, samplerate=M.FS, channels=1, dtype="int16",
                                         blocksize=BLOCK, latency="high", callback=callback)
            self.stream.start()
        except sd.PortAudioError as e:
            raise SystemExit(
                f"could not open {self.name} at {M.FS} Hz: {e}\n"
                "If the device is busy, the kiosk's browser (or another recorder) has the mic: "
                "stop it with robot/stop.sh, then run this again.")

    def stop(self) -> None:
        if self.stream:
            self.stream.stop()
            self.stream.close()


class FileSource:
    """Plays samples on loop at real-time pace, as if they came from the mic."""

    def __init__(self, samples: np.ndarray, name: str):
        self.samples = samples[: len(samples) // BLOCK * BLOCK]
        self.name = name
        self.card = None
        self.overflows = 0
        self.task = None

    def start(self, loop: asyncio.AbstractEventLoop, q: asyncio.Queue) -> None:
        async def run():
            t0, i = time.monotonic(), 0
            while True:
                q.put_nowait(self.samples[(i * BLOCK) % len(self.samples):][:BLOCK].copy())
                i += 1
                await asyncio.sleep(max(0.0, t0 + i * BLOCK / M.FS - time.monotonic()))
        self.task = loop.create_task(run())

    def stop(self) -> None:
        if self.task:
            self.task.cancel()


# ---------------------------------------------------------------------------- the engine
class Engine:
    """Meter + live detector on the incoming audio; keeps what the UI and the trials need."""

    def __init__(self, source, session: store.Session):
        self.source = source
        self.session = session
        self.meter = M.Meter()
        self.cal = store.load_calibration()
        self.params = D.merged_params(self.cal["params"] if self.cal else None)
        self.spl_offset = self.cal.get("spl_offset_db") if self.cal else None
        self.yam = classify.SoundClassifier() if classify.available() else None   # sirens and alarms
        self.live = D.LiveDetector(self.params, classes=("bang", "shout", "knock") if self.yam else D.CLASSES)
        self.knocks = classify.KnockGate(self.yam) if self.yam else None   # knocks wait for YAMNet's "not speech"
        self.hears: dict = {}
        self.recent: list[dict] = []
        self.recent_len = 0
        self.preroll: deque = deque()          # (block, floor_a, floor_c) for the last PRE_S
        self.trial: dict | None = None
        self.events: deque = deque(maxlen=300)
        self.event_id = 0
        self.clients: dict = {}            # websocket -> its send queue
        self.outbox: list[dict] = []
        self.blocks = 0
        self.started_wall = time.time()
        self.cpu_s = 0.0
        self.gain = read_gain(source.card)
        self.results: dict = {"linearity": None, "quiet": None, "fit": None, "verify": None}
        self.verify: dict | None = None
        if self.cal:
            self.results.update({k: self.cal.get(k) for k in ("linearity", "quiet", "fit", "verify")})

    # -- audio
    async def run(self, q: asyncio.Queue) -> None:
        while True:
            block = await q.get()
            c0 = time.process_time()
            self.process(block)
            self.cpu_s += time.process_time() - c0

    def process(self, block: np.ndarray) -> None:
        fa, fc = self.meter.floor_a, self.meter.floor_c
        f = self.meter.feed(block)
        self.blocks += 1
        self.preroll.append((block, fa, fc))
        while len(self.preroll) * BLOCK > PRE_S * M.FS:
            self.preroll.popleft()
        if self.trial is not None:
            self.trial["blocks"].append(block)
            if len(self.trial["blocks"]) * BLOCK / M.FS > MAX_TRIAL_S:
                log.warning("trial hit %.0f s; stopping it", MAX_TRIAL_S)
                asyncio.get_running_loop().create_task(self.stop_trial())
        if not len(f["t"]):
            return
        self.recent.append(f)
        self.recent_len += len(f["t"])
        self.outbox.append(f)
        if self.yam is not None:
            view, evs = self.yam.feed(block, float(f["la"][-1] - f["floor_a"][-1]))
            for ev in evs:
                # YAMNet's clock is samples seen; put the event on the meter's clock for the graph.
                shift = self.meter.hops * D.DT - self.yam.t
                self.report({**ev, "t0": ev["t0"] + shift, "t1": ev["t1"] + shift, "t": ev["t"] + shift})
            if view is not None:
                for ev in self.knocks.ready():
                    self.report(ev)
                was = self.hears.get("active")
                self.hears = view
                self.broadcast({"type": "hears", **view})
                if view["active"] != was:
                    self.broadcast({"type": "active", "active": view["active"]})
        if self.blocks % 2 == 0:
            self.detect()
        self.flush()

    def window(self, seconds: float) -> dict:
        w = M.concat_frames(self.recent)
        keep = int(RECENT_S / D.DT)
        if len(w["t"]) > keep:
            w = M.slice_frames(w, len(w["t"]) - keep)
        self.recent, self.recent_len = [w], len(w["t"])
        n = int(seconds / D.DT)
        return M.slice_frames(w, max(0, len(w["t"]) - n))

    def detect(self) -> None:
        was = dict(self.live.active)
        new_events = self.live.update(self.window(D.LiveDetector.WINDOW_S))
        if self.yam is None and self.live.active != was:
            self.broadcast({"type": "active", "active": self.live.active})
        for ev in new_events:
            if ev["cls"] == "knock" and self.knocks is not None:
                self.knocks.add(ev)
            else:
                self.report(ev)

    def report(self, ev: dict) -> None:
        self.event_id += 1
        ev = {**ev, "id": self.event_id, "wall": self.started_wall + ev["t0"]}
        if self.spl_offset is not None:
            ev["spl"] = round(ev["laf"] + self.spl_offset, 1) if ev.get("laf") is not None else None
        if self.trial is not None:
            self.trial["events"] += 1
        if self.verify is not None:
            self.verify["events"].append(ev["id"])
        self.events.append(ev)
        log.info("event %s score %.1f %s", ev["cls"], ev["score"], ev["info"])
        self.broadcast({"type": "event", "event": ev})

    def flush(self) -> None:
        if not self.clients or not self.outbox:
            self.outbox = []
            return
        f = M.concat_frames(self.outbox)
        self.outbox = []
        digits = {"t": 3, "periodicity": 2, "voice_ratio": 2}   # t needs ms: frames are 10 ms apart
        msg = {"type": "frames", "cols": {c: np.round(f[c], digits.get(c, 1)).tolist() for c in SEND_COLS}}
        msg["spec_t"] = np.round(f["t"][1::2], 2).tolist()      # every other spectrum (50 per second)
        msg["spec"] = np.round(f["spec"][1::2]).astype(int).tolist()
        self.broadcast(msg)

    def broadcast(self, msg: dict) -> None:
        text = json.dumps(msg)
        for q in list(self.clients.values()):
            if q.qsize() < 200:            # a stalled browser drops data instead of piling it up
                q.put_nowait(text)

    async def serve(self, ws) -> None:
        """One queue and one sender per browser, so messages arrive in order."""
        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait(json.dumps({"type": "hello", "state": self.state(), "events": list(self.events)}))
        self.clients[ws] = q
        try:
            while True:
                await ws.send_text(await q.get())
        finally:
            self.clients.pop(ws, None)

    # -- state
    def stale_reasons(self) -> list[str]:
        if not self.cal:
            return ["no calibration saved yet"]
        mic = self.cal.get("mic", {})
        out = []
        if mic.get("device") != self.source.name:
            out.append(f"calibrated on {mic.get('device')!r}, now {self.source.name!r}")
        if mic.get("fs") != M.FS:
            out.append(f"calibrated at {mic.get('fs')} Hz")
        g = (self.gain or {}).get("raw")
        if mic.get("gain_raw") != g:
            out.append(f"mic gain was {mic.get('gain_raw')} when calibrated, now {g}: recalibrate")
        return out

    def state(self) -> dict:
        audio_s = self.blocks * BLOCK / M.FS
        return {
            "source": self.source.name, "card": self.source.card, "fs": M.FS,
            "path": "ALSA hw, PipeWire bypassed" if self.source.card is not None else
                    ("file" if isinstance(self.source, FileSource) else "system default input"),
            "gain": self.gain, "overflows": self.source.overflows,
            "cpu_pct": round(100 * self.cpu_s / audio_s, 1) if audio_s > 1 else None,
            "t": self.meter.hops * D.DT, "started_wall": self.started_wall,
            "floor_a": _num(self.meter.floor_a), "floor_c": _num(self.meter.floor_c),
            "params": self.params, "defaults": D.DEFAULT_PARAMS, "spl_offset_db": self.spl_offset,
            "calibration": {"saved": bool(self.cal), "saved_at": (self.cal or {}).get("saved_at"),
                            "stale": self.stale_reasons()},
            "session": self.session.name, "sessions": store.sessions(), "trials": self.session.trials,
            "recording": None if self.trial is None else {k: v for k, v in self.trial.items() if k != "blocks"}
                         | {"elapsed_s": round(len(self.trial["blocks"]) * BLOCK / M.FS, 1)},
            "results": self.results, "verify": self.verify_summary(),
            "storage": str(store.ROOT),
        }

    def set_params(self, params: dict) -> dict:
        self.params = overlay(self.params, params)
        self.live.set_params(self.params)
        return self.params

    # -- trials
    def start_trial(self, meta: dict) -> dict:
        if self.trial is not None:
            raise ValueError("a trial is already recording; stop it first")
        if meta.get("kind") not in TRIAL_KINDS:
            raise ValueError(f"kind must be one of {TRIAL_KINDS}")
        if meta["kind"] == "positive" and meta.get("cls") not in D.CLASSES:
            raise ValueError(f"a positive trial needs cls, one of {D.CLASSES}")
        pre = [b for b, _, _ in self.preroll]
        floor_a, floor_c = (self.preroll[0][1], self.preroll[0][2]) if self.preroll else (math.nan, math.nan)
        self.trial = {"meta": {"label": str(meta.get("label") or meta["kind"]), "kind": meta["kind"],
                               "cls": meta.get("cls") if meta["kind"] == "positive" else None,
                               "expected": int(meta.get("expected") or (1 if meta["kind"] == "positive" else 0)),
                               "pre_frames": len(pre) * BLOCK // M.HOP,
                               "floor_a": _num(floor_a), "floor_c": _num(floor_c),
                               "gain_raw": (self.gain or {}).get("raw"), "device": self.source.name,
                               "note": meta.get("note", "")},
                      "pre": pre, "blocks": [], "events": 0, "t_start": self.meter.hops * D.DT}
        return self.trial["meta"]

    async def stop_trial(self) -> dict | None:
        if self.trial is None:
            return None
        tr, self.trial = self.trial, None
        samples = np.concatenate(tr["pre"] + tr["blocks"]) if (tr["pre"] or tr["blocks"]) else np.zeros(0, np.int16)
        trial = self.session.add(tr["meta"], samples)
        result = await asyncio.to_thread(self.score_one, trial)
        return {"trial": trial, "result": result}

    def score_one(self, trial: dict) -> dict:
        res = D.count_trial(trial, self.session.frames(trial["id"]), self.params)
        out = {"counts": res["counts"], "events": [_brief(e) for e in res["events"]]}
        if trial["kind"] in ("positive", "negative"):
            out.update(D.judge(trial, res["counts"]))
        return out

    # -- verify run
    def verify_summary(self) -> dict | None:
        v = self.verify if self.verify is not None else self.results.get("verify")
        if v is None:
            return None
        if "marks" not in v:          # a saved summary
            return v
        per = {c: {"hits": 0, "misses": v["missed"].get(c, 0), "false_alarms": 0} for c in D.CLASSES}
        by_id = {e["id"]: e for e in self.events}
        unmarked = 0
        for eid in v["events"]:
            mark = v["marks"].get(str(eid))
            cls = by_id.get(eid, {}).get("cls")
            if cls is None:
                continue
            if mark == "correct":
                per[cls]["hits"] += 1
            elif mark == "false":
                per[cls]["false_alarms"] += 1
            else:
                unmarked += 1
        hits = sum(p["hits"] for p in per.values())
        misses = sum(p["misses"] for p in per.values())
        fas = sum(p["false_alarms"] for p in per.values())
        minutes = ((v.get("stopped_t") or self.meter.hops * D.DT) - v["started_t"]) / 60.0
        rate = hits / (hits + misses) if hits + misses else None
        return {"active": self.verify is not None, "minutes": round(minutes, 2), "per_class": per,
                "hits": hits, "misses": misses, "false_alarms": fas, "unmarked": unmarked,
                "detection_rate": round(rate, 3) if rate is not None else None,
                "false_alarms_per_min": round(fas / minutes, 2) if minutes > 0 else None,
                "meets_target": bool(rate is not None and rate >= 0.95 and fas == 0 and minutes >= 2.0 and unmarked == 0),
                "marks": v["marks"], "missed": v["missed"], "events": v["events"], "started_t": v["started_t"]}


def overlay(base: dict, update: dict | None) -> dict:
    """base with update on top; class dicts are merged key by key, so {"bang": {"threshold": 20}}
    changes only that threshold."""
    out = copy.deepcopy(base)
    for k, v in (update or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return D.merged_params(out)


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 2)


def _brief(e: dict) -> dict:
    return {k: e[k] for k in ("cls", "t0", "len_s", "score", "level", "laf", "clipped", "info")}


# ---------------------------------------------------------------------------- analyses
def analyze_linearity(f: dict, trial: dict, schedule: list[dict], lead_s: float) -> dict:
    """Played pink-noise steps vs what the mic measured. schedule: [{start, dur, db, role}] in
    seconds from the start of playback; roles: 'step', 'hold_a', 'burst', 'hold_b', 'gap'."""
    pre = trial.get("pre_frames", 0)
    t = f["t"] - pre * D.DT                         # seconds since the trial started
    la, clip = f["la"], f["clip"]
    bg_mask = (t > 0.1) & (t < lead_s - 0.15)
    if bg_mask.sum() < 10:
        bg_mask = t < 0
    bg = 10 * np.log10(np.mean(10 ** (la[bg_mask] / 10)))
    # Align to where the first step actually starts (speaker and network latency).
    first = schedule[0]
    expect = lead_s + first["start"]
    win = (t > expect - 0.6) & (t < expect + 0.8)
    rise = np.flatnonzero(win & (la > bg + 6))
    offset = float(t[rise[0]] - expect) if len(rise) else 0.0
    offset = float(np.clip(offset, -0.5, 0.6))

    def seg_level(seg, guard=0.5):
        s0 = lead_s + offset + seg["start"] + guard
        s1 = lead_s + offset + seg["start"] + seg["dur"] - guard
        m = (t >= s0) & (t < s1)
        if not m.any():
            return None, False, None
        e = np.mean(10 ** (la[m] / 10))
        e_bg = 10 ** (bg / 10)
        snr = 10 * np.log10(e / e_bg)
        level = 10 * np.log10(max(e - e_bg, 1e-30)) if snr > 3 else None
        return level, bool(clip[m].sum() > 0), snr

    steps = []
    for seg in schedule:
        if seg["role"] != "step":
            continue
        lvl, clipped, snr = seg_level(seg)
        steps.append({"played_db": seg["db"], "measured_db": None if lvl is None else round(lvl, 2),
                      "snr_db": None if snr is None else round(snr, 1), "clipped": clipped})
    good = [s for s in steps if s["measured_db"] is not None and not s["clipped"] and s["snr_db"] >= 10]
    res: dict = {"background_db": round(bg, 2), "offset_s": round(offset, 3), "steps": steps}
    verdicts = []
    if len(good) >= 3:
        x = np.array([s["played_db"] for s in good], float)
        y = np.array([s["measured_db"] for s in good], float)
        slope, icpt = np.polyfit(x, y, 1)
        resid = y - (slope * x + icpt)
        res.update(slope=round(float(slope), 3), intercept=round(float(icpt), 2),
                   max_residual_db=round(float(np.abs(resid).max()), 2))
        if abs(slope - 1.0) <= 0.05 and np.abs(resid).max() <= 1.0:
            verdicts.append("LINEAR")
        else:
            verdicts.append("NOT LINEAR (compression or AGC?)")
    else:
        verdicts.append("TOO QUIET: fewer than 3 steps were 10 dB over the room; turn the laptop up or move it closer")
    if any(s["clipped"] for s in steps):
        verdicts.append("CLIPPED on the loudest step: turn the laptop down (or lower the mic gain)")
    roles = {seg["role"]: seg for seg in schedule}
    if "hold_a" in roles and "hold_b" in roles:
        a, _, snr_a = seg_level(roles["hold_a"])
        b, _, _ = seg_level(roles["hold_b"])
        hb = roles["hold_b"]
        b_early, _, _ = seg_level({"start": hb["start"], "dur": 1.5}, guard=0.25)
        b_late, _, _ = seg_level({"start": hb["start"] + hb["dur"] - 1.5, "dur": 1.5}, guard=0.25)
        if a is not None and b is not None:
            res["agc_delta_db"] = round(b - a, 2)
            if b_early is not None and b_late is not None:
                res["recovery_db"] = round(b_late - b_early, 2)
            agc = abs(b - a) > 1.0 or abs(res.get("recovery_db", 0.0)) > 1.0
            verdicts.append("AGC SUSPECTED: the level after the loud burst differs, or creeps back" if agc else "NO AGC")
    res["verdict"] = verdicts
    res["ok"] = verdicts[:1] == ["LINEAR"] and "NO AGC" in verdicts and not any(v.startswith("CLIPPED") for v in verdicts)
    return res


def analyze_quiet(f: dict, trial: dict) -> dict:
    pre = trial.get("pre_frames", 0)
    la, lc = f["la"][pre:], f["lc"][pre:]
    if len(la) < 100:
        raise ValueError("record at least a few seconds of the quiet room")
    return {"seconds": round(len(la) * D.DT, 1),
            "la_median": round(float(np.median(la)), 2), "la_p90": round(float(np.percentile(la, 90)), 2),
            "lc_median": round(float(np.median(lc)), 2), "lc_p90": round(float(np.percentile(lc, 90)), 2),
            "floor_a": round(float(np.percentile(la, M.FLOOR_PCT)), 2),
            "floor_c": round(float(np.percentile(lc, M.FLOOR_PCT)), 2),
            "above_16bit_db": round(float(np.percentile(la, M.FLOOR_PCT)) - (-101.0), 1)}


def speaker_test_wav() -> Path:
    """Speech-like sound plus a sweep, for measuring how loud Pupper's own speaker is in its mic."""
    x = np.concatenate([Y.talk(3.0, -18.0, seed=21), np.zeros(M.FS // 4)])
    t = np.arange(int(1.5 * M.FS)) / M.FS
    sweep = 0.25 * np.sin(2 * np.pi * (200 * t + (8000 - 200) / (2 * 1.5) * t * t))
    path = Path(tempfile.gettempdir()) / "pupper_speaker_test.wav"
    store.write_wav(path, Y.to_int16(np.concatenate([x, sweep])))
    return path


def sink_volume() -> str | None:
    try:
        return subprocess.run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"], capture_output=True, text=True,
                              timeout=3, env={**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"}).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------- web app
def make_app(engine: Engine):
    """The web app. Every endpoint is async, so it runs on the same thread as the audio loop and
    never races it; heavy work (re-measuring trials, fitting) goes to a worker thread."""
    E = engine

    @asynccontextmanager
    async def lifespan(app):
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        E.source.start(loop, q)
        task = loop.create_task(E.run(q))
        yield
        task.cancel()
        E.source.stop()

    app = FastAPI(title="Pupper hearing calibration", lifespan=lifespan)

    def bad(e: Exception):
        raise HTTPException(400, str(e))

    @app.get("/")
    async def index():
        return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/health")
    async def health():
        return {"ok": True, "source": E.source.name, "t": E.meter.hops * D.DT}

    @app.get("/state")
    async def state():
        return E.state()

    @app.get("/events")
    async def events():
        return list(E.events)

    @app.post("/params")
    async def params(body: dict = Body(...)):
        return E.set_params(body)

    @app.post("/params/reset")
    async def params_reset():
        E.params = D.merged_params(E.cal["params"] if E.cal else None)
        E.live.set_params(E.params)
        return E.params

    @app.post("/trial/start")
    async def trial_start(body: dict = Body(...)):
        try:
            return E.start_trial(body)
        except ValueError as e:
            bad(e)

    @app.post("/trial/stop")
    async def trial_stop():
        res = await E.stop_trial()
        if res is None:
            raise HTTPException(400, "no trial is recording")
        return res

    @app.post("/trial/cancel")
    async def trial_cancel():
        E.trial = None
        return {"ok": True}

    @app.patch("/trial/{tid}")
    async def trial_patch(tid: str, body: dict = Body(...)):
        allowed = {k: body[k] for k in ("label", "kind", "cls", "expected", "note") if k in body}
        if "expected" in allowed:
            allowed["expected"] = int(allowed["expected"])
        try:
            return E.session.update(tid, **allowed)
        except KeyError as e:
            raise HTTPException(404, str(e))

    @app.delete("/trial/{tid}")
    async def trial_delete(tid: str):
        try:
            E.session.delete(tid)
        except KeyError as e:
            raise HTTPException(404, str(e))
        return {"ok": True}

    def scored(params: dict) -> dict:
        trials = [t for t in E.session.trials if t["kind"] in ("positive", "negative")]
        res = D.score_trials(trials, E.session.all_frames(trials), params)
        for row in res["trials"]:
            row["events"] = [_brief(e) for e in row["events"]]
        return res

    @app.post("/score")
    async def score(body: dict = Body(default={})):
        return await asyncio.to_thread(scored, overlay(E.params, body.get("params")))

    @app.post("/fit")
    async def fit(body: dict = Body(default={})):
        trials = [t for t in E.session.trials if t["kind"] in ("positive", "negative")]
        if not trials:
            raise HTTPException(400, "record some positive and negative trials first")
        classes = tuple(body.get("classes") or D.CLASSES)
        frames = await asyncio.to_thread(E.session.all_frames, trials)
        rep = await asyncio.to_thread(D.fit, trials, frames, E.params, classes)
        E.results["fit"] = rep
        return rep

    @app.post("/analyze/linearity")
    async def linearity(body: dict = Body(...)):
        try:
            tr = E.session.get(body["trial_id"])
        except KeyError as e:
            raise HTTPException(404, str(e))
        f = await asyncio.to_thread(E.session.frames, tr["id"])
        res = analyze_linearity(f, tr, body["schedule"], float(body.get("lead_s", 1.0)))
        E.results["linearity"] = {**res, "trial": tr["id"], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        return E.results["linearity"]

    @app.post("/analyze/quiet")
    async def quiet(body: dict = Body(...)):
        try:
            tr = E.session.get(body["trial_id"])
            f = await asyncio.to_thread(E.session.frames, tr["id"])
            res = analyze_quiet(f, tr)
        except (KeyError, ValueError) as e:
            bad(e)
        E.results["quiet"] = {**res, "trial": tr["id"], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        return E.results["quiet"]

    @app.post("/floor_ref")
    async def floor_ref(body: dict = Body(...)):
        """Use the quiet-room floor as the reference (or clear it with {"clear": true})."""
        if body.get("clear"):
            return E.set_params({"floor_ref_a": None, "floor_ref_c": None})
        q = E.results.get("quiet")
        if not q:
            raise HTTPException(400, "record the quiet room first")
        return E.set_params({"floor_ref_a": q["floor_a"], "floor_ref_c": q["floor_c"]})

    @app.post("/analyze/spl")
    async def spl(body: dict = Body(...)):
        reading = float(body["reading_dba"])
        w = E.window(3.0)
        if len(w["t"]) < 200:
            raise HTTPException(400, "not enough audio yet")
        e = 10 ** (w["la"] / 10)
        laeq = 10 * np.log10(e.mean())
        spread = float(np.std(w["laf"]))
        E.spl_offset = round(reading - laeq, 2)
        return {"laeq_dbfs": round(float(laeq), 2), "reading_dba": reading, "spl_offset_db": E.spl_offset,
                "laf_spread_db": round(spread, 2), "steady": spread < 1.5}

    @app.post("/spl/clear")
    async def spl_clear():
        E.spl_offset = None
        return {"ok": True}

    @app.post("/gain")
    async def gain(body: dict = Body(...)):
        card = E.source.card
        if card is None:
            raise HTTPException(400, "this input has no ALSA hardware gain")
        value = int(body["value"])
        g = E.gain or {}
        if "min" in g and not g["min"] <= value <= g["max"]:
            raise HTTPException(400, f"gain must be {g['min']}..{g['max']}")
        r = subprocess.run(["amixer", "-c", str(card), "cset", "name=Mic Capture Volume", str(value)],
                           capture_output=True, text=True, timeout=3)
        if r.returncode != 0:
            raise HTTPException(500, r.stderr.strip() or "amixer failed")
        E.gain = read_gain(card)
        return {"gain": E.gain, "stale": E.stale_reasons()}

    @app.post("/speaker/play")
    async def speaker_play():
        path = speaker_test_wav()
        env = {**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"}
        try:
            proc = await asyncio.create_subprocess_exec("pw-play", str(path), env=env)
            await asyncio.wait_for(proc.wait(), 15)
        except (OSError, asyncio.TimeoutError) as e:
            raise HTTPException(500, f"could not play through the speaker: {e}")
        return {"ok": True, "sink_volume": sink_volume()}

    @app.post("/verify/start")
    async def verify_start():
        E.verify = {"started_t": E.meter.hops * D.DT, "events": [], "marks": {}, "missed": {}}
        return E.verify_summary()

    @app.post("/verify/stop")
    async def verify_stop():
        if E.verify is None:
            raise HTTPException(400, "no verify run")
        E.verify["stopped_t"] = E.meter.hops * D.DT
        s = E.verify_summary()
        s["active"] = False
        s["params"] = {c: E.params[c]["threshold"] for c in D.CLASSES}
        s["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        E.results["verify"] = {k: v for k, v in s.items() if k not in ("marks", "events")}
        E.verify = None
        return E.results["verify"]

    @app.post("/verify/mark")
    async def verify_mark(body: dict = Body(...)):
        if E.verify is None:
            raise HTTPException(400, "start a verify run first")
        if "missed" in body:
            cls = body["missed"]
            E.verify["missed"][cls] = E.verify["missed"].get(cls, 0) + 1
        else:
            E.verify["marks"][str(int(body["event_id"]))] = body["mark"]
        return E.verify_summary()

    @app.post("/session/new")
    async def session_new():
        if E.trial is not None:
            raise HTTPException(400, "stop the recording first")
        E.session = store.Session(store.new_session_name())
        return {"session": E.session.name}

    @app.post("/session/open")
    async def session_open(body: dict = Body(...)):
        name = str(body["name"])
        if name not in store.sessions():
            raise HTTPException(404, f"no session {name!r}")
        E.session = store.Session(name)
        return {"session": name}

    @app.get("/calibration")
    async def calibration():
        return store.load_calibration() or {}

    @app.post("/calibration/save")
    async def calibration_save(body: dict = Body(default={})):
        summary = await asyncio.to_thread(scored, E.params)
        cal = {
            "version": 1, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "note": body.get("note", ""),
            "mic": {"device": E.source.name, "card": E.source.card, "fs": M.FS,
                    "gain_raw": (E.gain or {}).get("raw"), "gain_db": (E.gain or {}).get("db"),
                    "path": E.state()["path"]},
            "params": E.params, "spl_offset_db": E.spl_offset, "session": E.session.name,
            "score": {"summary": summary["summary"], "negative_minutes": summary["negative_minutes"]},
            **{k: E.results.get(k) for k in ("linearity", "quiet", "fit", "verify")},
        }
        path = store.save_calibration(cal)
        E.cal = cal
        return {"path": str(path), "calibration": cal}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            await E.serve(sock)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


# ---------------------------------------------------------------------------- CLI
def bench() -> None:
    x, _ = Y.demo_scene()
    samples = Y.to_int16(x)
    m, live, recent = M.Meter(), D.LiveDetector(), []
    t0 = time.process_time()
    for i in range(0, len(samples), BLOCK):
        recent.append(m.feed(samples[i:i + BLOCK]))
        if (i // BLOCK) % 2:
            w = M.concat_frames(recent)
            w = M.slice_frames(w, max(0, len(w["t"]) - int(RECENT_S / D.DT)))
            recent = [w]
            live.update(M.slice_frames(w, max(0, len(w["t"]) - int(D.LiveDetector.WINDOW_S / D.DT))))
    dt = time.process_time() - t0
    print(f"60 s of audio in {dt:.2f} s of CPU: {100 * dt / 60:.1f}% of one core (blocks of {BLOCK}, "
          f"live detector every 100 ms)")


def rescore(name: str | None) -> None:
    names = store.sessions()
    if not names:
        sys.exit(f"no sessions in {store.ROOT / 'sessions'}")
    name = name or names[-1]
    sess = store.Session(name)
    cal = store.load_calibration()
    params = D.merged_params(cal["params"] if cal else None)
    trials = [t for t in sess.trials if t["kind"] in ("positive", "negative")]
    frames = sess.all_frames(trials)
    res = D.score_trials(trials, frames, params)
    print(f"session {name}: {len(trials)} trials, thresholds " +
          ", ".join(f"{c} {params[c]['threshold']}" for c in D.CLASSES))
    for row in res["trials"]:
        what = f"{row['kind']}/{row['cls']}" if row["cls"] else row["kind"]
        print(f"  {row['id']} {row['label'][:30]:30s} {what:14s} expected {row['expected']}: "
              f"{row['counts']}  hits {row['hits']} misses {row['misses']} false alarms {row['false_alarms']}")
    for c, s in res["summary"].items():
        print(f"  {c:6s} {s['hits']}/{s['expected']} detected, {s['false_alarms']} false alarms")
    rep = D.fit(trials, frames, params)
    for c, r in rep.items():
        print(f"  fit {c}: threshold {r['threshold']} (worst negative {r['worst_negative']}, "
              f"weakest positive {r['weakest_positive']}) {' '.join(r['notes'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="Brio", help="input device name (substring), or 'default' (default: Brio)")
    ap.add_argument("--wav", help="loop this recording instead of the mic")
    ap.add_argument("--demo", action="store_true", help="loop a synthetic scene instead of the mic")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 (default; use the ssh tunnel) or 0.0.0.0")
    ap.add_argument("--session", help="continue this session (default: the latest, or a new one)")
    ap.add_argument("--new-session", action="store_true", help="start a new session")
    ap.add_argument("--bench", action="store_true", help="measure CPU use and exit")
    ap.add_argument("--rescore", nargs="?", const="", metavar="SESSION",
                    help="re-score a session (default: the latest) with the saved calibration and exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.bench:
        return bench()
    if args.rescore is not None:
        return rescore(args.rescore or None)

    if args.demo:
        source = FileSource(Y.to_int16(Y.demo_scene()[0]), "demo scene (synthetic)")
    elif args.wav:
        source = FileSource(store.read_wav(Path(args.wav)), f"file {Path(args.wav).name}")
    else:
        source = MicSource(args.device)
    names = store.sessions()
    if args.session:
        name = args.session
    elif args.new_session or not names:
        name = store.new_session_name()
    else:
        name = names[-1]
    engine = Engine(source, store.Session(name))

    import uvicorn

    app = make_app(engine)
    g = engine.gain
    print(f"listening to {source.name} at {M.FS} Hz" + (f", mic gain {g['raw']} ({g.get('db')} dB)" if g else ""))
    print(f"session {name}; data in {store.ROOT}")
    for r in engine.stale_reasons():
        print(f"calibration: {r}")
    print(f"open http://localhost:{args.port}  (from a laptop: ssh -N -L {args.port}:localhost:{args.port} pi@pupper5.local)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
