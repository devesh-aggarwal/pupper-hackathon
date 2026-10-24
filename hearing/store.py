"""Where calibration data lives: trial recordings, their labels, and the saved calibration.

Everything is outside the repo, in ~/.config/gemini-pupper/hearing/ (or $PUPPER_HEARING_DIR):

    calibration.json             the calibration the tool (and later the robot) uses
    sessions/<name>/trials.json  the labelled trials of one calibration run
    sessions/<name>/<id>.wav     each trial's audio: 16-bit mono 48 kHz, starting with a pre-roll

A trial's WAV starts with a few seconds of pre-roll (the audio just before you pressed Record),
so the meter's filters and onset history have settled before the trial starts. The floor at the
time is saved too, and re-analysis starts from it, so a re-scored trial reads like it did live.
"""

from __future__ import annotations

import json
import os
import time
import wave
from pathlib import Path

import numpy as np

import meter

ROOT = Path(os.environ.get("PUPPER_HEARING_DIR", "~/.config/gemini-pupper/hearing")).expanduser()


def write_wav(path: Path, samples: np.ndarray, fs: int = meter.FS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(np.asarray(samples, np.int16).tobytes())


def read_wav(path: Path) -> np.ndarray:
    """Any WAV as int16 mono at 48 kHz (first channel; resampled if needed)."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    fs, x = wavfile.read(str(path))
    if x.ndim > 1:
        x = x[:, 0]
    if x.dtype == np.int16:
        y = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        y = x.astype(np.float64) / 2 ** 31
    elif x.dtype == np.uint8:
        y = (x.astype(np.float64) - 128) / 128.0
    else:
        y = x.astype(np.float64)
    if fs != meter.FS:
        g = np.gcd(int(fs), meter.FS)
        y = resample_poly(y, meter.FS // g, int(fs) // g)
    return np.clip(np.round(y * 32768), -32768, 32767).astype(np.int16)


class Session:
    """The labelled trials of one calibration run."""

    def __init__(self, name: str, root: Path = ROOT):
        self.name = name
        self.dir = root / "sessions" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = self.dir / "trials.json"
        self.trials: list[dict] = json.loads(self.index.read_text()) if self.index.exists() else []
        self._frames: dict[str, dict] = {}

    def _save(self) -> None:
        tmp = self.index.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.trials, indent=1))
        tmp.replace(self.index)

    def add(self, meta: dict, samples: np.ndarray) -> dict:
        n = 1 + max([int(t["id"][1:]) for t in self.trials] or [0])
        tid = f"t{n:03d}"
        write_wav(self.dir / f"{tid}.wav", samples)
        trial = {"id": tid, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "duration_s": round(len(samples) / meter.FS - meta.get("pre_frames", 0) * meter.HOP / meter.FS, 2),
                 **meta}
        self.trials.append(trial)
        self._save()
        return trial

    def update(self, tid: str, **fields) -> dict:
        tr = self.get(tid)
        tr.update(fields)
        self._save()
        return tr

    def delete(self, tid: str) -> None:
        tr = self.get(tid)
        self.trials.remove(tr)
        (self.dir / f"{tid}.wav").unlink(missing_ok=True)
        self._frames.pop(tid, None)
        self._save()

    def get(self, tid: str) -> dict:
        for tr in self.trials:
            if tr["id"] == tid:
                return tr
        raise KeyError(f"no trial {tid!r} in session {self.name!r}")

    def samples(self, tid: str) -> np.ndarray:
        return read_wav(self.dir / f"{tid}.wav")

    def frames(self, tid: str) -> dict:
        """The trial's features, re-measured from its WAV (cached: the audio never changes)."""
        if tid not in self._frames:
            tr = self.get(tid)
            seed = (tr["floor_a"], tr["floor_c"]) if tr.get("floor_a") is not None else None
            self._frames[tid] = meter.analyze(self.samples(tid), seed_floor=seed)
        return self._frames[tid]

    def all_frames(self, trials: list[dict] | None = None) -> dict[str, dict]:
        return {tr["id"]: self.frames(tr["id"]) for tr in (trials if trials is not None else self.trials)}


def sessions(root: Path = ROOT) -> list[str]:
    d = root / "sessions"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def new_session_name() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def load_calibration(root: Path = ROOT) -> dict | None:
    p = root / "calibration.json"
    return json.loads(p.read_text()) if p.exists() else None


def save_calibration(cal: dict, root: Path = ROOT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / "calibration.json"
    if p.exists():  # keep the previous one, in case the new calibration turns out worse
        p.replace(root / "calibration.prev.json")
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cal, indent=1))
    tmp.replace(p)
    return p
