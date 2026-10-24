"""Synthetic test sounds at known levels: room noise, claps, thuds, shouts, talking, alarms.

Used by test_signal.py, and by `mic_lab.py --demo` to drive the UI without a mic. Every function
returns float64 samples at 48 kHz, where 1.0 is full scale; to_int16() converts for the meter.
Levels are dBFS RMS of the sound itself (for a clap, over its loudest 10 ms).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

FS = 48000


def _rng(seed):
    return np.random.default_rng(seed)


def gain_for(x: np.ndarray, level_db: float, over: int | None = None) -> np.ndarray:
    """Scale x so its RMS (over its loudest `over` samples, if given) is level_db dBFS."""
    if over:
        e = np.convolve(x * x, np.ones(over) / over, "valid").max()
    else:
        e = np.mean(x * x)
    return x * (10 ** (level_db / 20) / np.sqrt(max(e, 1e-30)))


def pink(seconds: float, level_db: float, seed=0) -> np.ndarray:
    """Pink noise (Paul Kellet's filter), like a quiet room with a fan."""
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    x = signal.lfilter(b, a, _rng(seed).standard_normal(int(seconds * FS) + 4800))[4800:]
    return gain_for(x, level_db)


def clap(level_db: float, seed=1) -> np.ndarray:
    """A hand clap: a 1 ms attack, a 30 ms decay, mostly 500-5000 Hz. 300 ms long."""
    n = int(0.3 * FS)
    t = np.arange(n) / FS
    x = _rng(seed).standard_normal(n) * np.minimum(1.0, t / 0.001) * np.exp(-t / 0.03)
    x = signal.sosfilt(signal.butter(2, [500, 5000], "bandpass", fs=FS, output="sos"), x)
    return gain_for(x, level_db, over=480)


def thud(level_db: float, seed=2) -> np.ndarray:
    """Something heavy hitting the floor: low-frequency (below 200 Hz), 60 ms decay. 400 ms long."""
    n = int(0.4 * FS)
    t = np.arange(n) / FS
    x = _rng(seed).standard_normal(n) * np.minimum(1.0, t / 0.002) * np.exp(-t / 0.06)
    x = signal.sosfilt(signal.butter(4, [40, 200], "bandpass", fs=FS, output="sos"), x)
    return gain_for(x, level_db, over=480)


def vowel(seconds: float, level_db: float, f0=(300.0, 350.0, 280.0), formants=((800, 90), (1300, 110), (2700, 170)),
          seed=3, edge_s=0.04) -> np.ndarray:
    """A sung or shouted vowel: harmonics of a gliding pitch, shaped by formants."""
    n = int(seconds * FS)
    t = np.arange(n) / FS
    pitch = np.interp(t, np.linspace(0, seconds, len(f0)), f0)
    phase = 2 * np.pi * np.cumsum(pitch) / FS
    rng = _rng(seed)
    x = np.zeros(n)
    for k in range(1, 60):
        fk = pitch * k
        if fk.min() > 8000:
            break
        g = sum(1.0 / (1.0 + ((fk - F) / B) ** 2) for F, B in formants)
        x += g * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    x += 0.02 * rng.standard_normal(n) * np.abs(x).max()      # breath
    env = np.minimum(1.0, np.minimum(t, seconds - t) / edge_s)
    return gain_for(x * np.clip(env, 0, 1), level_db)


def shout(level_db: float, seconds=0.8, seed=3) -> np.ndarray:
    return vowel(seconds, level_db, seed=seed)


def talk(seconds: float, level_db: float, seed=4) -> np.ndarray:
    """Conversation: 120-250 ms syllables at a low pitch, with short gaps."""
    rng = _rng(seed)
    out = np.zeros(int(seconds * FS))
    i = 0
    while i < len(out):
        dur = rng.uniform(0.12, 0.25)
        f = rng.uniform(110, 170)
        syl = vowel(dur, level_db, f0=(f, f * rng.uniform(0.9, 1.1)), seed=int(rng.integers(1e9)),
                    formants=((rng.uniform(400, 800), 90), (rng.uniform(1000, 2000), 110), (2600, 170)), edge_s=0.02)
        j = min(len(out), i + len(syl))
        out[i:j] += syl[: j - i]
        i = j + int(rng.uniform(0.05, 0.2) * FS)
    return out


def tone_pattern(on_off: list[tuple[float, float]], hz: float, level_db: float, repeats: int) -> np.ndarray:
    """A beeping tone: (on seconds, off seconds) pairs, repeated. Level is the RMS while on."""
    parts = []
    for _ in range(repeats):
        for on, off in on_off:
            t = np.arange(int(on * FS)) / FS
            ramp = np.minimum(1.0, np.minimum(t, on - t) / 0.003)
            parts += [np.sin(2 * np.pi * hz * t) * ramp * np.sqrt(2) * 10 ** (level_db / 20), np.zeros(int(off * FS))]
    return np.concatenate(parts)


def smoke_alarm(level_db: float, repeats=3, hz=3100.0) -> np.ndarray:
    """The temporal-3 (T3) pattern every smoke alarm uses: three 0.5 s beeps, then 1.5 s quiet."""
    return tone_pattern([(0.5, 0.5), (0.5, 0.5), (0.5, 1.5)], hz, level_db, repeats)


def timer_beeps(level_db: float, seconds=4.0, hz=2700.0) -> np.ndarray:
    """A kitchen timer: four 0.1 s beeps a second."""
    return tone_pattern([(0.1, 0.15)], hz, level_db, int(seconds / 0.25))


def siren(level_db: float, seconds=6.0, kind="wail") -> np.ndarray:
    """An emergency siren: wail (600-1500 Hz, 4 s up and down), yelp (the same, 3 times a second)
    or hilo (two tones, 0.5 s each). Electronic sirens are square-ish: odd harmonics included."""
    t = np.arange(int(seconds * FS)) / FS
    if kind == "hilo":
        f = np.where((t % 1.0) < 0.5, 960.0, 770.0)
    else:
        rate = 0.25 if kind == "wail" else 3.0
        f = 1050.0 + 450.0 * np.sin(2 * np.pi * rate * t - np.pi / 2)
    phase = 2 * np.pi * np.cumsum(f) / FS
    x = np.sin(phase) + np.sin(3 * phase) / 3 + np.sin(5 * phase) / 5
    x *= np.minimum(1.0, np.minimum(t, seconds - t) / 0.05)
    return gain_for(x, level_db)


def place(bed: np.ndarray, sound: np.ndarray, at_s: float) -> np.ndarray:
    """Add sound into bed starting at at_s seconds (in place; returns bed)."""
    i = int(at_s * FS)
    j = min(len(bed), i + len(sound))
    bed[i:j] += sound[: j - i]
    return bed


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16)


def demo_scene(floor_db=-60.0, seed=0) -> tuple[np.ndarray, list[tuple[float, str]]]:
    """60 s of room noise (ends with a siren) with known events, for the --demo mode and the live-vs-batch test.
    Returns the samples and [(time, what)] of what was placed."""
    x = pink(60.0, floor_db, seed=seed)
    marks = []
    for at, what, snd in [
        (8.0, "clap", clap(floor_db + 35)), (10.0, "clap", clap(floor_db + 30, seed=5)),
        (14.0, "talk", talk(6.0, floor_db + 10)), (23.0, "shout", shout(floor_db + 32)),
        (28.0, "thud", thud(floor_db + 35)), (33.0, "smoke alarm", smoke_alarm(floor_db + 30, repeats=2)),
        (44.0, "talk", talk(5.0, floor_db + 12, seed=9)), (51.0, "timer", timer_beeps(floor_db + 28)),
        (56.0, "siren", siren(floor_db + 30, seconds=3.8)),
    ]:
        place(x, snd, at)
        marks.append((at, what))
    return x, marks
