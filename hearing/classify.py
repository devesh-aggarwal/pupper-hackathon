"""What a sound IS: Google's YAMNet (trained on AudioSet, 521 sound classes) through LiteRT, on-device.

The meter (meter.py) is good at how loud a sound is. Telling a siren from a person whistling is a
learned problem, so sirens, alarms, doorbells, crashes and a baby crying come from YAMNet. YAMNet is poor at knocks
(it calls them "Inside, small room" or "Door"), so knocking is the meter's rhythm detector
(detect.knock_events), with KnockGate dropping the knocks YAMNet hears as speech. Bangs and shouts still come from the
calibrated meter (detect.py).

YAMNet takes 0.975 s of 16 kHz audio and scores all 521 classes; this runs it every 0.5 s on the
last 0.975 s (about 1.5 ms each on a laptop). Classes are summed into groups (GROUPS). A siren
alarm or doorbell event starts when its group's score is >= on_score in MIN_WINDOWS of the last
3 windows (2 of 3, about 1 s; a doorbell: 1), and only while the sound is >= min_above_db over
the room; it is one event until the group stays under off_score for rearm_s.

Setup, once (on the Pupper; nothing goes to the cloud after this):
    python3 hearing/classify.py --install     # LiteRT into its own folder + the YAMNet model
    python3 hearing/classify.py some.wav      # what YAMNet hears, every 0.5 s
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import signal

HOME = Path(os.environ.get("PUPPER_HEARING_DIR", "~/.config/gemini-pupper/hearing")).expanduser()
MODEL_DIR = HOME / "yamnet"
LIB_DIR = HOME / "litert"   # ai-edge-litert lives here, away from the system's Python packages
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
CLASSES_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

GROUPS = {
    "siren": ["Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)",
              "Fire engine, fire truck (siren)", "Emergency vehicle"],
    "alarm": ["Smoke detector, smoke alarm", "Fire alarm", "Alarm", "Alarm clock", "Car alarm",
              "Beep, bleep", "Buzzer"],
    "shout": ["Shout", "Yell", "Children shouting", "Screaming", "Bellow"],
    "bang": ["Bang", "Slam", "Smash, crash", "Breaking", "Glass", "Shatter", "Thump, thud",
             "Explosion", "Gunshot, gunfire"],
    "doorbell": ["Doorbell", "Ding-dong"],
    "crash": ["Smash, crash", "Breaking", "Glass", "Shatter"],
    "knock": ["Knock"],
    "speech": ["Speech", "Conversation", "Narration, monologue", "Child speech, kid speaking"],
    "baby": ["Baby cry, infant cry"],
}
# Groups that raise events here (a siren wins over an alarm).
EVENT_GROUPS = ("siren", "alarm", "doorbell", "crash", "baby")   # knocking comes from the meter (KnockGate)
# Windows (of the last 3) that must score >= on_score: a doorbell, a crash or a knock lasts only a
# second or two; a baby crying must go on for 2 windows, so one squeal isn't enough.
MIN_WINDOWS = {"siren": 2, "alarm": 2, "doorbell": 1, "crash": 1, "knock": 1, "baby": 2}
REARM_S = {"doorbell": 4.0, "crash": 4.0, "knock": 4.0}   # a second one after this is a new event (others: rearm_s)
QUIET = {"Silence"}                 # what the screen's corner never shows
DEFAULTS = {"on_score": 0.3, "off_score": 0.15, "min_above_db": 6.0, "rearm_s": 8.0}

FS_IN, FS = 48000, 16000
WIN = 15600        # 0.975 s at 16 kHz
HOP = 8000         # 0.5 s


def available() -> bool:
    return (MODEL_DIR / "yamnet.tflite").exists() and (MODEL_DIR / "yamnet_class_map.csv").exists()


def _interpreter(path: Path):
    if str(LIB_DIR) not in sys.path and LIB_DIR.exists():
        sys.path.insert(0, str(LIB_DIR))
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path), num_threads=1)
    it.allocate_tensors()
    return it


class SoundClassifier:
    """Feed 48 kHz int16 blocks; every 0.5 s, get YAMNet's view and any new siren/alarm event."""

    def __init__(self, params: dict | None = None):
        self.p = {**DEFAULTS, **(params or {})}
        self.names = [r["display_name"] for r in csv.DictReader(open(MODEL_DIR / "yamnet_class_map.csv"))]
        self.groups = {g: [self.names.index(n) for n in ns if n in self.names] for g, ns in GROUPS.items()}
        self.it = _interpreter(MODEL_DIR / "yamnet.tflite")
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.fir = signal.firwin(97, 7600, fs=FS_IN)          # anti-alias before keeping every 3rd sample
        self.zi = np.zeros(len(self.fir) - 1)
        self.phase = 0
        self.buf = np.zeros(0, np.float32)
        self.loud: list[tuple[float, float]] = []   # (time, dB over the room) of recent blocks
        self.since = 0
        self.t = 0.0                                          # seconds of audio seen
        self.history: dict[str, list[float]] = {g: [] for g in GROUPS}
        self.episode: dict[str, dict | None] = {g: None for g in EVENT_GROUPS}
        self.latest: dict = {}
        self.speech: list[tuple[float, float]] = []   # (end time of a window, its speech score)

    def feed(self, block: np.ndarray, above_db: float) -> tuple[dict | None, list[dict]]:
        """block: int16 at 48 kHz. above_db: how loud the room is right now over its floor (la - floor_a).
        Returns (YAMNet's latest view if it just ran, else None; new events).
        The loudness check uses the loudest block of the audio YAMNet just looked at: a knock is
        over in 0.1 s, long before the window is scored."""
        x = np.asarray(block, np.float64) / 32768.0
        y, self.zi = signal.lfilter(self.fir, 1.0, x, zi=self.zi)
        y = y[(-self.phase) % 3::3]
        self.phase = (self.phase + len(x)) % 3
        self.t += len(x) / FS_IN
        self.loud = [(t, a) for t, a in self.loud if t > self.t - WIN / FS] + [(self.t, above_db)]
        self.buf = np.concatenate([self.buf, y.astype(np.float32)])[-WIN:]
        self.since += len(y)
        if self.since < HOP or len(self.buf) < WIN:
            return None, []
        self.since = 0
        self.it.set_tensor(self.inp["index"], self.buf.reshape(self.inp["shape"]))
        self.it.invoke()
        scores = self.it.get_tensor(self.out["index"]).reshape(-1)
        return self._update(scores, max(a for _, a in self.loud))

    def _update(self, scores: np.ndarray, above_db: float) -> tuple[dict, list[dict]]:
        top = np.argsort(scores)[::-1][:3]
        group = {g: float(scores[idx].max()) if idx else 0.0 for g, idx in self.groups.items()}
        self.speech = [(t, v) for t, v in self.speech if t > self.t - 10.0] + [(self.t, group["speech"])]
        for g, v in group.items():
            self.history[g] = (self.history[g] + [v])[-3:]
        events = []
        loud = above_db >= self.p["min_above_db"]
        for g in EVENT_GROUPS:
            h, ep = self.history[g], self.episode[g]
            on = loud and sum(v >= self.p["on_score"] for v in h) >= MIN_WINDOWS[g]
            if ep is not None:
                if group[g] >= self.p["off_score"]:
                    ep["last"] = self.t
                    ep["score"] = max(ep["score"], group[g])
                elif self.t - ep["last"] > REARM_S.get(g, self.p["rearm_s"]):
                    self.episode[g] = ep = None
            if on and ep is None and not (g == "alarm" and self.episode["siren"] is not None):
                label = max(self.groups[g], key=lambda i: scores[i])
                self.episode[g] = {"start": self.t, "last": self.t, "score": group[g]}
                events.append({"cls": g, "t0": round(self.t - 1.0, 2), "t1": round(self.t, 2), "t": round(self.t, 2),
                               "score": round(100 * group[g], 1), "len_s": 1.0, "level": None, "laf": None,
                               "clipped": False, "info": {"label": self.names[label], "confidence": round(group[g], 2),
                                                          "above_db": round(above_db, 1)}})
        self.latest = {"top": [[self.names[i], round(float(scores[i]), 2)] for i in top],
                       "now": None if self.names[top[0]] in QUIET or scores[top[0]] < 0.2
                              else [self.names[top[0]], round(float(scores[top[0]]), 2)],
                       "groups": {g: round(v, 2) for g, v in group.items()},
                       "active": {g: self.episode[g] is not None and self.t - self.episode[g]["last"] < 1.5
                                  for g in EVENT_GROUPS}}
        return self.latest, events


class KnockGate:
    """Holds the meter's knock events until YAMNet has heard the same moment, then lets through
    only the ones it didn't hear as speech (talking makes knock-shaped peaks too)."""

    def __init__(self, classifier: SoundClassifier, max_speech: float = 0.3):
        self.c = classifier
        self.max_speech = max_speech
        self.pending: list[dict] = []

    def add(self, ev: dict) -> None:
        self.pending.append(ev)

    def ready(self) -> list[dict]:
        """Knocks YAMNet has now scored (a window ending >= 0.6 s after them) and didn't call speech."""
        out, keep = [], []
        for ev in self.pending:
            if self.c.t < ev["t1"] + 0.6:
                keep.append(ev)
                continue
            heard = [v for t, v in self.c.speech if ev["t0"] < t <= ev["t1"] + 1.0]
            speech = max(heard) if heard else 0.0
            if speech < self.max_speech:
                out.append({**ev, "info": {**ev["info"], "speech": round(speech, 2)}})
        self.pending = keep
        return out


def install() -> None:
    """LiteRT into LIB_DIR (its own folder, so the robot's Python packages don't change) and the model."""
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # numpy is already installed; everything else LiteRT needs goes into LIB_DIR only.
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--target", str(LIB_DIR), "--no-deps",
                    "ai-edge-litert", "flatbuffers"], check=True)
    for url, name in ((MODEL_URL, "yamnet.tflite"), (CLASSES_URL, "yamnet_class_map.csv")):
        if not (MODEL_DIR / name).exists():
            subprocess.run(["curl", "-sSfL", "-o", str(MODEL_DIR / name), url], check=True)
    c = SoundClassifier()
    view, _ = c.feed(np.zeros(48000, np.int16), 0.0)
    print("YAMNet ready:", MODEL_DIR, "| on silence it hears:", view["top"] if view else "?")


if __name__ == "__main__":
    if sys.argv[1:] == ["--install"]:
        install()
    elif sys.argv[1:]:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import meter
        import store
        x = store.read_wav(Path(sys.argv[1]))
        c, m = SoundClassifier(), meter.Meter()
        for i in range(0, len(x), 2400):
            f = m.feed(x[i:i + 2400])
            above = float(f["la"][-1] - f["floor_a"][-1]) if len(f["t"]) else 0.0
            view, evs = c.feed(x[i:i + 2400], above)
            for e in evs:
                print(f"EVENT {e['cls']} at {e['t']:.1f} s: {e['info']}")
            if view:
                print(f"{c.t:6.1f}s " + ", ".join(f"{n} {s:.2f}" for n, s in view["top"]))
    else:
        print(__doc__)
