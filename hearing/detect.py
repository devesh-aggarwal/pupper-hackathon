"""Loud-noise detectors (bang, shout, alarm), scoring of labelled trials, and threshold fitting.

All detection logic lives here, and it runs on the feature columns from meter.py. The same
find_events() serves the live tool (on a sliding window, through LiveDetector) and the scoring of
saved trials, so what you calibrate is exactly what runs.

Each class has one threshold, the number calibration fits, and a few fixed gates that define what
the class is (shown and editable in the UI):

  bang   a sudden, short sound: a clap, a drop, a slam, a fall.
         It starts where lc_max jumps >= onset_db above the last 300 ms, and ends when it has
         decayed decay_db from its peak (or is back within end_db of the floor). It must last
         <= max_len_s and not be a beep (at most tone_frac_max of it tonal).
         Score: peak lc_max above the C floor (dB).
  shout  a loud voice: the A-weighted Fast level held for hold_s, lasting min_len_s..max_len_s,
         and voiced (mean periodicity >= periodicity; beeps and noise are low).
         Score: the held level above the A floor (dB).
  alarm  a steady or beeping pure tone in 2-5 kHz (smoke alarms, timers): tonal frames
         (tone_prom >= threshold, la >= floor + above_floor_db, periodicity <= periodicity_max,
         which leaves out the harmonics of a voice) at one frequency (about +-hz_tol) fill
         >= duty of the last window_s.  Score: tone prominence (dB).
  knock  knocking: >= min_knocks sharp taps, each rising >= rise_db within rise_s to >= threshold
         over the C floor and falling >= fall_db within fall_s, spaced min_gap_s..max_gap_s apart.
         A knock isn't a tone: the median tone strength (tone_prom, siren_prom) over it stays
         <= tone_max_db (real knocks 17-22 dB; beeps 31, loud talking 26-51). Loud talking can still
         make the shape, so classify.KnockGate also drops the ones YAMNet calls speech.
         Score: the quietest tap's dB over the floor.
  siren  an emergency siren (wail, yelp, hi-lo): a pure-ish tone (siren_prom >= threshold,
         la >= floor + above_floor_db, periodicity <= periodicity_max, pitch >= min_hz) that moves
         SMOOTHLY: its pitch's second difference stays <= smooth_hz. Such smooth tonal frames must
         fill >= tonal_frac of the last window_s, and the pitch must move >= min_sweep_hz across it.
         On real recordings a siren fills 0.94; talking, whistling and squeals at most 0.37 (their
         pitch jitters). A steady alarm doesn't move.  Score: tone prominence (dB).

"Floor" is the meter's adaptive noise floor. If params carry floor_ref_a/floor_ref_c (the floor
measured in the quiet-room step), the floor used is max(live floor, reference): in a room quieter
than at calibration, a sound still has to reach the calibrated level.
"""

from __future__ import annotations

import copy
import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from meter import HOP, FS

DT = HOP / FS  # seconds per frame
CLASSES = ("bang", "shout", "knock", "alarm", "siren")
TONAL = ("siren", "alarm")  # fire when they start and stay on while the sound goes on; a siren wins over an alarm

DEFAULT_PARAMS: dict = {
    "bang": {"threshold": 30.0, "onset_db": 15.0, "decay_db": 20.0, "end_db": 6.0, "max_len_s": 0.6,
             "tone_frac_max": 0.5, "tone_db": 25.0},
    "shout": {"threshold": 25.0, "hold_s": 0.25, "min_len_s": 0.25, "max_len_s": 3.0, "periodicity": 0.45},
    "alarm": {"threshold": 25.0, "above_floor_db": 10.0, "periodicity_max": 0.5, "hz_tol": 25.0,
              "window_s": 3.0, "duty": 0.4, "rearm_s": 5.0},
    "knock": {"threshold": 20.0, "rise_db": 12.0, "rise_s": 0.06, "fall_db": 10.0, "fall_s": 0.15,
              "min_gap_s": 0.12, "max_gap_s": 0.8, "min_knocks": 2, "tone_max_db": 25.0},
    "siren": {"threshold": 25.0, "above_floor_db": 10.0, "periodicity_max": 0.6, "min_hz": 530.0,
              "smooth_hz": 30.0, "window_s": 3.0, "tonal_frac": 0.75, "min_sweep_hz": 150.0, "rearm_s": 8.0},
    "refractory_s": 1.0,
    "floor_ref_a": None,
    "floor_ref_c": None,
}

# The range thresholds are searched over when fitting (dB).
FIT_RANGE = (0.0, 70.0)


def merged_params(params: dict | None) -> dict:
    """DEFAULT_PARAMS with params layered on top (unknown keys are kept, missing ones defaulted)."""
    out = copy.deepcopy(DEFAULT_PARAMS)
    for key, val in (params or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key].update(val)
        else:
            out[key] = val
    return out


def effective_floors(f: dict, p: dict) -> tuple[np.ndarray, np.ndarray]:
    fa, fc = f["floor_a"], f["floor_c"]
    if p.get("floor_ref_a") is not None:
        fa = np.maximum(fa, p["floor_ref_a"])
    if p.get("floor_ref_c") is not None:
        fc = np.maximum(fc, p["floor_ref_c"])
    return fa, fc


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of the True runs in a boolean array."""
    if not len(mask):
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _event(cls, f, i0, i1, fire, score, **info) -> dict:
    t = f["t"]
    ev = {"cls": cls, "t0": float(t[i0] - DT), "t1": float(t[min(i1, len(t)) - 1]),
          "t": float(t[min(fire, len(t) - 1)]), "score": round(float(score), 2),
          "level": round(float(f["lc_max"][i0:i1].max()), 2), "laf": round(float(f["laf"][i0:i1].max()), 2),
          "clipped": bool(f["clip"][i0:i1].sum() > 0)}
    ev["len_s"] = round(ev["t1"] - ev["t0"], 3)
    ev["info"] = {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in info.items()}
    return ev


# ---------------------------------------------------------------------------- bang
def bang_candidates(f: dict, p: dict, complete_only: bool = True, skip_before: int = 0) -> list[dict]:
    """Every sudden onset that passes the bang gates, with its score (independent of the threshold)."""
    q = p["bang"]
    lc = f["lc_max"]
    _, fc = effective_floors(f, p)
    x = lc - fc
    n = len(lc)
    look = int(round(2.0 / DT))
    out, i_next = [], 0
    for i in np.flatnonzero(f["onset_c"] >= q["onset_db"]):
        if i < max(i_next, skip_before):
            continue
        stop = min(n, i + look)
        seg = lc[i:stop]
        decayed = (seg < np.maximum.accumulate(seg) - q["decay_db"]) | (x[i:stop] < q["end_db"])
        decayed[0] = False
        ends = np.flatnonzero(decayed)
        if len(ends):
            j = i + int(ends[0])
        elif stop == n and complete_only:
            break                      # still ringing at the end of the data: decide later
        else:
            j = stop
        i_next = j
        length = (j - i) * DT
        if length > q["max_len_s"] + 1e-9:
            continue
        tone_frac = float((f["tone_prom"][i:j] >= q["tone_db"]).mean())
        if tone_frac > q["tone_frac_max"]:
            continue
        score = float(x[i:j].max())
        out.append(_event("bang", f, i, j, j, score, onset_db=float(f["onset_c"][i]), tone_frac=tone_frac,
                          floor_c=float(fc[i])))
    return out


# ---------------------------------------------------------------------------- shout
def shout_events(f: dict, p: dict, threshold: float, complete_only: bool = True, skip_before: int = 0) -> list[dict]:
    q = p["shout"]
    fa, _ = effective_floors(f, p)
    y = f["laf"] - fa
    n = len(y)
    H = max(1, int(round(q["hold_s"] / DT)))
    if n < H:
        return []
    held = np.full(n, -np.inf)
    held[H - 1:] = sliding_window_view(y, H).min(axis=1)   # level held for the last hold_s
    out = []
    for s, e in runs(held >= threshold):
        i0 = s - (H - 1)
        if i0 <= skip_before or (e == n and complete_only):
            continue                    # cut off by the start of the data, or still going
        length = (e - i0) * DT
        per = float(f["periodicity"][i0:e].mean())
        if not (q["min_len_s"] - 1e-9 <= length <= q["max_len_s"] + 1e-9) or per < q["periodicity"]:
            continue
        out.append(_event("shout", f, i0, e, e, float(held[s:e].max()), periodicity=per,
                          voice_ratio=float(f["voice_ratio"][i0:e].mean()), floor_a=float(fa[i0])))
    return out


# ---------------------------------------------------------------------------- knock
def knock_events(f: dict, p: dict, threshold: float, complete_only: bool = True, skip_before: int = 0) -> list[dict]:
    q = p["knock"]
    _, fc = effective_floors(f, p)
    x = f["lc_max"] - fc
    n = len(x)
    rise_n, fall_n = max(1, int(round(q["rise_s"] / DT))), max(1, int(round(q["fall_s"] / DT)))
    peaks, i = [], max(rise_n, skip_before)
    while i < n - 1:
        if (x[i] >= threshold and x[i] >= x[i - 1] and x[i] >= x[i + 1]
                and x[i] - x[i - rise_n:i].min() >= q["rise_db"]):
            if (x[i + 1:min(n, i + fall_n + 1)] <= x[i] - q["fall_db"]).any():
                peaks.append(i)
                i += int(round(q["min_gap_s"] / DT))
                continue
        i += 1
    out, run = [], []
    max_gap = int(round(q["max_gap_s"] / DT))

    def close(run):
        if len(run) >= q["min_knocks"]:
            a, b = run[0], run[-1] + 1
            tone = max(float(np.median(f["tone_prom"][a:b])), float(np.median(f["siren_prom"][a:b])))
            if tone > q["tone_max_db"]:
                return                  # a beep, or a voice: not a knock
            out.append(_event("knock", f, run[0], run[-1] + 1, run[-1], float(min(x[k] for k in run)),
                              knocks=len(run), tone_db=tone, gaps_s=[round((y - x_) * DT, 2) for x_, y in zip(run, run[1:])]))

    for k in peaks:
        if run and k - run[-1] <= max_gap:
            run.append(k)
        else:
            close(run)
            run = [k]
    if run and (not complete_only or n - 1 - run[-1] > max_gap):
        close(run)                      # live: only once no further knock can join it
    return out


# ---------------------------------------------------------------------------- alarm
def alarm_mask(f: dict, p: dict, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: is the alarm condition met, and the duty (fraction of the window tonal at one pitch)."""
    q = p["alarm"]
    fa, _ = effective_floors(f, p)
    n = len(f["t"])
    W = max(1, int(round(q["window_s"] / DT)))
    tonal = ((f["tone_prom"] >= threshold) & (f["la"] - fa >= q["above_floor_db"])
             & (f["periodicity"] <= q["periodicity_max"]))
    if not tonal.any():
        return np.zeros(n, bool), np.zeros(n)
    bins = np.floor((f["tone_hz"] - 2000.0) / q["hz_tol"]).astype(int)
    nb = int(math.ceil(3000.0 / q["hz_tol"])) + 2
    bins = np.clip(bins, 0, nb - 1)
    onehot = np.zeros((n, nb + 2), np.int32)
    idx = np.flatnonzero(tonal)
    for d in (0, 1, 2):          # smear by +-1 bin: one alarm = one pitch within +-hz_tol (and a bit)
        onehot[idx, bins[idx] + d] = 1
    cs = np.concatenate([np.zeros((1, nb + 2), np.int32), np.cumsum(onehot, axis=0)])
    lo = np.maximum(0, np.arange(1, n + 1) - W)
    counts = cs[1:] - cs[lo]
    duty = counts.max(axis=1) / W
    return duty >= q["duty"], duty


def siren_mask(f: dict, p: dict, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: is the siren condition met, and the tonal fraction of the window behind it."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d

    q = p["siren"]
    fa, _ = effective_floors(f, p)
    n = len(f["t"])
    W = max(1, int(round(q["window_s"] / DT)))
    hz = f["siren_hz"]
    tonal = ((f["siren_prom"] >= threshold) & (f["la"] - fa >= q["above_floor_db"])
             & (f["periodicity"] <= q["periodicity_max"]) & (hz >= q["min_hz"]))
    if not tonal.any():
        return np.zeros(n, bool), np.zeros(n)
    # Smooth: tonal, after two tonal frames, and the pitch's second difference is small.
    d2 = np.full(n, np.inf)
    d2[2:] = np.abs(hz[2:] - 2 * hz[1:-1] + hz[:-2])
    prev2 = np.zeros(n, bool)
    prev2[2:] = tonal[1:-1] & tonal[:-2]
    tonal = tonal & prev2 & (d2 <= q["smooth_hz"])
    cs = np.concatenate([[0], np.cumsum(tonal)])
    frac = (cs[1:] - cs[np.maximum(0, np.arange(1, n + 1) - W)]) / W
    # Pitch range over the trailing window, tonal frames only (trailing: origin shifts the filter back).
    hi = maximum_filter1d(np.where(tonal, hz, -np.inf), W, origin=(W - 1) // 2)
    lo = minimum_filter1d(np.where(tonal, hz, np.inf), W, origin=(W - 1) // 2)
    sweep = np.where(np.isfinite(hi) & np.isfinite(lo), hi - lo, 0.0)
    return (frac >= q["tonal_frac"]) & (sweep >= q["min_sweep_hz"]), frac


MASKS = {"alarm": alarm_mask, "siren": siren_mask}


def _tonal_info(cls: str, f: dict, s: int, e: int, threshold: float, strength: np.ndarray) -> dict:
    prom, hz = (f["tone_prom"], f["tone_hz"]) if cls == "alarm" else (f["siren_prom"], f["siren_hz"])
    on = hz[s:e][prom[s:e] >= threshold]
    info = {"tone_hz": float(np.median(on)) if len(on) else 0.0}
    info["duty" if cls == "alarm" else "tonal_frac"] = float(strength[s:e].max())
    if cls == "siren" and len(on):
        info["sweep_hz"] = float(np.ptp(on))
    return info


def alarm_events(f: dict, p: dict, threshold: float, skip_before: int = 0, cls: str = "alarm") -> list[dict]:
    q = p[cls]
    cond, duty = MASKS[cls](f, p, threshold)
    rearm = int(round(q["rearm_s"] / DT))
    merged: list[list[int]] = []
    for s, e in runs(cond):
        if merged and s - merged[-1][1] < rearm:
            merged[-1][1] = e           # the same alarm, between beeps
        else:
            merged.append([s, e])
    out = []
    prom = f["tone_prom"] if cls == "alarm" else f["siren_prom"]
    for s, e in merged:
        if s < skip_before:
            continue
        out.append(_event(cls, f, s, e, s, float(prom[s:e].max()), **_tonal_info(cls, f, s, e, threshold, duty)))
    return out


# ---------------------------------------------------------------------------- all classes
def apply_refractory(events: list[dict], refractory_s: float) -> list[dict]:
    kept, last = [], {}
    for ev in sorted(events, key=lambda e: e["t0"]):
        if ev["cls"] not in TONAL and ev["t0"] < last.get(ev["cls"], -math.inf) + refractory_s:
            continue
        kept.append(ev)
        last[ev["cls"]] = ev["t0"]
    return kept


def find_events(f: dict, params: dict | None = None, classes=CLASSES, complete_only: bool = True,
                skip_before: int = 0, thresholds: dict | None = None) -> list[dict]:
    """All events in the frames, in time order. skip_before: frames before this index are only
    context (a trial's pre-roll, or the edge of a live window) and can't start an event."""
    p = merged_params(params)
    th = {c: p[c]["threshold"] for c in CLASSES}
    th.update(thresholds or {})
    events: list[dict] = []
    if "bang" in classes:
        events += [e for e in bang_candidates(f, p, complete_only, skip_before) if e["score"] >= th["bang"]]
    if "shout" in classes:
        events += shout_events(f, p, th["shout"], complete_only, skip_before)
    if "knock" in classes:
        events += knock_events(f, p, th["knock"], complete_only, skip_before)
    for cls in TONAL:
        if cls in classes:
            events += alarm_events(f, p, th[cls], skip_before, cls)
    sirens = [(e["t0"], e["t1"]) for e in events if e["cls"] == "siren"]
    events = [e for e in events if not (e["cls"] == "alarm" and any(a <= e["t1"] and e["t0"] <= b for a, b in sirens))]
    return apply_refractory(events, p["refractory_s"])


class LiveDetector:
    """Runs find_events() on a sliding window of recent frames and reports each event once."""

    WINDOW_S = 8.0

    def __init__(self, params: dict | None = None, classes=CLASSES):
        self.p = merged_params(params)
        self.classes = tuple(classes)   # with YAMNet (classify.py), only bang and shout come from here
        self.last_t0 = {c: -math.inf for c in CLASSES}
        self.tonal_seen_t = {c: -math.inf for c in TONAL}
        self.active = {c: False for c in TONAL}  # is a siren / alarm going on right now

    def set_params(self, params: dict | None) -> None:
        """New thresholds from now on; events already reported stay reported."""
        self.p = merged_params(params)

    def update(self, window: dict) -> list[dict]:
        if len(window["t"]) < 40:
            return []
        p = self.p
        # The first 300 ms of the window is onset context; alarms need a full duty window behind them.
        events = find_events(window, p, classes=tuple(c for c in ("bang", "shout", "knock") if c in self.classes), skip_before=32)
        new = []
        for ev in events:
            if ev["t0"] >= self.last_t0[ev["cls"]] + p["refractory_s"] - 1e-6:
                new.append(ev)
                self.last_t0[ev["cls"]] = ev["t0"]
        t = window["t"]
        for cls in (c for c in TONAL if c in self.classes):
            th = p[cls]["threshold"]
            cond, strength = MASKS[cls](window, p, th)
            W = int(round(p[cls]["window_s"] / DT))
            prom = window["tone_prom"] if cls == "alarm" else window["siren_prom"]
            self.active[cls] = bool(cond[-20:].any()) if len(cond) else False  # within the last 0.2 s
            for s, e in runs(cond):
                # A run in the first window_s has a partial window behind it: it can only be the
                # tail of one already seen, so it just extends tonal_seen_t.
                t_start, t_end = float(t[s]), float(t[e - 1])
                siren_on = cls == "alarm" and self.tonal_seen_t["siren"] >= t_start - 1.0
                if t_start > self.tonal_seen_t[cls] + p[cls]["rearm_s"] and s >= W and not siren_on:
                    new.append(_event(cls, window, s, e, s, float(prom[s:e].max()),
                                      **_tonal_info(cls, window, s, e, th, strength)))
                self.tonal_seen_t[cls] = max(self.tonal_seen_t[cls], t_end)
        return sorted(new, key=lambda e: e["t0"])


# ---------------------------------------------------------------------------- trials
# A trial is {"id", "label", "kind": "positive"|"negative", "cls": class or None, "expected": int,
# "pre_frames": int} plus its frames (from meter.analyze on the saved WAV, with the saved floor as seed).

def count_trial(trial: dict, frames: dict, params: dict, thresholds: dict | None = None, classes=CLASSES) -> dict:
    events = find_events(frames, params, classes=classes, complete_only=False,
                         skip_before=trial.get("pre_frames", 0), thresholds=thresholds)
    counts = {c: 0 for c in CLASSES}
    for ev in events:
        counts[ev["cls"]] += 1
    return {"counts": counts, "events": events}


def judge(trial: dict, counts: dict) -> dict:
    """Hits, misses and false alarms for one trial, from its event counts."""
    total = sum(counts.values())
    if trial["kind"] == "positive":
        cls, exp = trial["cls"], int(trial.get("expected", 1))
        got = counts.get(cls, 0)
        return {"hits": min(got, exp), "misses": max(0, exp - got), "extra": max(0, got - exp),
                "other": total - got, "false_alarms": max(0, got - exp)}
    return {"hits": 0, "misses": 0, "extra": 0, "other": 0, "false_alarms": total}


def score_trials(trials: list[dict], frames: dict[str, dict], params: dict) -> dict:
    """Run the detectors over every trial; per-trial results plus a summary per class."""
    p = merged_params(params)
    rows, summary = [], {c: {"expected": 0, "hits": 0, "misses": 0, "false_alarms": 0} for c in CLASSES}
    neg_s = 0.0
    for tr in trials:
        res = count_trial(tr, frames[tr["id"]], p)
        j = judge(tr, res["counts"])
        rows.append({"id": tr["id"], "label": tr["label"], "kind": tr["kind"], "cls": tr.get("cls"),
                     "expected": tr.get("expected", 0), "counts": res["counts"], **j, "events": res["events"]})
        if tr["kind"] == "positive" and tr.get("cls") in summary:
            s = summary[tr["cls"]]
            s["expected"] += int(tr.get("expected", 1))
            s["hits"] += j["hits"]
            s["misses"] += j["misses"]
            s["false_alarms"] += j["extra"]
        elif tr["kind"] == "negative":
            neg_s += max(0.0, (len(frames[tr["id"]]["t"]) - tr.get("pre_frames", 0)) * DT)
            for ev in res["events"]:
                summary[ev["cls"]]["false_alarms"] += 1
    for s in summary.values():
        s["hit_rate"] = round(s["hits"] / s["expected"], 3) if s["expected"] else None
    return {"trials": rows, "summary": summary, "negative_minutes": round(neg_s / 60.0, 2)}


def _max_threshold(ok, lo: float, hi: float) -> float | None:
    """Highest threshold in [lo, hi] where ok(threshold) holds, on a 1 dB grid refined to 0.25 dB.
    None if it doesn't hold anywhere."""
    best = None
    for T in np.arange(hi, lo - 1e-9, -1.0):
        if ok(float(T)):
            best = float(T)
            break
    if best is None:
        return None
    for T in np.arange(min(hi, best + 0.75), best, -0.25):
        if ok(float(T)):
            return float(T)
    return best


def fit(trials: list[dict], frames: dict[str, dict], params: dict, classes=CLASSES) -> dict:
    """Fit each class's threshold from the trials.

    For each positive trial of the class: the highest threshold at which all its expected sounds
    are still detected. For each negative trial: the highest threshold at which it still raises
    this class (a false alarm). The threshold goes in the gap between the two groups, 60% of the
    way up from the worst negative (leaning toward fewer false alarms). If they overlap, it goes
    just above the worst negative, and the positives that would be missed are listed.
    """
    p = merged_params(params)
    report = {}
    lo_T, hi_T = FIT_RANGE
    for cls in classes:
        pos, neg = [], []
        for tr in trials:
            fr = frames[tr["id"]]
            if tr["kind"] == "positive" and tr.get("cls") == cls:
                exp = int(tr.get("expected", 1))
                T = _max_threshold(lambda th: count_trial(tr, fr, p, {cls: th}, (cls,))["counts"][cls] >= exp, lo_T, hi_T)
                pos.append({"id": tr["id"], "label": tr["label"], "max_threshold": T})
            elif tr["kind"] == "negative":
                T = _max_threshold(lambda th: count_trial(tr, fr, p, {cls: th}, (cls,))["counts"][cls] >= 1, lo_T, hi_T)
                neg.append({"id": tr["id"], "label": tr["label"], "max_threshold": T})
        detectable = [x["max_threshold"] for x in pos if x["max_threshold"] is not None]
        alarming = [x["max_threshold"] for x in neg if x["max_threshold"] is not None]
        lo = max(alarming) if alarming else None     # thresholds at or below this raise a false alarm
        hi = min(detectable) if detectable else None  # thresholds above this miss a positive
        undetectable = [x["label"] for x in pos if x["max_threshold"] is None]
        notes = []
        if undetectable:
            notes.append(f"the gates reject {undetectable} at every threshold; check the gate values in their events")
        if not pos:
            T, notes = p[cls]["threshold"], notes + ["no positive trials: threshold left as it was"]
        elif hi is None:
            T = p[cls]["threshold"]
            notes.append("no positive trial is detected at any threshold: threshold left as it was")
        elif not neg:
            T = max(lo_T, hi - 3.0)
            notes.append("no negative trials yet: threshold set 3 dB under the weakest positive. Record negatives!")
        elif lo is None:
            T = max(lo_T, hi - 3.0)
            notes.append("no negative trial raises this class at any threshold: threshold 3 dB under the weakest positive")
        elif hi > lo:
            T = lo + 0.6 * (hi - lo)
            if hi - lo < 3.0:
                notes.append(f"thin margin ({hi - lo:.1f} dB): small changes in the room will cause misses or false alarms")
        else:
            T = lo + 0.25
            missed = [x["label"] for x in pos if x["max_threshold"] is None or x["max_threshold"] < T]
            notes.append(f"OVERLAP: negatives raise {cls} up to {lo:.2f} dB but positives are only all detected "
                         f"up to {hi:.2f} dB. Threshold set just above the worst negative; missed: {missed}")
        report[cls] = {"threshold": round(float(T), 2), "worst_negative": lo, "weakest_positive": hi,
                       "margin_db": round(hi - lo, 2) if (hi is not None and lo is not None) else None,
                       "positives": pos, "negatives": neg, "notes": notes}
    return report
