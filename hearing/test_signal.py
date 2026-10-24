#!/usr/bin/env python3
"""Checks for the hearing signal chain, on synthetic sounds at known levels (no mic needed).

    python3 hearing/test_signal.py          # everything
    python3 hearing/test_signal.py meter    # one group: meter, detect, live, fit, store, speed

Needs numpy and scipy (already on the Pupper).
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect as D  # noqa: E402
import meter as M  # noqa: E402
import store  # noqa: E402
import synth as Y  # noqa: E402

PASS = FAIL = 0
FS = M.FS


def check(name: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f": {detail}" if detail else ""))
    return ok


def sine(hz, level_db_peak, seconds, dither_db=-100.0):
    t = np.arange(int(seconds * FS)) / FS
    x = 10 ** (level_db_peak / 20) * np.sin(2 * np.pi * hz * t)
    return x + np.random.default_rng(7).standard_normal(len(t)) * 10 ** (dither_db / 20)


def iec(kind, f):
    """Exact IEC 61672 weighting (dB), normalized to 0 at 1 kHz."""
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    def r(f):
        if kind == "A":
            return f4**2 * f**4 / ((f**2 + f1**2) * np.sqrt((f**2 + f2**2) * (f**2 + f3**2)) * (f**2 + f4**2))
        return f4**2 * f**2 / ((f**2 + f1**2) * (f**2 + f4**2))
    return 20 * np.log10(r(f) / r(1000.0))


def kinds(events):
    return [e["cls"] for e in events]


def scene(*items, seconds=12.0, floor_db=-60.0, seed=0):
    """Room noise with sounds placed at given times: items are (at_s, samples)."""
    bed = Y.pink(seconds, floor_db, seed=seed)
    for at, snd in items:
        Y.place(bed, snd, at)
    return M.analyze(Y.to_int16(bed))


# ----------------------------------------------------------------------------- meter
def test_meter():
    print("meter: levels, weightings, floor (a sound level meter has to read right first)")
    freqs = [31.5, 50, 63, 100, 125, 200, 250, 500, 1000, 2000, 4000, 5000]
    for kind in ("A", "C"):
        _, h = signal.sosfreqz(M.weighting_sos(kind), worN=freqs + [8000, 10000], fs=FS)
        err = 20 * np.log10(np.abs(h)) - iec(kind, np.array(freqs + [8000, 10000], float))
        check(f"{kind}-weighting within 0.1 dB of IEC 61672 from 31.5 Hz to 5 kHz",
              np.abs(err[:-2]).max() < 0.1, f"worst {np.abs(err[:-2]).max():.2f} dB")
        check(f"{kind}-weighting within IEC Class 1 tolerance at 8 and 10 kHz",
              -2.5 <= err[-2] <= 1.5 and -3.0 <= err[-1] <= 2.0, f"8k {err[-2]:+.2f}, 10k {err[-1]:+.2f}")

    f = M.analyze(Y.to_int16(sine(1000, 20 * np.log10(32767 / 32768), 2.0, dither_db=-200)))
    check("full-scale 1 kHz sine reads -3.01 dBFS (A, C) and 0 dBFS peak",
          abs(np.median(f["la"][50:]) + 3.01) < 0.05 and abs(np.median(f["lc"][50:]) + 3.01) < 0.05
          and abs(f["peak"][50:].max()) < 0.01, f"la {np.median(f['la'][50:]):.2f} peak {f['peak'].max():.3f}")
    f = M.analyze(Y.to_int16(sine(1000, -40, 2.0)))
    vals = {k: np.median(f[k][50:]) for k in ("la", "lc", "lc_max", "laf")}
    check("1 kHz sine at -40 dBFS peak reads -43.01 on la, lc, lc_max and laf (+-0.1)",
          all(abs(v + 43.01) < 0.1 for v in vals.values()), str({k: round(v, 2) for k, v in vals.items()}))
    f = M.analyze(Y.to_int16(sine(100, -20, 2.0)))
    check("at 100 Hz, A reads 18.85 dB below C (why bangs and thuds use C)",
          abs((np.median(f["lc"][50:]) - np.median(f["la"][50:])) - 18.8) < 0.3,
          f"C - A = {np.median(f['lc'][50:]) - np.median(f['la'][50:]):.2f} (IEC: 19.15 - 0.30 = 18.85)")

    f = M.analyze(Y.to_int16(sine(1000, 20 * np.log10(1.5), 1.0)))
    g = M.analyze(Y.to_int16(sine(1000, -6, 1.0)))
    check("clipping is counted (an overdriven sine) and a -6 dBFS sine has none",
          f["clip"].sum() > 1000 and g["clip"].sum() == 0, f"overdriven {f['clip'].sum()}, -6 dB {g['clip'].sum()}")

    x = np.concatenate([np.zeros(FS), sine(1000, -20, 2.0)])
    f = M.analyze(Y.to_int16(x + np.random.default_rng(1).standard_normal(len(x)) * 1e-6))
    L = -23.01
    at = lambda s: f["laf"][int(round(s / 0.01)) - 1]  # noqa: E731
    check("LAF follows the 125 ms Fast time constant: 2.0 dB under at 125 ms, settled at 1 s",
          abs(at(1.125) - (L - 2.0)) < 0.3 and abs(at(2.0) - L) < 0.1, f"at 125 ms {at(1.125):.2f}, at 1 s {at(2.0):.2f}")

    # A 5 ms click placed at different offsets inside a hop: lc_max shouldn't care where it lands.
    reads, reads_hop = [], []
    for off in range(0, 480, 60):
        x = np.zeros(FS)
        x[FS // 2 + off: FS // 2 + off + 240] = 0.1 * np.random.default_rng(3).standard_normal(240)
        f = M.analyze(Y.to_int16(x))
        reads.append(f["lc_max"].max())
        reads_hop.append(f["lc"].max())
    check("a short click reads the same lc_max wherever it falls in a 10 ms hop (+-0.5 dB)",
          np.ptp(reads) < 0.5, f"lc_max spread {np.ptp(reads):.2f} dB (plain hop level spreads {np.ptp(reads_hop):.2f} dB)")

    quiet = M.analyze(Y.to_int16(Y.pink(20, -60)))
    noisy_x = Y.pink(20, -60)
    for at_s in (4, 8, 12, 16):
        Y.place(noisy_x, Y.clap(-20, seed=at_s), at_s)
    noisy = M.analyze(Y.to_int16(noisy_x))
    check("noise floor tracks steady room noise (within 1.5 dB of its median level)",
          abs(quiet["floor_a"][-1] - np.median(quiet["la"])) < 1.5, f"floor {quiet['floor_a'][-1]:.2f}, median {np.median(quiet['la']):.2f}")
    check("a few loud claps move the floor by less than 1 dB",
          abs(noisy["floor_a"][-1] - quiet["floor_a"][-1]) < 1.0 and abs(noisy["floor_c"][-1] - quiet["floor_c"][-1]) < 1.0,
          f"A {noisy['floor_a'][-1] - quiet['floor_a'][-1]:+.2f}, C {noisy['floor_c'][-1] - quiet['floor_c'][-1]:+.2f}")

    v = M.analyze(Y.to_int16(Y.vowel(2.0, -30) + Y.pink(2.0, -70)))
    n = M.analyze(Y.to_int16(Y.pink(2.0, -30)))
    t = M.analyze(Y.to_int16(Y.tone_pattern([(2.0, 0.0)], 3100, -30, 1) + Y.pink(2.0, -70)))
    pv, pn, pt = (np.median(q["periodicity"][20:-20]) for q in (v, n, t))
    check("periodicity: a voice is high, pink noise and a pure alarm tone are low",
          pv > 0.7 and pn < 0.35 and pt < 0.45, f"voice {pv:.2f}, noise {pn:.2f}, tone {pt:.2f}")
    check("tone prominence finds a 3.1 kHz tone at the right frequency (+-10 Hz)",
          np.median(t["tone_prom"][20:-20]) > 40 and abs(np.median(t["tone_hz"][20:-20]) - 3100) < 10,
          f"prom {np.median(t['tone_prom'][20:-20]):.1f} dB at {np.median(t['tone_hz'][20:-20]):.1f} Hz")


# ----------------------------------------------------------------------------- detectors
def test_detect():
    print("detect: each class fires on its sound, and not on the others")
    f = scene((5.0, Y.clap(-30)))
    ev = D.find_events(f, complete_only=False)
    check("a clap 30 dB over the room gives exactly one bang", kinds(ev) == ["bang"], str(kinds(ev)))
    check("the bang is timed right (+-20 ms) and short", ev and abs(ev[0]["t0"] - 5.0) < 0.02 and ev[0]["len_s"] < 0.3,
          str(ev[:1]))
    ramp = Y.pink(2.0, -30, seed=9) * np.linspace(0, 1, 2 * FS)
    f = scene((4.0, ramp))
    ev = D.find_events(f, complete_only=False)
    check("noise faded in over 2 s up to the clap's level gives no bang (not sudden)", "bang" not in kinds(ev), str(kinds(ev)))
    f = scene((5.0, Y.thud(-28)))
    ev = D.find_events(f, complete_only=False)
    check("a low thud (a fall) 32 dB over the room gives one bang", kinds(ev) == ["bang"], str(kinds(ev)))

    f = scene((4.0, Y.shout(-30)))
    ev = D.find_events(f, complete_only=False)
    check("a 0.8 s shout 30 dB over the room gives one shout", kinds(ev) == ["shout"], str(kinds(ev)))
    f = scene((2.0, Y.talk(8.0, -52)))
    ev = D.find_events(f, complete_only=False)
    check("8 s of talking 8 dB over the room gives nothing", ev == [], str(kinds(ev)))
    f = scene((2.0, Y.talk(8.0, -45)))
    ev = D.find_events(f, complete_only=False)
    check("8 s of louder talking (15 dB over the room) gives nothing", ev == [], str([(e["cls"], e["score"]) for e in ev]))

    f = scene((2.0, Y.smoke_alarm(-30, repeats=2)), seconds=12)
    ev = D.find_events(f, complete_only=False)
    check("a smoke alarm (T3 pattern, 3.1 kHz) gives one alarm", kinds(ev) == ["alarm"], str(kinds(ev)))
    check("...within 3 s of its first beep", ev and ev[0]["t"] - 2.0 <= 3.0, str([round(e["t"], 2) for e in ev]))
    f = scene((2.0, Y.timer_beeps(-32, seconds=5)), seconds=10)
    ev = D.find_events(f, complete_only=False)
    check("a kitchen timer (2.7 kHz, 4 beeps/s) gives one alarm", kinds(ev) == ["alarm"], str(kinds(ev)))
    f = scene((2.0, Y.pink(4.0, -40, seed=11)), seconds=10)
    ev = D.find_events(f, complete_only=False)
    check("4 s of broadband noise 20 dB over the room gives nothing", ev == [], str(kinds(ev)))
    f = scene((2.0, Y.vowel(4.0, -35, f0=(400.0, 400.0))), seconds=10)
    ev = D.find_events(f, complete_only=False)
    check("a long sung note (steady voice, 4 s) is not an alarm", "alarm" not in kinds(ev), str(kinds(ev)))

    for kind in ("wail", "yelp", "hilo"):
        f = scene((2.0, Y.siren(-35, kind=kind)), seconds=12)
        ev = D.find_events(f, complete_only=False)
        check(f"a {kind} siren 25 dB over the room gives one siren (and nothing else), within 3 s",
              kinds(ev) == ["siren"] and ev[0]["t"] - 2.0 <= 3.0, str([(e["cls"], round(e["t"], 1)) for e in ev]))
    f = scene((2.0, Y.vowel(6.0, -30, f0=(220.0, 330.0, 260.0, 300.0))), seconds=12)
    ev = D.find_events(f, complete_only=False)
    check("a voice gliding up and down (6 s) is not a siren", "siren" not in kinds(ev), str(kinds(ev)))
    f = scene((2.0, Y.smoke_alarm(-30, repeats=2)), seconds=12)
    check("a smoke alarm (one steady pitch) is not a siren", "siren" not in kinds(D.find_events(f, complete_only=False)))

    f = scene(*[(2.0 + 1.5 * i, Y.clap(-30, seed=i)) for i in range(4)])
    ev = D.find_events(f, complete_only=False)
    check("4 claps 1.5 s apart give 4 bangs", kinds(ev) == ["bang"] * 4, str(kinds(ev)))
    f = scene(*[(3.0 + 0.1 * i, Y.clap(-30, seed=i)) for i in range(10)])
    ev = D.find_events(f, classes=("bang",), complete_only=False)
    check("10 claps in 1 s give one bang (1 s refractory period)", kinds(ev) == ["bang"], str(kinds(ev)))

    f = scene(*[(3.0 + 0.3 * i, Y.clap(-38, seed=i)) for i in range(3)])
    ev = D.find_events(f, complete_only=False)
    check("three taps 0.3 s apart, 22 dB over the room, are knocking", kinds(ev) == ["knock"], str(kinds(ev)))
    f = scene((3.0, Y.clap(-38)))
    check("a single tap is not knocking", "knock" not in kinds(D.find_events(f, complete_only=False)))

    # Soft knocks (under the threshold) are offered to YAMNet only by a live detector that asks for them
    f = scene(*[(3.0 + 0.3 * i, Y.clap(-44, seed=i)) for i in range(3)])
    check("three soft taps (16 dB over the room) are not knocking on their own",
          "knock" not in kinds(D.find_events(f, complete_only=False)))
    soft = [e for e in D.LiveDetector(soft_knocks=True).update(f) if e["cls"] == "knock"]
    check("...but a live detector with soft_knocks offers them, marked soft",
          len(soft) == 1 and soft[0]["info"].get("soft") is True, str(soft))
    check("...and one without soft_knocks doesn't",
          not [e for e in D.LiveDetector().update(f) if e["cls"] == "knock"])
    test_knock_gate()

    f = scene((5.0, Y.clap(-30)))
    hi = D.find_events(f, {"floor_ref_c": -40.0}, complete_only=False)
    check("with a calibrated floor (floor_ref) louder than this room, the same clap is too quiet",
          hi == [], str(kinds(hi)))


def test_knock_gate():
    """KnockGate's rules, with a stand-in for YAMNet's scores (no model needed)."""
    import math
    import classify as C

    class FakeYamnet:
        def __init__(self, speech, knocklike):
            self.t, self.last_knock_t = 5.0, -math.inf
            self.speech, self.knocklike = [(3.5, speech)], [(3.5, knocklike)]

    def passes(speech, knocklike, soft=False):
        g = C.KnockGate(FakeYamnet(speech, knocklike))
        g.add({"cls": "knock", "t0": 3.0, "t1": 3.6, "score": 15.0 if soft else 22.0, "info": {"soft": soft}})
        return len(g.ready()) == 1

    check("knock gate: a knock with some talking (speech 0.4) is kept", passes(0.4, 0.0))
    check("knock gate: a knock YAMNet is sure was speech (0.8), with nothing knock-like, is dropped", not passes(0.8, 0.0))
    check("knock gate: a knock over loud talking (0.8) is kept when YAMNet also heard a knock (0.2)", passes(0.8, 0.2))
    check("knock gate: a soft knock is kept only if YAMNet heard something knock-like",
          passes(0.0, 0.2, soft=True) and not passes(0.0, 0.0, soft=True))
    g = C.KnockGate(FakeYamnet(0.0, 0.3))
    g.c.last_knock_t = 4.0
    g.add({"cls": "knock", "t0": 3.0, "t1": 3.6, "score": 22.0, "info": {}})
    check("knock gate: no second alert when YAMNet already reported this knocking", g.ready() == [])


# ----------------------------------------------------------------------------- live == batch
def test_live():
    print("live: the streaming detector reports what the batch detector finds, once each")
    x, _ = Y.demo_scene()
    samples = Y.to_int16(x)
    batch = D.find_events(M.analyze(samples))
    m, live, hist = M.Meter(), D.LiveDetector(), []
    found = []
    block = 2400  # 50 ms, like the mic
    for i in range(0, len(samples), block):
        hist.append(m.feed(samples[i:i + block]))
        if (i // block) % 2 == 1:
            win = M.concat_frames(hist)
            keep = int(D.LiveDetector.WINDOW_S / D.DT)
            win = M.slice_frames(win, max(0, len(win["t"]) - keep))
            hist = [win]
            found += live.update(win)
    key = lambda evs: [(e["cls"], round(e["t0"], 2)) for e in evs]  # noqa: E731
    check("live events match batch events (class and start time)", key(found) == key(batch),
          f"live {key(found)}\n         batch {key(batch)}")
    check("the demo scene finds 2 claps + thud, the shout, the alarm, the timer and the siren, and nothing in the talking",
          sorted(kinds(batch)) == sorted(["bang", "bang", "bang", "shout", "alarm", "alarm", "siren"]), str(key(batch)))


# ----------------------------------------------------------------------------- fit
def test_fit():
    print("fit: thresholds come from the trials")
    trials, frames = [], {}

    def add(tid, kind, cls, expected, f, label):
        trials.append({"id": tid, "label": label, "kind": kind, "cls": cls, "expected": expected, "pre_frames": 100})
        frames[tid] = f

    add("p1", "positive", "bang", 3, scene(*[(2.0 + 1.5 * i, Y.clap(-35, seed=i)) for i in range(3)], seconds=8), "claps far")
    add("p2", "positive", "bang", 2, scene((2.0, Y.clap(-25)), (4.0, Y.clap(-28, seed=4)), seconds=7), "claps near")
    add("n1", "negative", None, 0, scene((1.5, Y.talk(6.0, -48))), "talking")
    add("n2", "negative", None, 0, scene((2.0, Y.clap(-42)), (4.0, Y.clap(-41, seed=2)), seconds=7), "soft taps")
    rep = D.fit(trials, frames, {}, classes=("bang",))["bang"]
    lo, hi, T = rep["worst_negative"], rep["weakest_positive"], rep["threshold"]
    check("bang threshold lands in the gap between the soft taps and the far claps",
          lo is not None and hi is not None and lo < T < hi, str({k: rep[k] for k in ("worst_negative", "weakest_positive", "threshold")}))
    s = D.score_trials(trials, frames, {"bang": {"threshold": T}})
    check("at the fitted threshold: all 5 claps detected, no false alarms",
          s["summary"]["bang"]["hits"] == 5 and s["summary"]["bang"]["false_alarms"] == 0, str(s["summary"]["bang"]))

    add("n3", "negative", None, 0, scene((3.0, Y.clap(-28, seed=8)), seconds=6), "door (should not alert)")
    rep = D.fit(trials, frames, {}, classes=("bang",))["bang"]
    check("overlapping classes are reported, not hidden", any("OVERLAP" in n for n in rep["notes"]), str(rep["notes"]))
    s = D.score_trials(trials, frames, {"bang": {"threshold": rep["threshold"]}})
    check("...and the threshold keeps the negatives quiet (zero false alarms)", s["summary"]["bang"]["false_alarms"] == 0,
          str(s["summary"]["bang"]))


# ----------------------------------------------------------------------------- store
def test_store():
    print("store: a saved trial re-scores like it did live")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sess = store.Session("test", root)
        bed = Y.pink(9.0, -60)
        Y.place(bed, Y.clap(-30), 6.0)
        x = Y.to_int16(bed)
        live = M.analyze(x)
        pre = 300  # 3 s of pre-roll before the trial
        cut = pre * M.HOP
        tr = sess.add({"label": "clap", "kind": "positive", "cls": "bang", "expected": 1, "pre_frames": pre,
                       "floor_a": float(live["floor_a"][200]), "floor_c": float(live["floor_c"][200])}, x[200 * M.HOP:])
        sess2 = store.Session("test", root)
        f = sess2.frames(tr["id"])
        ev = D.count_trial(sess2.get(tr["id"]), f, {})["events"]
        check("WAV round trip is exact", np.array_equal(sess2.samples(tr["id"]), x[200 * M.HOP:]))
        check("the reloaded trial finds its clap, at the same level as live (+-0.5 dB)",
              kinds(ev) == ["bang"] and abs(ev[0]["level"] - live["lc_max"].max()) < 0.5, str(ev))
        check("trial duration excludes the pre-roll", abs(tr["duration_s"] - (len(x) / FS - 2.0 - cut / FS)) < 0.02,
              str(tr["duration_s"]))
        store.save_calibration({"params": {"bang": {"threshold": 22.5}}}, root)
        store.save_calibration({"params": {"bang": {"threshold": 23.0}}}, root)
        check("saving a calibration keeps the previous one", (root / "calibration.prev.json").exists()
              and store.load_calibration(root)["params"]["bang"]["threshold"] == 23.0)


# ----------------------------------------------------------------------------- speed
def test_speed():
    print("speed: fast enough for the Pi")
    x, _ = Y.demo_scene()
    samples = Y.to_int16(x)
    t0 = time.process_time()
    f = M.analyze(samples)
    D.find_events(f)
    dt = time.process_time() - t0
    check("60 s of audio is measured and scanned in under 3 s of CPU", dt < 3.0, f"{dt:.2f} s")
    print(f"         ({100 * dt / 60:.1f}% of one core here; run `python3 hearing/mic_lab.py --bench` on the Pupper)")


GROUPS = {"meter": test_meter, "detect": test_detect, "live": test_live, "fit": test_fit, "store": test_store,
          "speed": test_speed}

if __name__ == "__main__":
    for name in sys.argv[1:] or GROUPS:
        if name not in GROUPS:
            sys.exit(f"unknown group {name!r}; choose from {list(GROUPS)}")
        GROUPS[name]()
        print()
    if FAIL:
        print(f"{FAIL} check(s) failed, {PASS} passed.")
        raise SystemExit(1)
    print(f"All {PASS} checks passed.")
