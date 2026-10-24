"""The signal: levels, weightings, noise floor and spectrum, measured the way a sound level meter does.

Pure numpy/scipy with no I/O, so it runs the same on the Pupper, on a laptop and in the tests
(hearing/test_signal.py). Feed it int16 mono samples at 48 kHz in blocks of any size. It returns
one row of features for every 10 ms hop, as a dict of numpy columns (see COLUMNS).

Levels are in dBFS: a sample value of 32768 is 1.0, so a full-scale sine reads -3.01 dB RMS and
0 dB peak. Once you enter a reading from a phone SPL meter, dB SPL = dBFS + spl_offset_db.

Weightings follow IEC 61672:
  A  follows how loud a sound seems to people (shouts, alarms). It cuts 100 Hz by 19 dB.
  C  is nearly flat from 30 Hz to 8 kHz, so thuds and falls aren't hidden (bangs).
  Z  is unweighted, apart from a 20 Hz high-pass that removes DC (the spectrum).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.ndimage import uniform_filter1d

FS = 48000
HOP = 480                  # 10 ms: one feature row per hop
WIN = 2048                 # spectrum window (42.7 ms, Hann), zero-padded to NFFT
NFFT = 4096                # 11.7 Hz bins
TAU_FAST = 0.125           # sound level meter "Fast" time constant, in seconds
FLOOR_S = 30.0             # the noise floor looks back this far...
FLOOR_PCT = 10.0           # ...and takes this percentile of the 10 ms levels
FLOOR_EVERY = 50           # hops between floor updates (0.5 s)
ONSET_FROM, ONSET_TO = 30, 3   # onset compares now with the loudest hop 300 ms to 30 ms ago
SPEC_BINS = 96             # log-spaced bins for the spectrogram
SPEC_LO, SPEC_HI = 50.0, 20000.0
VOICE_BAND = (300.0, 3400.0)
SPEECH_SPAN = (100.0, 8000.0)  # voice_ratio and centroid only look at this span (no rumble)
TONE_BAND = (2000.0, 5000.0)   # where smoke alarms and timers beep
SIREN_BAND = (500.0, 4000.0)   # where sirens sweep (wail, yelp, hi-lo)
PITCH_BAND = (100.0, 4000.0)   # periodicity is measured on this band only
PITCH_LAGS = (48, 600)         # 1000 Hz down to 80 Hz, in samples
WHITEN_BINS = 41               # 480 Hz smoothing for the spectral envelope
EPS = 1e-20

# One value per 10 ms hop, in this order. dB values are dBFS unless noted.
COLUMNS = (
    "t",          # s, end of the hop
    "peak",       # raw sample peak (headroom = -peak)
    "clip",       # samples at full scale in this hop
    "dc",         # mean of the raw samples (1.0 = full scale)
    "la",         # A-weighted level over the hop (energy mean)
    "lc",         # C-weighted level over the hop (energy mean)
    "lc_max",     # loudest C-weighted 10 ms window ending inside this hop (doesn't depend on hop alignment)
    "laf",        # A-weighted Fast level (tau 125 ms), the loudest point in this hop
    "floor_a",    # noise floor of la: 10th percentile over the last 30 s
    "floor_c",    # same for lc
    "onset_c",    # dB: lc_max now minus the loudest lc_max 300..30 ms ago
    "voice_ratio",  # energy in 300-3400 Hz / energy in 100-8000 Hz
    "centroid",   # Hz, spectral centroid over 100-8000 Hz
    "periodicity",  # 0..1, best normalized autocorrelation at a pitch lag (voices, tones: high)
    "tone_prom",  # dB: strongest bin in 2-5 kHz over the band's median bin
    "tone_hz",    # Hz, frequency of that bin
    "siren_prom",  # dB: same, over 500-4000 Hz (sirens)
    "siren_hz",   # Hz, frequency of that bin
)


def weighting_sos(kind: str, fs: int = FS) -> np.ndarray:
    """IEC 61672 A or C weighting as a digital filter (bilinear transform), 0 dB at 1 kHz.

    At 48 kHz this is within 0.1 dB of the standard up to 5 kHz and within Class 1 tolerance at
    8-10 kHz (-0.5 and -1.2 dB).
    """
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    w = lambda f: 2 * np.pi * f  # noqa: E731
    if kind == "A":
        zeros, poles = [0.0] * 4, [-w(f1)] * 2 + [-w(f2), -w(f3)] + [-w(f4)] * 2
    elif kind == "C":
        zeros, poles = [0.0] * 2, [-w(f1)] * 2 + [-w(f4)] * 2
    else:
        raise ValueError(f"unknown weighting {kind!r}; choose 'A' or 'C'")
    z, p, k = signal.bilinear_zpk(zeros, poles, 1.0, fs)
    sos = signal.zpk2sos(z, p, k)
    _, h = signal.sosfreqz(sos, worN=[1000.0], fs=fs)
    sos[0, :3] /= abs(h[0])
    return sos


def db(power) -> np.ndarray:
    """Mean-square value (1.0 = full scale) to dB."""
    return 10.0 * np.log10(np.maximum(power, EPS))


class Meter:
    """Streaming meter: feed() int16 blocks, get feature columns for every completed 10 ms hop.

    seed_floor=(floor_a, floor_c) starts the noise floor at known values (a saved trial carries
    the floor from when it was recorded); otherwise the floor is learned from the first hops.
    """

    def __init__(self, fs: int = FS, seed_floor: tuple[float, float] | None = None):
        if fs != FS:
            raise ValueError(f"the meter runs at {FS} Hz; got {fs} (resample first)")
        self.sos = {k: weighting_sos(k) for k in ("A", "C")}
        self.sos["Z"] = signal.butter(2, 20.0, "highpass", fs=fs, output="sos")
        self.zi = {k: np.zeros((s.shape[0], 2)) for k, s in self.sos.items()}
        self.alpha = 1.0 - np.exp(-1.0 / (TAU_FAST * fs))
        self.fast_state = EPS                    # Fast detector's mean square
        self.pending = np.zeros(0, np.int16)     # samples that don't fill a hop yet
        self.c2_tail = np.zeros(HOP - 1)         # last C-weighted squares, for the sliding 10 ms window
        self.z_tail = np.zeros(WIN - HOP)        # last Z samples, for the spectrum window
        self.lcmax_hist = np.full(ONSET_FROM, -200.0)
        n_floor = int(FLOOR_S * FS / HOP)
        self.hist_a = np.full(n_floor, np.nan)
        self.hist_c = np.full(n_floor, np.nan)
        self.hist_i = 0
        self.floor_a = self.floor_c = np.nan
        if seed_floor is not None:
            self.hist_a[:] = seed_floor[0]
            self.hist_c[:] = seed_floor[1]
            self.floor_a, self.floor_c = seed_floor
        self.hops = 0
        self._spectrum_setup()

    def _spectrum_setup(self) -> None:
        self.window = signal.windows.hann(WIN, sym=False)
        self.amp_norm = self.window.sum() / 2.0          # a sine of amplitude A peaks at A * amp_norm
        self.freqs = np.fft.rfftfreq(NFFT, 1.0 / FS)
        df = FS / NFFT
        band = lambda lo, hi: (self.freqs >= lo) & (self.freqs < hi)  # noqa: E731
        self.m_voice, self.m_span = band(*VOICE_BAND), band(*SPEECH_SPAN)
        self.tone_idx = np.flatnonzero(band(*TONE_BAND))
        self.siren_idx = np.flatnonzero(band(*SIREN_BAND))
        self.m_pitch = band(*PITCH_BAND)
        # The window's own autocorrelation, so periodicity isn't biased toward short lags.
        w_pow = np.abs(np.fft.rfft(self.window, NFFT)) ** 2
        rw = np.fft.irfft(w_pow, NFFT)[: PITCH_LAGS[1] + 1]
        self.rw = rw / rw[0]                              # smooth, 1.0 at lag 0, about 0.5 at lag 600
        edges = np.geomspace(SPEC_LO, SPEC_HI, SPEC_BINS + 1)
        lo = np.ceil(edges[:-1] / df).astype(int)
        hi = np.floor(edges[1:] / df).astype(int)
        centre = np.round(np.sqrt(edges[:-1] * edges[1:]) / df).astype(int)
        starts = np.where(hi >= lo, lo, centre)          # a bin narrower than one FFT bin takes the nearest
        self.spec_starts = np.maximum.accumulate(starts)
        self.spec_end = int(hi[-1]) + 1
        self.spec_freqs = np.sqrt(edges[:-1] * edges[1:])

    # ------------------------------------------------------------------ feeding
    def feed(self, samples: np.ndarray) -> dict[str, np.ndarray]:
        """Process a block of int16 samples. Returns columns (and 'spec', hops x 96 dB) for the
        hops completed by this block; empty columns if the block didn't complete one."""
        x = np.concatenate([self.pending, np.asarray(samples, np.int16).ravel()])
        k = len(x) // HOP
        self.pending = x[k * HOP:]
        if k == 0:
            return empty_frames()
        raw = x[: k * HOP].astype(np.float64) / 32768.0
        return self._process(raw, k)

    def _process(self, raw: np.ndarray, k: int) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        hops = raw.reshape(k, HOP)
        out["t"] = (self.hops + 1 + np.arange(k)) * HOP / FS
        out["peak"] = db(np.abs(hops).max(axis=1) ** 2)
        out["clip"] = (np.abs(hops) >= 32767 / 32768.0).sum(axis=1).astype(float)
        out["dc"] = hops.mean(axis=1)

        a, self.zi["A"] = signal.sosfilt(self.sos["A"], raw, zi=self.zi["A"])
        c, self.zi["C"] = signal.sosfilt(self.sos["C"], raw, zi=self.zi["C"])
        z, self.zi["Z"] = signal.sosfilt(self.sos["Z"], raw, zi=self.zi["Z"])
        a2, c2 = a * a, c * c
        out["la"] = db(a2.reshape(k, HOP).mean(axis=1))
        out["lc"] = db(c2.reshape(k, HOP).mean(axis=1))

        # Sliding 10 ms C level at every sample, so a clap reads the same wherever it falls in a hop.
        c2_ext = np.concatenate([self.c2_tail, c2])
        cs = np.concatenate([[0.0], np.cumsum(c2_ext)])
        moving = (cs[HOP:] - cs[:-HOP]) / HOP           # one value per sample of this block
        out["lc_max"] = db(moving.reshape(k, HOP).max(axis=1))
        self.c2_tail = c2_ext[-(HOP - 1):]

        # Fast exponential detector on the A-weighted squares (exact, per sample).
        fast, _ = signal.lfilter([self.alpha], [1.0, self.alpha - 1.0], a2, zi=[self.fast_state * (1 - self.alpha)])
        self.fast_state = fast[-1]
        out["laf"] = db(fast.reshape(k, HOP).max(axis=1))

        # Onset: how far lc_max jumped above the loudest of 300..30 ms ago.
        hist = np.concatenate([self.lcmax_hist, out["lc_max"]])
        past = sliding_window_view(hist[:-1], ONSET_FROM - ONSET_TO + 1)[: k].max(axis=1)
        out["onset_c"] = out["lc_max"] - past
        self.lcmax_hist = hist[-ONSET_FROM:]

        out["floor_a"], out["floor_c"] = self._floors(out["la"], out["lc"])
        self._spectrum(z, k, out)
        self.hops += k
        return out

    def _floors(self, la: np.ndarray, lc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fa, fc = np.empty(len(la)), np.empty(len(la))
        n = len(self.hist_a)
        for i in range(len(la)):
            self.hist_a[self.hist_i] = la[i]
            self.hist_c[self.hist_i] = lc[i]
            self.hist_i = (self.hist_i + 1) % n
            filled = self.hops + i + 1
            if np.isnan(self.floor_a) or (self.hops + i) % FLOOR_EVERY == 0 or filled < FLOOR_EVERY:
                self.floor_a = float(np.nanpercentile(self.hist_a, FLOOR_PCT))
                self.floor_c = float(np.nanpercentile(self.hist_c, FLOOR_PCT))
            fa[i], fc[i] = self.floor_a, self.floor_c
        return fa, fc

    def _tone(self, P: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Strongest bin in a band: how far it stands above the band's median (dB), and its frequency."""
        k = P.shape[0]
        Pt = P[:, idx]
        j = Pt.argmax(axis=1)
        prom = db(Pt[np.arange(k), j]) - db(np.median(Pt, axis=1))
        # Parabolic interpolation on the log spectrum for a steadier frequency.
        jl, jr = np.clip(j - 1, 0, Pt.shape[1] - 1), np.clip(j + 1, 0, Pt.shape[1] - 1)
        yl, y0, yr = (np.log(Pt[np.arange(k), q] + EPS) for q in (jl, j, jr))
        den = yl - 2 * y0 + yr
        delta = np.where(np.abs(den) > 1e-12, 0.5 * (yl - yr) / np.where(den == 0, 1, den), 0.0)
        return prom, self.freqs[idx[j]] + np.clip(delta, -0.5, 0.5) * FS / NFFT

    def _spectrum(self, z: np.ndarray, k: int, out: dict) -> None:
        z_ext = np.concatenate([self.z_tail, z])
        self.z_tail = z_ext[-(WIN - HOP):]
        frames = sliding_window_view(z_ext, WIN)[::HOP][:k] * self.window
        X = np.fft.rfft(frames, NFFT, axis=1)
        P = X.real ** 2 + X.imag ** 2
        span = P[:, self.m_span].sum(axis=1) + EPS
        out["voice_ratio"] = P[:, self.m_voice].sum(axis=1) / span
        out["centroid"] = (P[:, self.m_span] * self.freqs[self.m_span]).sum(axis=1) / span

        # Whiten first (divide by the smoothed spectrum), so rumble and other colored noise don't
        # look periodic; the harmonics of a voice or a tone stay peaky.
        env = uniform_filter1d(P, size=WHITEN_BINS, axis=1, mode="nearest") + EPS
        r = np.fft.irfft((P / env) * self.m_pitch, NFFT, axis=1)[:, : PITCH_LAGS[1] + 1]
        nr = (r / np.maximum(r[:, :1], EPS)) / self.rw
        per = nr[:, PITCH_LAGS[0]: PITCH_LAGS[1] + 1].max(axis=1)
        out["periodicity"] = np.where(r[:, 0] > 1e-14, np.clip(per, 0.0, 1.0), 0.0)

        out["tone_prom"], out["tone_hz"] = self._tone(P, self.tone_idx)
        out["siren_prom"], out["siren_hz"] = self._tone(P, self.siren_idx)

        spec_db = db(P[:, : self.spec_end] / self.amp_norm ** 2)  # dBFS per bin: a full-scale sine peaks at 0
        out["spec"] = np.maximum.reduceat(spec_db, self.spec_starts, axis=1).astype(np.float32)


def empty_frames() -> dict[str, np.ndarray]:
    f = {c: np.zeros(0) for c in COLUMNS}
    f["spec"] = np.zeros((0, SPEC_BINS), np.float32)
    return f


def concat_frames(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    parts = [p for p in parts if len(p["t"])]
    if not parts:
        return empty_frames()
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def slice_frames(f: dict[str, np.ndarray], start: int, stop: int | None = None) -> dict[str, np.ndarray]:
    return {key: v[start:stop] for key, v in f.items()}


def analyze(samples: np.ndarray, seed_floor: tuple[float, float] | None = None,
            chunk_s: float = 1.0) -> dict[str, np.ndarray]:
    """Run a whole recording through a fresh Meter (in 1 s chunks to keep memory small)."""
    m = Meter(seed_floor=seed_floor)
    step = int(chunk_s * FS)
    return concat_frames([m.feed(samples[i: i + step]) for i in range(0, len(samples), step)])
