#!/usr/bin/env python3
"""Desk tests for the eyes service's free-space summary, on synthetic depth maps (no model, no robot).

    python3 vision/test_eyes.py

Pupper's camera is a 180-degree fisheye a known height above the floor, looking level: in the disc
crop, a row's elevation below the horizon is linear in its distance from the centre row, and the
floor at elevation e is height / tan(e) away. We build closeness maps (bigger = closer, like the
depth model's output, with an arbitrary scale and offset) for an empty floor and for things standing
on it (a thin chair leg, a wide box, a cable, a wall, a shoe) and check that freespace_from_closeness
finds each thing's base and turns it into about the right distance, and sees floor where there is
only floor. The threshold lab/hackathon_vision.py enforces is closest_m <= 0.35.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eyes import STRIPS, circle_mask, freespace_from_closeness  # noqa: E402

STOP_M = 0.45  # keep in step with lab/hackathon_vision.py
N = 252  # the disc crop, as the depth model sees it
H = 0.22  # the camera's height above the floor, metres (standing)
PASS = FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    PASS += ok
    FAIL += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if ok or not detail else f": {detail}"))


def rays():
    """Per pixel of the crop: bearing (deg, + right) and elevation (deg, + below the horizon)."""
    yy, xx = np.mgrid[0:N, 0:N]
    bearing = (xx / N - 0.5) * 180.0
    elevation = (yy / N - 0.5) * 180.0
    return bearing, elevation


def floor(h: float = H) -> np.ndarray:
    """Depth (metres along the ray) of an empty floor; far (6 m) above the horizon."""
    _, elev = rays()
    d = np.full((N, N), 6.0)
    below = elev > 1.0
    d[below] = np.minimum(6.0, h / np.sin(np.radians(elev[below])))
    return d


def put_object(depth: np.ndarray, bearing0: float, bearing1: float, distance_m: float, height_m: float,
               h: float = H) -> np.ndarray:
    """Stand a thing on the floor, distance_m ahead, spanning bearings bearing0..bearing1, height_m tall."""
    d = depth.copy()
    bearing, elev = rays()
    base = np.degrees(np.arctan2(h, distance_m))  # its base, below the horizon
    top = np.degrees(np.arctan2(h - height_m, distance_m))  # its top (negative: above the horizon)
    hit = (bearing >= bearing0) & (bearing <= bearing1) & (elev <= base) & (elev >= top)
    ray = distance_m / np.cos(np.radians(elev[hit]))
    d[hit] = np.minimum(d[hit], ray)
    return d


def summary(depth: np.ndarray, scale: float = 3.0, offset: float = 0.2, h: float = H) -> dict[str, dict]:
    closeness = scale / depth + offset  # a relative model's output: 1/depth up to an unknown scale and offset
    return {s["name"]: s for s in freespace_from_closeness(closeness, None, 180.0, circle_mask(N), h)}


def blocked(st: dict[str, dict]) -> list[str]:
    return [n for n in STRIPS if st[n]["closest_m"] is not None and st[n]["closest_m"] <= STOP_M]


BLOCK_NEARNESS = 0.68  # keep in step with lab/hackathon_vision.py
FRAME = (1400, 1050)
GEOM = (680.0 - 467.0, 479.0 - 467.0, 934.0)  # the disc crop around Lab 5's calibrated centre (fisheye_geom CX, CY)


def floor_scene(things=(), h: float = H, n: int = 224, scale: float = 3.0, offset: float = 0.2) -> np.ndarray:
    """A closeness map of the disc crop, ray-cast through the calibrated lens: a floor h below the
    camera, and things standing on it, each (ahead_m, right_from_m, right_to_m, height_m): a box face
    that far ahead. Closeness = scale / distance along the ray + offset, like a relative model's."""
    import eyes

    rays = eyes.floor_rays(n, GEOM, FRAME)
    dist = np.where(rays.d > 1e-3, h / np.maximum(rays.d, 1e-3), 20.0)
    for thing in things:
        ahead, r0, r1, height = thing[:4]
        bottom = thing[4] if len(thing) > 4 else 0.0  # above the floor: a shelf, with nothing beneath it
        t = ahead / np.maximum(rays.f, 1e-6)
        hit = (rays.f > 1e-3) & (t * rays.r >= r0) & (t * rays.r <= r1) & (h - t * rays.d >= bottom) & (h - t * rays.d <= height)
        dist = np.where(hit, np.minimum(dist, t), dist)
    out = np.zeros(n * n)
    out[rays.flat] = scale / np.minimum(dist, 20.0) + offset
    return out.reshape(n, n)


def floor_rule() -> None:
    """The floor-anchored rule (freespace_floor) on ray-cast scenes, and the lens rim finder."""
    import eyes

    print("\nEyes: metres from the floor (freespace_floor, through Lab 5's calibrated lens)")

    def run(things=(), h: float = H, **kw):
        strips, fit = eyes.freespace_floor(floor_scene(things, h, **kw), GEOM, FRAME, h)
        return ({s["name"]: s for s in strips} if strips else None), fit

    def hits(st):
        return [n for n in STRIPS if st[n]["nearness"] is not None and st[n]["nearness"] >= BLOCK_NEARNESS]

    import fisheye_geom

    rays, n = eyes.floor_rays(224, GEOM, FRAME), 224
    worst = 0.0
    for i in rays.flat[:: max(1, len(rays.flat) // 400)]:
        row, col = divmod(int(i), n)
        u, v = GEOM[0] + (col + 0.5) / n * GEOM[2], GEOM[1] + (row + 0.5) / n * GEOM[2]
        x, y, z = fisheye_geom.unproject(u, v)
        j = int(np.searchsorted(rays.flat, i))
        worst = max(worst, abs(rays.r[j] - x), abs(rays.d[j] - y), abs(rays.f[j] - z))
    check("every pixel's ray is Lab 5's (fisheye_geom.unproject)", worst < 1e-6, f"worst difference {worst:.2e}")

    st, fit = run()
    check("an empty floor fits, and is clear in every strip", st is not None and all(s["closest_m"] is None for s in st.values()),
          str(fit))
    st, _ = run([(0.30, -0.02, 0.02, 0.4)])
    check("a chair leg 30 cm ahead blocks the centre strip only (by Lab 5's row, as the robot reads it)",
          st is not None and hits(st) == ["center"], str(st and hits(st)))
    check("... at about the right distance (0.30 m, within 15 %)", st is not None and st["center"]["closest_m"] is not None
          and abs(st["center"]["closest_m"] - 0.30) <= 0.045, str(st and st["center"]))
    st, _ = run([(0.80, -0.02, 0.02, 0.4)])
    check("the same leg 0.8 m away is seen at about 0.8 m and does not block", st is not None and not hits(st)
          and st["center"]["closest_m"] is not None and abs(st["center"]["closest_m"] - 0.8) <= 0.12, str(st and st["center"]))
    st, _ = run([(0.30, -0.30, -0.05, 0.3)])
    check("a wide box ahead-left blocks centre-left, not the right", st is not None and "center-left" in hits(st)
          and "right" not in hits(st) and "center-right" not in hits(st), str(st and hits(st)))
    st, _ = run([(0.30, 0.12, 0.25, 0.10)])
    check("a shoe 30 cm ahead on the right blocks centre-right only", st is not None and hits(st) == ["center-right"],
          str(st and hits(st)))
    st, _ = run([(0.30, -0.6, 0.6, 0.015)])
    check("a 1.5 cm cable lying flat is not an obstacle", st is not None and not hits(st), str(st and hits(st)))
    st, _ = run([(0.30, -0.15, 0.15, 0.40, 0.25)])
    check("a shelf 25-40 cm up, 30 cm ahead, nothing beneath: found at its own distance (no base on the floor)",
          st is not None and st["center"]["closest_m"] is not None and abs(st["center"]["closest_m"] - 0.30) <= 0.045,
          str(st and st["center"]))
    st, _ = run([(0.06, 0.12, 0.16, 0.10)])
    check("Pupper's own paw beside the camera is not an obstacle", st is not None and not hits(st), str(st and hits(st)))
    a, _ = run([(0.30, -0.02, 0.02, 0.4)])
    b, _ = run([(0.30, -0.02, 0.02, 0.4)], scale=0.5, offset=1.0)
    check("the result does not depend on the depth model's scale or offset", a == b)
    st, _ = run([(0.30, -0.02, 0.02, 0.4)], h=0.12)
    check("lying down (camera 12 cm up) the leg is still found at about 0.30 m", st is not None
          and st["center"]["closest_m"] is not None and abs(st["center"]["closest_m"] - 0.30) <= 0.045, str(st and st["center"]))
    st, fit = run([(0.15, -3.0, 3.0, 2.0)])
    check("no floor in view (a wall 15 cm ahead): no fit, so no floor-anchored answer", st is None and fit is None)
    cl = floor_scene([(0.15, -3.0, 3.0, 2.0)])
    _, used = eyes.freespace_summary(cl, None, 180.0, circle_mask(224), H, GEOM, FRAME)
    check("... and the base-of-surface rule answers instead", used is None)
    _, used = eyes.freespace_summary(floor_scene(), None, 180.0, circle_mask(224), H, GEOM, FRAME, "surface")
    check("--depth-method surface: the base-of-surface rule only", used is None)

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  (the rim finder's check needs pillow: skipped)")
        return
    im = Image.new("RGB", FRAME)
    draw = ImageDraw.Draw(im)
    draw.rectangle((380, 0, 980, 60), fill=(255, 255, 255))  # the ceiling lights' glare above the lens
    draw.ellipse((680 - 467, 479 - 467, 680 + 467, 479 + 467), fill=(120, 120, 110), outline=(230, 120, 40), width=8)
    cx, cy, r = eyes.ring_circle(im) or (0, 0, 0)
    check("the rim finder finds the disc from its orange rim, glare or not",
          abs(cx - 680 / 1400) < 0.01 and abs(cy - 479 / 1050) < 0.01 and abs(r - 467 / 1400) < 0.01, f"{cx}, {cy}, {r}")


def depth_models() -> None:
    """The depth model's plumbing, with no model and no network: which file is loaded, the checked
    download, where it runs, and the mirror averaging."""
    import hashlib
    import re
    import tempfile

    import eyes

    print("\nEyes: the depth model (which one, the checked download, where it runs)")
    pinned = all(re.search(r"/resolve/[0-9a-f]{40}/", m["url"]) and re.fullmatch(r"[0-9a-f]{64}", m["sha256"])
                 for m in eyes.DEPTH_MODELS.values())
    check("every depth model is pinned to one commit and has a SHA-256", pinned)
    check("the default is distill-any-depth-small", eyes.DEFAULT_DEPTH_NAME == "distill-any-depth-small"
          and eyes.DEFAULT_DEPTH_MODEL.name == eyes.DEPTH_MODELS["distill-any-depth-small"]["file"])

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = d / eyes.DEPTH_MODELS["depth-anything-v2-small"]["file"]
        check("no model file at all: no depth (and no crash)", eyes.choose_depth_model(models_dir=d) == (None, eyes.DEFAULT_DEPTH_NAME))
        old.write_bytes(b"old model")
        check("only the old model there (no internet): it is used, by its name",
              eyes.choose_depth_model(models_dir=d) == (old, "depth-anything-v2-small"))
        new = d / eyes.DEPTH_MODELS["distill-any-depth-small"]["file"]
        new.write_bytes(b"new model")
        check("both there: the default", eyes.choose_depth_model(models_dir=d) == (new, "distill-any-depth-small"))
        check("--depth-model-name picks the other", eyes.choose_depth_model("depth-anything-v2-small", models_dir=d)
              == (old, "depth-anything-v2-small"))
        mine = d / "mine.onnx"
        check("--depth-model: a file that isn't there is no depth, not a fallback",
              eyes.choose_depth_model(path=mine, models_dir=d) == (None, "mine"))

        src = d / "served.onnx"
        src.write_bytes(b"the real weights")
        good = hashlib.sha256(b"the real weights").hexdigest()
        dest = d / "got" / "model.onnx"
        try:
            eyes.fetch_depth_model(dest, src.as_uri(), "0" * 64)
            refused = False
        except ValueError:
            refused = True
        check("a download that doesn't match its SHA-256 is refused and deleted",
              refused and not dest.exists() and not dest.with_suffix(".part").exists())
        check("one that matches is kept", eyes.fetch_depth_model(dest, src.as_uri(), good) == dest and dest.read_bytes() == b"the real weights")

    cpu, mac, nv = ["CPUExecutionProvider"], ["CoreMLExecutionProvider", "CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]
    check("provider cpu: the CPU only, even with an accelerator", eyes.onnx_providers("cpu", mac) == cpu)
    check("provider auto on a Mac: CoreML, then the CPU", eyes.onnx_providers("auto", mac) == mac)
    check("provider auto on a Jetson: CUDA, then the CPU", eyes.onnx_providers("auto", nv) == nv)
    check("provider auto with nothing else: the CPU", eyes.onnx_providers("auto", cpu) == cpu)
    check("provider coreml without CoreML: the CPU (with a warning)", eyes.onnx_providers("coreml", nv) == cpu)

    class Session:  # a stand-in model: a ramp that rises to the right (a bias), plus the input's red channel
        def run(self, _outputs, feeds):
            x = feeds["pixel_values"]
            return [np.linspace(0.0, 1.0, x.shape[-1])[None, None, :] + x[:, 0]]

    dm = eyes.Depth.__new__(eyes.Depth)
    dm.session, dm.input_name, dm.tta = Session(), "pixel_values", False
    x = np.zeros((1, 3, 14, 28), dtype=np.float32)
    x[0, 0, :, 3] = 1.0  # a bright column, left of centre
    plain = dm.predict(x)
    dm.tta = True
    avg = dm.predict(x)
    check("tta: the same shape as one pass", avg.shape == plain.shape == (14, 28), f"{avg.shape} {plain.shape}")
    # The mirror pass, mirrored back, has the column in the same place and the ramp reversed: the
    # average keeps the column (a real thing) and flattens the ramp (the model's left-right bias).
    check("tta: things stay where they are, a left-right bias averages out", np.allclose(avg, 0.5 + x[0, 0]),
          str(np.round(avg[0], 2)))


def main() -> None:
    print("Eyes: free space from synthetic fisheye depth")
    d0 = floor()
    st = summary(d0)
    check("an empty floor is floor only in every strip", all(s["base_deg"] is None for s in st.values()),
          str({n: s["base_deg"] for n, s in st.items()}))

    leg = put_object(d0, -2.0, 2.0, distance_m=0.30, height_m=0.4)  # a chair leg, 4 degrees wide, 30 cm ahead
    st = summary(leg)
    check("a thin chair leg 30 cm ahead blocks the centre strip only", blocked(st) == ["center"], str(blocked(st)))
    check("... at about the right distance (0.30 m, within 25 %)", st["center"]["closest_m"] is not None
          and abs(st["center"]["closest_m"] - 0.30) <= 0.075, str(st["center"]))

    leg_far = put_object(d0, -2.0, 2.0, distance_m=0.8, height_m=0.4)
    st = summary(leg_far)
    check("the same leg 0.8 m away is seen but does not block", not blocked(st) and st["center"]["closest_m"] is not None
          and abs(st["center"]["closest_m"] - 0.8) <= 0.2, str(st["center"]))

    box = put_object(d0, -40.0, -5.0, distance_m=0.3, height_m=0.3)
    st = summary(box)
    check("a wide box ahead-left blocks centre-left (and the centre edge), not the right",
          "center-left" in blocked(st) and "right" not in blocked(st) and "center-right" not in blocked(st), str(blocked(st)))

    cable = put_object(d0, -60.0, 60.0, distance_m=0.3, height_m=0.015)
    st = summary(cable)
    check("a 1.5 cm cable is below what the eyes can tell from floor (the detector is not for cables on the floor)",
          not blocked(st), str(blocked(st)))

    wall = put_object(d0, -90.0, 90.0, distance_m=0.3, height_m=2.0)
    st = summary(wall)
    check("a wall 30 cm ahead blocks every strip", set(blocked(st)) == set(STRIPS), str(blocked(st)))

    wall_far = put_object(d0, -90.0, 90.0, distance_m=1.5, height_m=2.0)
    st = summary(wall_far)
    check("a wall 1.5 m ahead does not", not blocked(st), str({n: s["closest_m"] for n, s in st.items()}))

    both = put_object(put_object(d0, -90.0, 90.0, distance_m=1.5, height_m=2.0), -2.0, 2.0, distance_m=0.3, height_m=0.4)
    st = summary(both)
    check("a near chair leg in front of a far wall: the leg is reported, not the wall", blocked(st) == ["center"]
          and st["center"]["closest_m"] is not None and st["center"]["closest_m"] <= 0.4
          and (st["left"]["closest_m"] is None or st["left"]["closest_m"] > 1.0), str({n: s["closest_m"] for n, s in st.items()}))

    shoe = put_object(d0, 20.0, 40.0, distance_m=0.3, height_m=0.1)
    st = summary(shoe)
    check("a shoe 30 cm ahead on the right blocks centre-right only", blocked(st) == ["center-right"], str(blocked(st)))

    lying = summary(put_object(floor(0.12), -2.0, 2.0, distance_m=0.30, height_m=0.4, h=0.12), h=0.12)
    check("lying down (camera 12 cm up) the same leg is still found at about 0.30 m", lying["center"]["closest_m"] is not None
          and abs(lying["center"]["closest_m"] - 0.30) <= 0.075, str(lying["center"]))

    a, b = summary(leg), summary(leg, scale=0.5, offset=1.0)
    check("the summary does not depend on the depth model's scale or offset",
          all(a[n]["base_deg"] == b[n]["base_deg"] for n in STRIPS), f"{[a[n]['base_deg'] for n in STRIPS]} vs {[b[n]['base_deg'] for n in STRIPS]}")

    from eyes import IDLE_HZ, next_period
    check("gating: full rate while the robot needs eyes", next_period({"attention": True}, 5.0) == 0.2)
    check("gating: one frame every 5 s while idle", next_period({"attention": False}, 5.0) == 1.0 / IDLE_HZ)
    check("gating: stand down while another eyes service posts", next_period({"attention": True, "others_posting": True}, 5.0) is None)
    check("gating: no status at all means full rate (the robot's reply decides, not a guess)", next_period({"attention": True}, 2.0) == 0.5)

    floor_rule()
    depth_models()

    print()
    if FAIL:
        print(f"{FAIL} check(s) failed, {PASS} passed.")
        raise SystemExit(1)
    print(f"All {PASS} eyes checks passed.")


if __name__ == "__main__":
    main()
