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

    print()
    if FAIL:
        print(f"{FAIL} check(s) failed, {PASS} passed.")
        raise SystemExit(1)
    print(f"All {PASS} eyes checks passed.")


if __name__ == "__main__":
    main()
