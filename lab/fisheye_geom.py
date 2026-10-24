"""Pupper's fisheye, the way Lab 5 measured it: where in the world a pixel looks.

Lab 5 (follow_me_pupper) calibrated the lens as a double sphere camera model
(robot/camera_params.yaml: fx, fy, cx, cy, xi, alpha for the 1400 x 1050 frame) and ran its
detector on the undistorted *equirectangular* image: IMAGE_WIDTH x IMAGE_HEIGHT = 700 x 572 px
spanning H_FOV_DEG = 220 degrees across and 180 degrees top to bottom, so every row is a latitude
(0 at the top = 90 degrees up, 572 at the bottom = 90 degrees down) and every column a longitude.
Its distance sensor is the row of a thing's bottom edge, where it meets the floor: lower in the
image is nearer (Lab 5 step 4). STOP_BOTTOM = 454 is the row its follower stops at, measured on
the robot: a polite distance from a person's feet. Nothing here needs metres or a camera height,
which is the point: the pitch of the camera and the lens are all inside the calibration.

This file is the same model without numpy (lab/ runs without it in the desk tests): the inverse
projection, pixel -> ray, and the ray -> equirectangular row and column.

  unproject(u, v)             a raw-frame pixel -> a unit ray (x right, y down, z forward)
  latlon_deg(u, v)            that ray's latitude (+ down) and longitude (+ right), degrees
  nearness(u_frac, v_frac)    the equirectangular row of a point, as a fraction of IMAGE_HEIGHT
                              (Lab 5's distance sensor; STOP_NEARNESS = 454 / 572)
  bearing_deg(u_frac, ...)    the longitude: degrees right of straight ahead
"""

from __future__ import annotations

import math

# Lab 5's calibration (follow_me_pupper/robot/camera_params.yaml), for the raw 1400 x 1050 frame.
FX = 228.81030
FY = 228.81030
CX = 680.0
CY = 479.0
XI = -0.00092
ALPHA = 0.62007
FRAME_W = 1400
FRAME_H = 1050

# Lab 5's undistorted view and its numbers (follow_me_pupper/lab/boxes.py, lab/target_lock.py).
IMAGE_WIDTH = 700
IMAGE_HEIGHT = 572
H_FOV_DEG = 220.0
STOP_BOTTOM = 454  # px: the follower stops when the person's feet reach this row (measured on the robot)
STOP_NEARNESS = STOP_BOTTOM / IMAGE_HEIGHT  # 0.794
SLOWDOWN_PX = 120  # px before STOP_BOTTOM over which the follower slows to a stop (lab/target_lock.py)
SLOW_NEARNESS = (STOP_BOTTOM - SLOWDOWN_PX) / IMAGE_HEIGHT  # 0.584
STOP_TOLERANCE_PX = 15  # px: within this of STOP_BOTTOM the follower stands still


def unproject(u: float, v: float) -> tuple[float, float, float] | None:
    """A raw-frame pixel -> a unit ray (x right, y down, z forward), or None outside the lens."""
    mx = (u - CX) / FX
    my = (v - CY) / FY
    r2 = mx * mx + my * my
    inner = 1.0 - (2.0 * ALPHA - 1.0) * r2
    if inner < 0.0:
        return None  # outside the image circle
    mz = (1.0 - ALPHA * ALPHA * r2) / (ALPHA * math.sqrt(inner) + 1.0 - ALPHA)
    k = (mz * XI + math.sqrt(mz * mz + (1.0 - XI * XI) * r2)) / (mz * mz + r2)
    x, y, z = k * mx, k * my, k * mz - XI
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    return x / n, y / n, z / n


def latlon_deg(u: float, v: float) -> tuple[float, float] | None:
    """(latitude, longitude) in degrees of a raw-frame pixel: latitude + is down (toward the
    floor), longitude + is to the right. None outside the lens."""
    ray = unproject(u, v)
    if ray is None:
        return None
    x, y, z = ray
    return math.degrees(math.asin(max(-1.0, min(1.0, y)))), math.degrees(math.atan2(x, z))


def nearness(u_frac: float, v_frac: float, w: int = FRAME_W, h: int = FRAME_H) -> float | None:
    """Lab 5's distance sensor for a point given as fractions of the frame: its row in the
    equirectangular image as a fraction of IMAGE_HEIGHT (0 = straight up, 0.5 = the horizon,
    1 = straight down). Compare with STOP_NEARNESS. None outside the lens."""
    ll = latlon_deg(u_frac * w, v_frac * h)
    if ll is None:
        return None
    lat, _ = ll
    return (lat + 90.0) / 180.0


def bearing_deg(u_frac: float, v_frac: float, w: int = FRAME_W, h: int = FRAME_H) -> float | None:
    """Degrees to the right of straight ahead for a point given as fractions of the frame."""
    ll = latlon_deg(u_frac * w, v_frac * h)
    return None if ll is None else ll[1]


CAM_HEIGHT_M = 0.22  # only for pushing a row along with the robot's own walking (a proxy, not a measurement)


def row_after_walking(row: float, walked_m: float) -> float:
    """Where a floor point that was at ``row`` sits after the robot walked ``walked_m`` toward it:
    the row -> a proxy distance for a level camera CAM_HEIGHT_M up -> less the walk -> a row. Dead
    reckoning for a fix that is a second or two old; the proxy's metres cancel out."""
    lat = row * 180.0 - 90.0
    if lat < 1.0:
        return row
    d = max(0.05, CAM_HEIGHT_M / math.tan(math.radians(lat)) - walked_m)
    return (math.degrees(math.atan2(CAM_HEIGHT_M, d)) + 90.0) / 180.0


def box_nearness(box: list, w: int = FRAME_W, h: int = FRAME_H) -> float | None:
    """A box [x, y, w, h] in frame fractions -> the nearness of its bottom edge's centre."""
    return nearness(box[0] + box[2] / 2.0, box[1] + box[3], w, h)


def box_bearing_deg(box: list, w: int = FRAME_W, h: int = FRAME_H) -> float:
    """A box's bearing; outside the lens, the straight-line guess from the frame's width."""
    b = bearing_deg(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0, w, h)
    return b if b is not None else (box[0] + box[2] / 2.0 - 0.5) * 180.0


if __name__ == "__main__":
    for name, u, v in (("centre", CX, CY), ("disc bottom", CX, CY + 400), ("disc right", CX + 400, CY),
                       ("STOP row on axis", CX, None)):
        if v is None:  # which raw row sits at Lab 5's STOP_BOTTOM: scan down the centre column
            for vv in range(int(CY), FRAME_H):
                n = nearness(u / FRAME_W, vv / FRAME_H)
                if n is not None and n >= STOP_NEARNESS:
                    print(f"{name}: raw row {vv} ({vv / FRAME_H:.3f} of the frame) -> nearness {n:.3f}")
                    break
            continue
        print(f"{name}: lat/lon {latlon_deg(u, v)}, nearness {nearness(u / FRAME_W, v / FRAME_H)}")
