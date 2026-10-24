#!/usr/bin/env python3
"""Pupper's eyes: depth for not bumping into things, Gemma for understanding. Runs beside the robot
server, on a laptop on the same network (fast) or on the Pi itself (slower).

    python3 vision/eyes.py --robot http://localhost:8000          # through the SSH tunnel to the Pupper
    python3 vision/eyes.py --image photo.jpg                      # one image: print the depth summary and exit
    python3 vision/eyes.py --image photo.jpg --look "what is on the floor?"   # ... and ask Gemma
    python3 vision/eyes.py --robot http://localhost:8000 --no-gemma           # depth only, no Ollama needed

What it does, a few times a second (``--hz``) while the robot needs eyes (a motion running or queued,
follow mode on, a tool called in the last 10 s), once every 5 s while it is idle, and nothing at all
while another eyes service (a laptop's, the Jetson's) is posting, so the Pi stays cool:

  1. fetches the camera frame the robot server already serves (GET /camera/image),
  2. runs a monocular depth model on it (Depth Anything V2 Small, ONNX, on the CPU) and posts a
     free-space summary to POST /vision/freespace: for five vertical strips of the lower half of the
     frame, how close the nearest surface is. lab/hackathon_vision.py turns that into "blocked" and
     refuses or stops moves. This is class-agnostic: a chair leg, a cable, a shoe all count,
  3. asks the robot what go_to wants found (GET /vision/wanted) and, when there is something, asks
     Gemma where it is in the frame and posts the box to POST /vision/target (about once a second),
  4. when nothing is wanted, every ``--describe-every`` seconds asks Gemma to describe the scene and
     posts it to POST /vision/scene (what_is_around reads it),
  5. answers POST /look {"question"} on its own port (``--port``, default 8100) with Gemma's answer
     about the latest frame (the robot's look tool), and POST /exists {"name"} with whether a thing
     is in the frame right now: the detector's word for a thing with a class, else Gemma's yes or no
     (the robot's explore tool asks at every stop).

Gemma runs in Ollama (https://ollama.com): ``ollama pull gemma3:4b`` (a vision model; on-device, no
cloud), or, until the Jetson is here, through Google's API with the Gemini key (``--gemma-api``:
gemini-3.5-flash-lite by default, cloud, so not for the final hackathon build). Depth needs the ONNX file (see
fetch_depth_model() or --depth-model). Without the depth model this still does 3-5; without Gemma it
still does 1-2.

Install:  python3 -m pip install -r vision/requirements.txt
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

log = logging.getLogger("eyes")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lab"))
import fisheye_geom  # noqa: E402  Lab 5's calibrated lens: pixel -> latitude/longitude, the row of a base

DEFAULT_DEPTH_MODEL = HERE / "models" / "depth_anything_v2_small.onnx"
# Depth Anything V2 Small, exported to ONNX by the onnx-community project (relative depth, ~99 MB fp32;
# the quantized file, ~25 MB, is 'onnx/model_quantized.onnx' in the same repo and works too).
DEPTH_MODEL_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
DEPTH_INPUT = 252  # pixels, a multiple of 14 (the model's patch); 518 is the model's native size, 4x slower.
# A laptop does 252 in 40 ms. The Pi 5 with the robot stack running needs 168 (--depth-size): 252 took
# 1.9 s there, 168 about 0.8 s, and the floor rule only needs the row profile.
DETECT_PX = 320  # the detector runs on a copy this big: 280 -> 210 ms on the loaded Pi, same boxes

STRIPS = ["left", "center-left", "center", "center-right", "right"]
STRIP_EDGES_DEG = [-45.0, -15.0, 15.0, 45.0]  # the strips' borders, as bearings: centre = within 15 degrees of straight on
OUTER_DEG = 80.0  # the left and right strips stop here: the disc's rim shows the robot's own legs and body
HFOV_DEG = 180.0  # Pupper's fisheye: bearing is about proportional to the distance from the centre column
LOWER_BAND = 0.5  # the lower half of the frame is where the floor and low obstacles are
PINK_RATIO = 1.2  # a frame whose red mean is this much above its green mean gets the software white balance
NEAR_RATIO = 0.75  # relative mode: a row counts as "near" when its closeness is this fraction of the bottom row's
ROW_PERCENTILE = 90  # per row and strip, the closeness of the nearer things in it (a thin chair leg still counts)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================ the fisheye's image circle
def find_circle(image) -> tuple[float, float, float]:
    """Pupper's fisheye draws a bright disc in the middle of a black frame; the 180 degrees span the
    disc's diameter. Returns (cx, cy, r) as fractions of the width (cy of the height), found from
    where the bright pixels are. A frame with no black border gives (0.5, 0.5, 0.5)."""
    from PIL import Image

    g = np.asarray(image.convert("L").resize((128, 96), Image.BILINEAR), dtype=np.float32)
    bright = g > 24.0
    cols = np.where(bright.mean(axis=0) > 0.08)[0]
    rows = np.where(bright.mean(axis=1) > 0.08)[0]
    if len(cols) < 8 or len(rows) < 8:
        return 0.5, 0.5, 0.5
    x0, x1 = cols[0] / 128.0, (cols[-1] + 1) / 128.0
    y0, y1 = rows[0] / 96.0, (rows[-1] + 1) / 96.0
    r = (x1 - x0) / 2.0  # as a fraction of the width (the disc is clipped top and bottom in some frames)
    if r > 0.47:
        return 0.5, 0.5, 0.5
    return round(float(x0 + x1) / 2.0, 3), round(float(y0 + y1) / 2.0, 3), round(float(r), 3)


def white_balance(image, circle: tuple[float, float, float] | None = None):
    """Take the pink cast off a frame (the lens's UV layer leaves the camera's auto white balance
    out of range: Lab 5 fixed it at the source with fixed gains, see README). Gray-world on the
    pixels inside the disc: scale each channel so the disc averages neutral. Cheap, and good
    enough for Gemma to call a red mug red; the depth model doesn't care either way."""
    from PIL import Image

    thumb = np.asarray(image.convert("RGB").resize((96, 72), Image.BILINEAR), dtype=np.float32)  # the check is cheap
    h, w, _ = thumb.shape
    if circle is not None and circle[2] < 0.5:
        yy, xx = np.mgrid[0:h, 0:w]
        inside = ((xx - circle[0] * w) / (circle[2] * w)) ** 2 + ((yy - circle[1] * h) / (circle[2] * w)) ** 2 <= 0.9
    else:
        inside = np.ones((h, w), dtype=bool)
    means = thumb[inside].reshape(-1, 3).mean(axis=0) + 1e-3
    if means[0] < PINK_RATIO * means[1]:  # not pink (the camera node's fixed gains are on): leave the colors alone
        return image
    gains = np.clip(means.mean() / means, 0.5, 3.0)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    return Image.fromarray(np.clip(arr * gains, 0, 255).astype(np.uint8))


def crop_to_circle(image, circle: tuple[float, float, float]):
    """The square around the image circle, so that left to right is -90 to +90 degrees."""
    cx, cy, r = circle
    w, h = image.size
    side = 2 * r * w
    x0, y0 = cx * w - side / 2, cy * h - side / 2
    return image.crop((int(round(x0)), int(round(y0)), int(round(x0 + side)), int(round(y0 + side)))), (x0, y0, side)


def box_to_frame(box: list[float], geom: tuple[float, float, float], size: tuple[int, int]) -> list[float]:
    """A box in crop fractions -> full-frame fractions (the robot and the detector use the full frame)."""
    x0, y0, side = geom
    w, h = size
    return [round((box[0] * side + x0) / w, 3), round((box[1] * side + y0) / h, 3), round(box[2] * side / w, 3), round(box[3] * side / h, 3)]


def circle_mask(n: int) -> np.ndarray:
    """Which pixels of an n x n crop are inside the disc (the corners are the black border)."""
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    return (xx - c) ** 2 + (yy - c) ** 2 <= (0.98 * n / 2.0) ** 2


# ============================================================ depth -> free space
def strip_edges(width: int, hfov_deg: float = HFOV_DEG) -> list[int]:
    """Pixel columns where the strips begin and end, from the bearings in STRIP_EDGES_DEG."""
    degs = [-OUTER_DEG, *STRIP_EDGES_DEG, OUTER_DEG]
    return [int(round(width * (0.5 + d / hfov_deg))) for d in degs]


# Rows of the disc crop searched: from the horizon down to 45 degrees below it. Lower than that the floor
# itself flattens in closeness (the ray to the floor at elevation e is height / sin(e), which changes
# little past 45 degrees) and would pass as a surface; a thing whose base is lower than 45 degrees is
# closer than the camera's height and shows as a run that reaches the band's bottom.
BAND_TOP, BAND_BOTTOM = 0.50, 0.75
FLAT_RATIO = 0.88  # a row whose closeness is at least this fraction of the row RUN_STEP below it is part of a vertical surface
RUN_STEP = 0.06  # of the crop height (about 11 degrees of elevation)
MIN_RUN = 0.02  # of the crop height: shorter flat runs are noise
FAR_FRACTION = 0.1  # rows whose closeness (above the map's floor value) is under this fraction of the band's max are "far", never a surface
CAM_HEIGHT_STANDING_M = 0.22  # the camera's height above the floor; the robot sends the one for its state
CAM_HEIGHT_LYING_M = 0.12


def nearest_surface_deg(rows: np.ndarray, offset: float = 0.0) -> float | None:
    """The elevation (degrees below the horizon) of the base of the nearest vertical surface in one
    strip, from its per-row closeness profile (row 0 = top of the disc crop, row n-1 = bottom).

    On an empty floor the closeness falls steadily going up the image (the floor recedes). Something
    standing on the floor is a run of rows with about the same closeness (a vertical surface) sitting
    on that curve; its base is where the run ends and the floor's fall resumes. The base's elevation
    gives its distance: floor at elevation e is cam_height / tan(e) away, whatever the depth model's
    scale, which is why this needs no metric depth and no per-frame calibration."""
    n = len(rows)
    r = rows.copy() - offset  # the depth model's relative scale has an offset: take it off, so far things read ~0
    bad = np.isnan(r)
    if bad.all():
        return None
    r[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), r[~bad])
    k = max(3, int(round(n * 0.02)))  # smooth over about 2 % of the height
    r = np.convolve(r, np.ones(k) / k, mode="same")
    step = max(2, int(round(n * RUN_STEP)))
    top, bottom = int(n * BAND_TOP), int(n * BAND_BOTTOM)
    far = FAR_FRACTION * max(float(np.max(r[top:bottom])), 1e-6)
    min_run = max(2, int(round(n * MIN_RUN)))
    base_row = None  # of the LOWEST run: the nearest surface (a far wall above a near chair leg must not hide the leg)
    run = 0
    for y in range(top, bottom - step):  # from the horizon down
        below = r[y + step]
        flat = r[y] > far and r[y] >= FLAT_RATIO * below
        if flat:
            run += 1
        else:
            if run >= min_run:
                base_row = y - 1 + step  # the run was measured against rows `step` below: the base is there
            run = 0
    if run >= min_run:  # a surface that reaches the bottom of the band: closer than the camera is high
        base_row = bottom - 1
    if base_row is None:
        return None
    return (base_row / n - 0.5) * 180.0  # equidistant fisheye: elevation is linear in the row


def row_percentile(col: np.ndarray, pct: float, min_valid: int) -> np.ndarray:
    """The pct-th percentile of each row's valid (non-NaN) values; NaN for rows with fewer than
    min_valid of them. np.sort puts NaN last, so the k-th sorted value of each row is its
    percentile over the valid values alone: ten times faster than np.nanpercentile."""
    n_valid = np.sum(~np.isnan(col), axis=1)
    srt = np.sort(col, axis=1)
    k = np.clip(np.round(pct / 100.0 * (n_valid - 1)).astype(int), 0, col.shape[1] - 1)
    out = srt[np.arange(col.shape[0]), k]
    out[n_valid < min_valid] = np.nan
    return out


def freespace_from_closeness(closeness: np.ndarray, metric_depth: np.ndarray | None = None,
                             hfov_deg: float = HFOV_DEG, mask: np.ndarray | None = None,
                             cam_height_m: float = CAM_HEIGHT_STANDING_M,
                             geom: tuple[float, float, float] | None = None, size: tuple[int, int] | None = None) -> list[dict]:
    """Summarize a closeness map (bigger = closer; any scale) of the disc crop into the five strips:
    for each, base_deg (the elevation of the nearest vertical surface's base, or None: floor only)
    and closest_m (its distance for a camera cam_height_m above the floor). ``mask``: which pixels
    are real (inside the fisheye's disc). A metric depth map, when there is one, is not needed."""
    h, w = closeness.shape
    valid = np.ones_like(closeness, dtype=bool) if mask is None else mask
    cl = np.where(valid, closeness, np.nan)
    offset = float(np.nanpercentile(cl, 2))  # the farthest things in view: the model's zero
    edges = strip_edges(w, hfov_deg)
    out = []
    for name, x0, x1 in zip(STRIPS, edges[:-1], edges[1:]):
        col = cl[:, x0:x1]
        rows = row_percentile(col, ROW_PERCENTILE, max(3, (x1 - x0) // 4))  # one number per row: its nearer things
        deg = nearest_surface_deg(rows, offset) if np.sum(~np.isnan(rows)) >= 8 else None
        st = {"name": name, "base_deg": None if deg is None else round(deg, 1), "closest_m": None, "nearness": None}
        if deg is not None and deg > 0.5:
            st["closest_m"] = round(min(9.99, cam_height_m / np.tan(np.radians(deg))), 2)
        if deg is not None and geom is not None and size is not None:
            # Lab 5's distance sensor: the base's row in the undistorted image, through the calibrated lens.
            # The base row in the crop -> the raw frame pixel at the strip's centre column -> its latitude.
            x0c, y0c, side = geom
            u = (x0c + (x0 + x1) / 2.0 / w * side) / size[0]
            v = (y0c + (deg / 180.0 + 0.5) * side) / size[1]
            n = fisheye_geom.nearness(u, v, size[0], size[1])
            st["nearness"] = None if n is None else round(n, 3)
        out.append(st)
    return out


class Depth:
    """Depth Anything V2 (relative) through onnxruntime. metric=True if the file is a metric model."""

    def __init__(self, model: Path, metric: bool = False, threads: int = 4, size: int = DEPTH_INPUT):
        import onnxruntime as ort  # here: the rest of the file works without it

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.metric = metric
        self.size = size - size % 14  # the model wants a multiple of its 14 px patch

    def closeness(self, image) -> tuple[np.ndarray, np.ndarray | None]:
        """(closeness map, metric depth map or None) at DEPTH_INPUT resolution."""
        from PIL import Image

        im = image.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        x = (np.asarray(im, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = x.transpose(2, 0, 1)[None]
        y = self.session.run(None, {self.input_name: x})[0]
        y = np.squeeze(y).astype(np.float32)
        if self.metric:  # metres: closeness is 1/depth
            depth = np.maximum(y, 0.05)
            return 1.0 / depth, depth
        return np.maximum(y, 0.0), None  # relative inverse depth: bigger = closer


# ============================================================ the COCO detector (MediaPipe)
DEFAULT_DETECTOR_MODEL = HERE / "models" / "efficientdet_lite0.tflite"
# The same 80-class EfficientDet-Lite0 the web app runs in the browser (services/objectDetector.ts).
DETECTOR_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
# What people call things -> the detector's class names (the rest is matched by name).
COCO_SYNONYMS = {
    "phone": "cell phone", "mobile": "cell phone", "mug": "cup", "glass": "cup", "water bottle": "bottle",
    "ball": "sports ball", "computer": "laptop", "bag": "backpack", "purse": "handbag", "controller": "remote",
    "television": "tv", "monitor": "tv", "screen": "tv", "someone": "person", "somebody": "person", "people": "person",
    "human": "person", "man": "person", "woman": "person", "kid": "person", "seat": "chair", "sofa": "couch",
    "plant": "potted plant", "bike": "bicycle", "puppy": "dog", "kitty": "cat", "notebook": "book",
}


class Detector:
    """EfficientDet-Lite0 through MediaPipe: boxes with real sizes for 80 everyday things (person,
    chair, backpack, cup, bottle, laptop, sports ball ...), in about 50 ms on a laptop. For go_to
    this beats Gemma's grounding whenever the thing has a COCO class: a box every frame, not a point
    every 2-3 s, and the box's bottom edge is where the thing really meets the floor."""

    def __init__(self, model: Path, score: float = 0.3):  # 0.3, not the browser's 0.4: the fisheye lowers every score
        import mediapipe as mp  # here: the rest of the file works without it
        from mediapipe.tasks.python import vision as mpv

        self.mp = mp
        opts = mpv.ObjectDetectorOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
                                         running_mode=mpv.RunningMode.IMAGE, score_threshold=score, max_results=12)
        self.detector = mpv.ObjectDetector.create_from_options(opts)
        self.names: set[str] = set()

    def detect(self, image) -> list[dict]:
        """[{label, score, box: [x, y, w, h] as fractions of the image}] for one PIL image."""
        rgb = np.asarray(image.convert("RGB"))
        res = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        h, w = rgb.shape[:2]
        out = []
        for d in res.detections:
            cat = d.categories[0]
            bb = d.bounding_box
            out.append({"label": cat.category_name.lower(), "score": round(cat.score, 2),
                        "box": [bb.origin_x / w, bb.origin_y / h, bb.width / w, bb.height / h]})
        return out

    @staticmethod
    def wanted_label(name: str) -> str | None:
        """The detector's class for what a person calls a thing, or None if it has none."""
        n = " ".join(name.lower().split())
        for lead in ("the ", "a ", "an ", "my ", "that ", "this ", "nearest ", "closest "):
            if n.startswith(lead):
                n = n[len(lead):]
        n = COCO_SYNONYMS.get(n, n)
        return n if n in COCO_CLASSES else None

    @staticmethod
    def matches(label: str, wanted_label: str) -> bool:
        """Does a detection's class count as the wanted class? The fisheye makes the detector hedge
        between look-alikes (a backpack on the floor comes out as 'suitcase'), so each class accepts
        its look-alikes too."""
        return label == wanted_label or label in COCO_LOOKALIKES.get(wanted_label, set())


COCO_LOOKALIKES = {
    "backpack": {"suitcase", "handbag"}, "handbag": {"backpack", "suitcase"}, "suitcase": {"backpack", "handbag"},
    "chair": {"couch", "bench"}, "couch": {"chair", "bed"}, "cup": {"bowl", "wine glass", "bottle"},
    "bottle": {"cup", "vase"}, "sports ball": {"apple", "orange", "frisbee"}, "laptop": {"tv", "keyboard"},
    "tv": {"laptop"}, "cell phone": {"remote"}, "remote": {"cell phone"}, "dog": {"cat", "teddy bear"},
    "cat": {"dog", "teddy bear"}, "book": {"laptop"}, "dining table": {"bench"},
}

COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}


def fetch_detector_model(path: Path = DEFAULT_DETECTOR_MODEL, url: str = DETECTOR_MODEL_URL) -> Path:
    """Download the detector model once (about 4 MB) into vision/models/."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {path} ...", file=sys.stderr)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    return path


# ============================================================ Gemma through Ollama
FRAME = ("This is one frame from a small robot dog's fisheye camera, about 20 cm above the floor, looking "
         "forward: a circular 180-degree view, so straight lines curve, the floor right in front of the robot is "
         "at the bottom of the circle, and things at the left and right edges are beside the robot, not ahead.")


class Gemma:
    def __init__(self, url: str, model: str, timeout: float = 60.0):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def ask(self, image_jpeg: bytes, prompt: str, json_mode: bool = False, max_tokens: int = 120) -> str:
        body = {"model": self.model, "prompt": prompt, "images": [base64.b64encode(image_jpeg).decode()],
                "stream": False, "options": {"temperature": 0.1, "num_predict": max_tokens}}
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(self.url + "/api/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode()).get("response", "").strip()

    def exists(self, image_jpeg: bytes, name: str) -> bool:
        """Is ``name`` clearly in the frame? Asked before locate(): asked for a box, a small model draws
        one for almost anything ('a cup of coffee' on a breadboard); asked yes or no, it is far more honest."""
        prompt = (f"Is there a {name} clearly visible in this image? Answer with one word, yes or no.")
        try:
            return self.ask(image_jpeg, prompt, max_tokens=3).strip().lower().startswith("y")
        except (urllib.error.URLError, OSError, ValueError) as e:
            log.warning("exists %r failed: %s", name, e)
            return False

    def locate(self, image_jpeg: bytes, name: str) -> tuple[list[float] | None, float, bool]:
        """Where ``name`` is in the frame: ([x, y, w, h] as fractions, confidence, bottom_known).

        Lab 5's rule: the bottom edge of a thing's box, where it meets the floor, is its distance
        (lower in the frame = nearer), so the box is asked for with its bottom spelled out. A box
        whose size is implausibly small (a model that only pointed) is kept for the bearing but
        marked bottom_known=False: the robot steers on it and never calls "arrived" from it."""
        prompt = (f"Where is the {name} in this image? Answer with JSON only: "
                  '{"found": true or false, "x": 0-1000, "bottom": 0-1000, "width": 0-1000, "height": 0-1000}: '
                  "x is the centre of it from the left edge; bottom is where it meets the floor (its lowest "
                  "visible point) from the top; width and height are its size as fractions of the image. If "
                  "there are several, pick the nearest. If it is not there, found is false.")
        try:
            data = json.loads(self.ask(image_jpeg, prompt, json_mode=True, max_tokens=80))
            if not data.get("found"):
                return None, 0.0, False
            x = max(0.0, min(1000.0, float(data["x"]))) / 1000.0
            bottom = max(0.0, min(1000.0, float(data.get("bottom", data.get("y", 500))))) / 1000.0
            w = max(0.0, min(1000.0, float(data.get("width", 0)))) / 1000.0
            h = max(0.0, min(1000.0, float(data.get("height", 0)))) / 1000.0
            known = w >= 0.02 and h >= 0.02 and bottom > 0.0
            w, h = max(0.03, w), max(0.03, h)
            return [max(0.0, x - w / 2), max(0.0, bottom - h), w, h], 0.8, known
        except (ValueError, TypeError, KeyError, urllib.error.URLError, OSError) as e:
            log.warning("locate %r failed: %s", name, e)
            return None, 0.0, False

    def describe(self, image_jpeg: bytes) -> str:
        prompt = (FRAME + " "
                  "Describe only what is really in the picture, in two short plain sentences, no preamble: first "
                  "the kind of place, then anything on the floor or low down that a 20 cm tall robot could bump "
                  "into, with left / ahead / right and a rough distance in metres. If the floor ahead looks clear, "
                  "say so. Never guess at things you cannot see.")
        return self.ask(image_jpeg, prompt, max_tokens=110)

    def answer(self, image_jpeg: bytes, question: str) -> str:
        prompt = (FRAME + " Answer "
                  "the question from what is really in the picture, in one or two short plain sentences, no preamble, "
                  "with left / ahead / right and rough distances in metres when they help. If the picture does not "
                  f"show the answer, say that. Question: {question}")
        return self.ask(image_jpeg, prompt, max_tokens=100)


class GemmaAPI(Gemma):
    """The eyes' model served by Google's API with the Gemini key, instead of a local Gemma in Ollama.

    For when nothing local can run it: the Pi cannot, and the Jetson is not here yet. This is cloud,
    which the hackathon's on-device rule forbids for the final build; it is a stand-in for the Jetson
    running Gemma, and the rest of the robot does not know the difference. Any vision model the API
    serves works here. Measured from the Pi on a real frame: Gemma 4 (gemma-4-26b-a4b-it) thinks
    before every answer, cannot be told not to, and takes 15-20 s; gemini-3.5-flash-lite answers
    "is there a backpack" in 1 s and places it in 1 s, so it is the default. The key is read from a
    file and never logged."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, key: str, model: str = "gemini-3.5-flash-lite", timeout: float = 40.0):
        self.key, self.model, self.timeout = key.strip(), model, timeout
        self.url = self.URL.format(model=model)

    def ask(self, image_jpeg: bytes, prompt: str, json_mode: bool = False, max_tokens: int = 120) -> str:
        if json_mode:
            prompt += " Reply with the JSON only, no code fences, no words around it."
        # A thinking model (Gemma 4 on the API thinks and cannot be told not to) spends its output
        # budget on thoughts first: a 3-token budget for "yes or no" came back empty. So the budget
        # is generous and the thought parts are skipped when reading the answer.
        body = {"contents": [{"parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_jpeg).decode()}},
                    {"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 600 + max_tokens}}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        data = None
        for attempt in range(2):  # the API answers 500 now and then: once more
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                if e.code < 500 or attempt:
                    raise
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"no text in the API reply: {str(data)[:200]}")
        text = text.strip()
        if text.startswith("```"):  # a fenced block despite the ask
            text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        return text


# ============================================================ the robot server
class RobotLink:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.aspect = 4.0 / 3.0  # frame width / height, so the disc's radius can be given in height fractions too

    def _json(self, method: str, path: str, payload: dict | None = None, timeout: float = 3.0) -> dict | None:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, OSError, ValueError) as e:
            log.warning("%s %s failed: %s", method, path, e)
            return None

    def frame(self) -> bytes | None:
        try:
            with urllib.request.urlopen(self.url + "/camera/image", timeout=3.0) as r:
                return r.read() if r.status == 200 else None
        except (urllib.error.URLError, OSError):
            return None

    def status(self, source: str) -> dict:
        """The robot's cheap answer to "should I look now?": attention and others_posting."""
        return self._json("GET", "/vision?source=" + urllib.parse.quote(source)) or {"attention": True}

    def post_freespace(self, strips: list[dict], mode: str, eyes_url: str, circle: tuple[float, float, float],
                       frame_age_s: float = 0.0, source: str = "") -> dict:
        """Posts the strips with the age of the frame they came from (a slow poster must not make
        old data look fresh); the reply carries what go_to wants found and the camera's height in
        the robot's current state (lying or standing), saving a request."""
        cx, cy, r = circle
        return self._json("POST", "/vision/freespace", {"strips": strips, "mode": mode, "eyes_url": eyes_url,
                                                        "circle": [cx, cy, r, r * self.aspect],
                                                        "frame_age_s": round(frame_age_s, 2), "source": source}) or {}

    def wanted(self) -> str | None:
        return ((self._json("GET", "/vision/wanted") or {}).get("name")) or None

    def post_detections(self, dets: list[dict], size: tuple[int, int]) -> None:
        """The detector's boxes, on the endpoint the browser's detector uses (lab/hackathon_vision.py):
        find's looks, what_is_around and the named obstacle rule then work without the web app."""
        self._json("POST", "/vision/detections", {"detections": [{"label": d["label"], "score": d["score"],
                                                                  "box": [round(v, 3) for v in d["box"]]} for d in dets],
                                                  "width": size[0], "height": size[1]})

    def post_target(self, name: str, box: list | None, score: float, source: str = "gemma", bottom_known: bool = True) -> None:
        self._json("POST", "/vision/target", {"name": name, "box": [round(v, 3) for v in box] if box else None,
                                              "score": score, "source": source, "bottom_known": bottom_known})

    def post_scene(self, text: str) -> None:
        self._json("POST", "/vision/scene", {"text": text})


# ============================================================ the service
DETECTOR_PATIENCE_S = 3.0  # the detector gets this long to find a thing with a class before Gemma is asked too
# Attention gating: the robot's replies say whether the eyes matter right now (a motion running or
# queued, follow mode on, a tool called in the last 10 s) and whether another eyes service is
# posting. Full rate when they matter; IDLE_HZ otherwise; asleep while another service is on.
# The cheap status poll (GET /vision) runs every POLL_S whatever the state, so a sleeping or idle
# service wakes within half a second of a tool call, well before the legs move.
IDLE_HZ = 0.2
POLL_S = 0.5
STANDBY_S = 2.0


def next_period(status: dict, hz: float) -> float | None:
    """How long until the next frame given the robot's status: None = stand down (another eyes
    service is posting), else the period for full rate or idle."""
    if status.get("others_posting"):
        return None
    return 1.0 / hz if status.get("attention") else 1.0 / IDLE_HZ


class Eyes:
    def __init__(self, robot: RobotLink | None, depth: Depth | None, gemma: Gemma | None, *, hz: float,
                 describe_every: float, advertise: str, hfov_deg: float = HFOV_DEG, wb: bool = True,
                 detector: "Detector | None" = None, source: str = "", always: bool = False):
        self.robot, self.depth, self.gemma, self.detector = robot, depth, gemma, detector
        self.source = source or socket.gethostname()
        self.always = always  # no gating: full rate whatever the robot is doing (for testing)
        self.state = "full"  # full | idle | standby, for /health
        self.detections: list[dict] = []  # the detector's latest boxes, in full-frame fractions
        self.detector_hit = 0.0  # when the detector last had a box for the wanted thing (else Gemma takes over)
        self.hz, self.describe_every, self.advertise, self.hfov_deg, self.wb = hz, describe_every, advertise, hfov_deg, wb
        self.latest: bytes | None = None  # the latest JPEG
        self.latest_time = 0.0
        self.lock = threading.Lock()
        self.wanted: str | None = None
        self.summary: list[dict] = []
        self.circle: tuple[float, float, float] = (0.5, 0.5, 0.5)  # the fisheye's disc, found from the frames
        self.mask = circle_mask(depth.size if depth else DEPTH_INPUT)
        self.cam_height_m = CAM_HEIGHT_STANDING_M  # the robot says which (standing or lying) with each reply
        self.stats = {"frames": 0, "depth_ms": 0.0, "gemma_ms": 0.0}

    def crop(self, jpeg: bytes):
        """The latest frame cropped to its disc: (PIL image, crop geometry, full size)."""
        from PIL import Image

        im = Image.open(io.BytesIO(jpeg))
        if self.robot:
            self.robot.aspect = im.size[0] / max(1, im.size[1])
        if self.stats["frames"] % 50 == 1:  # the lens doesn't move: look for the disc now and then
            self.circle = find_circle(im)
        if self.wb:
            im = white_balance(im, self.circle)
        crop, geom = crop_to_circle(im, self.circle)
        return crop, geom, im.size

    # ---- depth loop: a few times a second while it matters, idle otherwise, asleep while another eyes service is on
    def depth_loop(self) -> None:
        from PIL import Image

        next_frame = 0.0
        while True:
            t0 = time.time()
            if not self.always and self.robot:
                status = self.robot.status(self.source)
                period = next_period(status, self.hz)
                self.state = "standby" if period is None else ("full" if status.get("attention") else "idle")
                if period is None:  # another eyes service is posting: nothing to add, keep cool
                    time.sleep(STANDBY_S)
                    continue
                if t0 < next_frame:  # idle: not yet time for the next frame; just keep polling
                    time.sleep(min(POLL_S, next_frame - t0))
                    continue
                next_frame = t0 + period
            else:
                next_frame = t0 + 1.0 / self.hz
            jpeg = self.robot.frame() if self.robot else None
            if jpeg:
                with self.lock:
                    self.latest, self.latest_time = jpeg, t0
                self.stats["frames"] += 1
                if self.detector:
                    try:
                        crop, geom, size = self.crop(jpeg)
                        dets = self.detector.detect(crop.resize((DETECT_PX, DETECT_PX)) if crop.size[0] > DETECT_PX else crop)
                        for d in dets:
                            d["box"] = box_to_frame(d["box"], geom, size)
                        self.detections = dets
                        self.robot.post_detections(dets, size)
                        label = Detector.wanted_label(self.wanted) if self.wanted else None
                        if label:  # the wanted thing has a class: the detector's box, every frame, nearest first
                            hits = [d for d in dets if Detector.matches(d["label"], label)]
                            best = max(hits, key=lambda d: d["box"][1] + d["box"][3]) if hits else None  # lowest = nearest
                            if best:
                                self.detector_hit = time.time()
                                self.robot.post_target(self.wanted, best["box"], best["score"], source="detector")
                            elif time.time() - self.detector_hit > DETECTOR_PATIENCE_S + 5:
                                self.robot.post_target(self.wanted, None, 0.0, source="detector")  # Gemma had its turn too
                    except Exception as e:
                        log.warning("detector failed: %s", e)
                if self.depth:
                    try:
                        crop, geom, size = self.crop(jpeg)
                        closeness, metric = self.depth.closeness(crop)
                        mask = self.mask if self.circle[2] < 0.5 else None
                        self.summary = freespace_from_closeness(closeness, metric, self.hfov_deg, mask, self.cam_height_m, geom, size)
                        self.stats["depth_ms"] = (time.time() - t0) * 1000
                        reply = self.robot.post_freespace(self.summary, "metric", self.advertise if self.gemma else "", self.circle,
                                                          frame_age_s=time.time() - t0, source=self.source)
                        self.wanted = reply.get("wanted") or None
                        try:
                            self.cam_height_m = float(reply.get("cam_height_m") or self.cam_height_m)
                        except (TypeError, ValueError):
                            pass
                    except Exception as e:  # a bad frame must not kill the eyes
                        log.warning("depth failed: %s", e)
                else:
                    self.wanted = self.robot.wanted()
            if self.always or not self.robot:
                time.sleep(max(0.0, next_frame - time.time()))

    # ---- Gemma loop: track the wanted thing, else describe now and then
    def gemma_loop(self) -> None:
        last_described = 0.0
        confirmed: str | None = None  # the wanted thing exists() said yes to
        previous: list[float] | None = None  # the last box, to check the next one agrees with it
        while True:
            with self.lock:
                jpeg = self.latest
            if not jpeg or not self.gemma or not self.robot or self.state == "standby":
                time.sleep(0.5)
                continue
            t0 = time.time()
            wanted = self.wanted
            crop, geom, size = self.crop(jpeg)
            buf = io.BytesIO()
            crop.save(buf, "JPEG", quality=85)
            jpeg = buf.getvalue()  # Gemma sees the disc, not the black border
            if wanted and self.detector and Detector.wanted_label(wanted) and time.time() - self.detector_hit <= DETECTOR_PATIENCE_S:
                time.sleep(0.2)  # the detector has this one (every frame, with a real box)
                continue
            if wanted:
                if confirmed != wanted:  # a new target: first ask whether it is there at all
                    previous = None
                    if not self.gemma.exists(jpeg, wanted):
                        self.robot.post_target(wanted, None, 0.0)
                        log.info("target %r: not in the frame (%.0f ms)", wanted, (time.time() - t0) * 1000)
                        self.stats["gemma_ms"] = (time.time() - t0) * 1000
                        continue
                    confirmed = wanted
                box, score, known = self.gemma.locate(jpeg, wanted)
                if box is not None:
                    box = box_to_frame(box, geom, size)
                # Two fixes in a row must roughly agree before the robot steers on them (a stray box
                # moves the robot; a missed one only delays it).
                agree = box is not None and (previous is None or (
                    abs(box[0] + box[2] / 2 - previous[0] - previous[2] / 2) <= 0.2
                    and abs(box[1] + box[3] / 2 - previous[1] - previous[3] / 2) <= 0.2))
                self.robot.post_target(wanted, box if agree and previous is not None else None, score if agree else 0.0,
                                       bottom_known=known)
                log.info("target %r: %s (%.0f ms)", wanted, "box %s" % [round(v, 2) for v in box] if box else "not seen",
                         (time.time() - t0) * 1000)
                previous = box
            elif time.time() - last_described >= self.describe_every:
                confirmed, previous = None, None
                text = self.gemma.describe(jpeg)
                if text:
                    self.robot.post_scene(text)
                    log.info("scene (%.0f ms): %s", (time.time() - t0) * 1000, text)
                last_described = time.time()
            else:
                time.sleep(0.2)
                continue
            self.stats["gemma_ms"] = (time.time() - t0) * 1000

    # ---- /exists: is a thing in the frame? (explore asks this at every stop)
    def exists(self, name: str) -> dict:
        with self.lock:
            jpeg, age = self.latest, time.time() - self.latest_time
        if not jpeg:
            return {"found": False, "error": "no camera frame yet"}
        t0 = time.time()
        label = Detector.wanted_label(name) if self.detector else None
        if label:  # a thing with a class: the detector's word, this frame
            hits = [d for d in self.detections if Detector.matches(d["label"], label)]
            if hits:
                best = max(hits, key=lambda d: d["score"])
                return {"found": True, "box": best["box"], "source": "detector", "frame_age_s": round(age, 1)}
        if not self.gemma:
            return {"found": False, "source": "detector", "frame_age_s": round(age, 1)}
        crop, _, _ = self.crop(jpeg)
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=85)
        found = self.gemma.exists(buf.getvalue(), name)
        return {"found": found, "source": "gemma", "frame_age_s": round(age, 1), "ms": round((time.time() - t0) * 1000)}

    # ---- /look
    def look(self, question: str) -> dict:
        with self.lock:
            jpeg, age = self.latest, time.time() - self.latest_time
        if not jpeg:
            return {"answer": "", "error": "no camera frame yet"}
        if not self.gemma:
            return {"answer": "", "error": "Gemma is off (--no-gemma, or Ollama isn't running)"}
        t0 = time.time()
        crop, _, _ = self.crop(jpeg)
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=85)
        answer = self.gemma.answer(buf.getvalue(), question)
        return {"answer": answer, "frame_age_s": round(age, 1), "ms": round((time.time() - t0) * 1000)}

    def serve(self, port: int) -> ThreadingHTTPServer:
        eyes = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, code: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/health"):
                    self._send(200, {"ok": True, "depth": eyes.depth is not None, "gemma": eyes.gemma is not None,
                                     "detector": eyes.detector is not None, "detections": eyes.detections,
                                     "state": eyes.state, "source": eyes.source,
                                     "wanted": eyes.wanted, "freespace": eyes.summary, **eyes.stats})
                else:
                    self._send(404, {"error": "GET /health, POST /look or POST /exists"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    payload = json.loads(self.rfile.read(n).decode() or "{}")
                except ValueError:
                    return self._send(400, {"error": "bad json"})
                if self.path.startswith("/look"):
                    try:
                        return self._send(200, eyes.look(str(payload.get("question", ""))))
                    except Exception as e:
                        return self._send(500, {"answer": "", "error": f"{type(e).__name__}: {e}"})
                if self.path.startswith("/exists"):
                    try:
                        return self._send(200, eyes.exists(str(payload.get("name", ""))))
                    except Exception as e:
                        return self._send(500, {"found": False, "error": f"{type(e).__name__}: {e}"})
                self._send(404, {"error": "POST /look or POST /exists"})

        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server


def fetch_depth_model(path: Path = DEFAULT_DEPTH_MODEL, url: str = DEPTH_MODEL_URL) -> Path:
    """Download the depth model once (about 99 MB) into vision/models/."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {path} ...", file=sys.stderr)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    return path


def ollama_has(url: str, model: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=3.0) as r:
            names = [m.get("name", "") for m in json.loads(r.read().decode()).get("models", [])]
        return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="http://localhost:8000", help="the robot server (through the SSH tunnel)")
    ap.add_argument("--image", help="one image instead of the robot: print the depth summary and exit")
    ap.add_argument("--look", help="with --image: a question for Gemma about it")
    ap.add_argument("--hz", type=float, default=5.0, help="depth frames per second while the robot needs eyes (idle: one every 5 s)")
    ap.add_argument("--always", action="store_true", help="full rate always: no idling, no standing down for another eyes service")
    ap.add_argument("--source", default="", help="this service's name in the robot's bookkeeping (default: the hostname)")
    ap.add_argument("--port", type=int, default=8100, help="where /look is served")
    ap.add_argument("--advertise", default="http://127.0.0.1:8100",
                    help="the url the robot uses to reach /look (127.0.0.1 works on the Pi, or through ssh -R 8100:localhost:8100)")
    ap.add_argument("--depth-model", type=Path, default=DEFAULT_DEPTH_MODEL)
    ap.add_argument("--fetch-depth-model", action="store_true", help="download the depth model if it is missing")
    ap.add_argument("--metric", action="store_true", help="the depth model outputs metres (a metric-indoor export)")
    ap.add_argument("--depth-size", type=int, default=DEPTH_INPUT, help="the depth model's input in pixels (a multiple of 14; the Pi: 168)")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--detector-model", type=Path, default=DEFAULT_DETECTOR_MODEL)
    ap.add_argument("--fetch-detector-model", action="store_true", help="download the COCO detector model if it is missing (about 4 MB)")
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--gemma", default="gemma3:4b", help="the Ollama vision model (gemma3:4b, gemma3:12b, gemma3n:e4b)")
    ap.add_argument("--no-gemma", action="store_true")
    ap.add_argument("--gemma-api", action="store_true", help="the eyes' model through Google's API (cloud) instead of a local Gemma")
    ap.add_argument("--gemma-api-model", default="gemini-3.5-flash-lite",
                    help="any vision model the API serves: gemini-3.5-flash-lite (1 s), gemini-3.6-flash (2-3 s), gemma-4-26b-a4b-it (15-20 s: it thinks)")
    ap.add_argument("--gemma-api-key-file", type=Path, default=Path.home() / ".config" / "gemini-pupper" / "api_key",
                    help="the Gemini key (never put it in the repo); $GEMINI_API_KEY also works")
    ap.add_argument("--describe-every", type=float, default=15.0, help="seconds between scene descriptions when idle")
    ap.add_argument("--hfov", type=float, default=HFOV_DEG, help="the camera's horizontal field of view in degrees (fisheye: 180)")
    ap.add_argument("--cam-height", type=float, default=CAM_HEIGHT_STANDING_M, help="with --image: the camera's height above the floor in metres")
    ap.add_argument("--no-white-balance", action="store_true", help="keep the frames' colors as they come (default: gray-world fix for the pink cast)")
    ap.add_argument("--save-frames", type=Path, help="with --image: write the white-balanced disc crop here (to look at it)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    depth = None
    if not a.no_depth:
        if a.fetch_depth_model:
            fetch_depth_model(a.depth_model)
        if a.depth_model.exists():
            depth = Depth(a.depth_model, metric=a.metric, threads=a.threads, size=a.depth_size)
            log.info("depth model %s (%s, %d px)", a.depth_model.name, "metric" if a.metric else "relative", depth.size)
        else:
            log.warning("no depth model at %s: run with --fetch-depth-model (about 99 MB); depth is off", a.depth_model)
    detector = None
    if not a.no_detector:
        if a.fetch_detector_model:
            fetch_detector_model(a.detector_model)
        if a.detector_model.exists():
            try:
                detector = Detector(a.detector_model)
                log.info("COCO detector %s", a.detector_model.name)
            except Exception as e:
                log.warning("COCO detector unavailable (%s: pip install mediapipe); go_to relies on Gemma for everything", e)
        else:
            log.warning("no detector model at %s: run with --fetch-detector-model (about 4 MB); go_to relies on Gemma", a.detector_model)
    gemma = None
    if a.gemma_api and not a.no_gemma:
        key = os.environ.get("GEMINI_API_KEY", "") or (a.gemma_api_key_file.read_text().strip() if a.gemma_api_key_file.exists() else "")
        if key:
            gemma = GemmaAPI(key, a.gemma_api_model)
            log.info("eyes' model: %s through Google's API (cloud: a stand-in for Gemma on the Jetson)", a.gemma_api_model)
        else:
            log.warning("--gemma-api but no key in $GEMINI_API_KEY or %s; Gemma is off", a.gemma_api_key_file)
    elif not a.no_gemma:
        if ollama_has(a.ollama, a.gemma):
            gemma = Gemma(a.ollama, a.gemma)
            log.info("Gemma: %s at %s", a.gemma, a.ollama)
        else:
            log.warning("Ollama at %s has no %s (ollama pull %s); Gemma is off", a.ollama, a.gemma, a.gemma)

    if a.image:
        from PIL import Image

        jpeg = Path(a.image).read_bytes()
        im = Image.open(io.BytesIO(jpeg))
        circle = find_circle(im)
        if not a.no_white_balance:
            im = white_balance(im, circle)
        crop, geom = crop_to_circle(im, circle)
        if a.save_frames:
            crop.save(a.save_frames)
        print(f"  image circle: centre ({circle[0]:.2f}, {circle[1]:.2f}), radius {circle[2]:.2f} of the width"
              + ("" if circle[2] < 0.5 else " (no black border: the whole frame)"))
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=85)
        jpeg = buf.getvalue()
        if detector:
            t0 = time.time()
            dets = detector.detect(crop.resize((DETECT_PX, DETECT_PX)) if crop.size[0] > DETECT_PX else crop)
            print(f"  detector ({(time.time() - t0) * 1000:.0f} ms): " + ("; ".join(
                f"{d['label']} {d['score']:.2f} at {fisheye_geom.box_bearing_deg(box_to_frame(d['box'], geom, im.size)):+.0f} deg,"
                f" row {fisheye_geom.box_nearness(box_to_frame(d['box'], geom, im.size)) or 0:.2f}" for d in dets) or "nothing"))
        if depth:
            t0 = time.time()
            closeness, metric = depth.closeness(crop)
            ms = (time.time() - t0) * 1000
            for st in freespace_from_closeness(closeness, metric, a.hfov, circle_mask(depth.size) if circle[2] < 0.5 else None,
                                               a.cam_height, geom, im.size):
                print(f"  {st['name']:>13}: " + ("floor only" if st["base_deg"] is None else
                      f"surface base {st['base_deg']:5.1f} deg down the disc -> Lab 5 row {st['nearness']:.3f}"
                      f" ({'blocked' if st['nearness'] is not None and st['nearness'] >= 0.68 else 'clear'})"))
            print(f"  ({ms:.0f} ms; blocked when the base's Lab 5 row >= 0.68 of the image; the stop row is 0.794: lab/fisheye_geom.py)")
        if gemma and a.look:
            print("  Gemma:", gemma.answer(jpeg, a.look))
        elif gemma:
            print("  Gemma:", gemma.describe(jpeg))
        return

    eyes = Eyes(RobotLink(a.robot), depth, gemma, hz=a.hz, describe_every=a.describe_every, advertise=a.advertise,
                hfov_deg=a.hfov, wb=not a.no_white_balance, detector=detector, source=a.source, always=a.always)
    eyes.serve(a.port)
    log.info("serving /look on port %d (robot reaches it at %s); frames from %s", a.port, a.advertise, a.robot)
    threading.Thread(target=eyes.gemma_loop, daemon=True).start()
    try:
        eyes.depth_loop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
