"""Hackathon, part D: seeing things. Don't walk into them; find them when asked.

Pupper's eyes for this are an on-device object detector in the web app
(services/objectDetector.ts, MediaPipe EfficientDet-Lite0 on the camera frames):
the browser posts what it sees to POST /vision/detections about twice a second, and
the robot keeps the latest here (VISION). Each detection is {label, score, box}
with box = [x, y, w, h] as fractions of the frame, x to the right, y down.

Two behaviors use it:

  path_blocked()  the hardcoded rule: when something big and close fills the view in
                  the direction of a move, the move is refused, or stopped early if
                  it appears mid-walk. walk() in part2_walk.py calls it (the lab's
                  move, turn and arc all go through walk). Gemini also sees the camera
                  and has its own rules in prompt.md; this is the one that is enforced.
  seen_now()      is a named thing in view: the detector's word, else the eyes service's
                  Gemma. The find tool (hackathon_explore.py) asks it at every stop.

A second sense, class-agnostic: the eyes service
(vision/eyes.py, on a laptop or the Pi) runs a monocular depth model on the same camera
frames and posts a free-space summary to POST /vision/freespace about 5 times a second:
for five vertical strips of the lower half of the frame (left ... right), how close the
nearest surface is. path_blocked() checks it first, so a chair leg, a cable or a shoe
stops a move even though no detector has a class for it. The eyes service also posts a
tracked target for go_to (POST /vision/target, see hackathon_eyes.py) and a scene
description from Gemma (POST /vision/scene).

No detections, or stale ones (the app isn't running the detector), and no fresh free
space (the eyes service isn't running): the rules stay out of the way and Gemini's eyes
are what there is.
"""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.error

from hackathon_alerts import EVENTS
from part3_follow import prepare_to_move
from robot_api import FELL_MESSAGE, LLM_CMD_RATE, LOCOMOTION, LOCOMOTION_CONTROLLER, Result, with_note

log = logging.getLogger("vision")

FRESH_S = 2.0  # detections older than this are no eyes at all
MIN_SCORE = 0.4
VISION: dict = {"time": 0.0, "detections": [], "width": 0, "height": 0}

# Free space from the eyes service's depth model (class-agnostic). Five strips of the lower
# half of the frame, left to right, cut by bearing (the camera is a 180-degree fisheye, so
# the eyes service cuts them at -45, -15, +15 and +45 degrees, not at equal widths): "ahead"
# is the middle three, within 45 degrees of straight on, which is what a forward step can hit.
# Each strip: closest_m (metres, from a metric depth model; None with a relative one) and
# near_frac (0..1: how far up the lower band the nearest surface reaches; floor alone decays
# fast going up the image, an obstacle standing on it does not).
STRIPS = ["left", "center-left", "center", "center-right", "right"]
AHEAD = {"center-left", "center", "center-right"}
LEFT = {"left", "center-left"}
RIGHT = {"center-right", "right"}
# A depth summary older than this (counting from its frame, not its post) is not trusted. A laptop's
# eyes post at 5 Hz; the Pi's own, throttled by the stack's load and its heat, at about 0.5 Hz, which
# is a fallback, not a sense to walk on: stale means blind, never wrong.
FREESPACE_FRESH_S = 2.5
# "Metres" here are only as good as the camera-height constants below, which are not measured: what
# matters is that the robot never touches things, so the stop distance has margin. With these
# heights the robot read a backpack a metre away as 0.78 m, so real distances are a little larger
# than reported (the safe side); if a reading ever comes out larger than life, raise STOP_M, not the
# heights.
STOP_M = 0.45  # nothing closer than this (as reported) in the direction of a move (metric posts only)
CAM_HEIGHT_STANDING_M = 0.22  # the eyes service turns the elevation of a thing's base into a range with
CAM_HEIGHT_LYING_M = 0.12  # the height for the robot's state (floor at elevation e is height / tan(e) away)
# Lab 5's distance sensor instead of metres: the row of a thing's base in the undistorted
# equirectangular image, as a fraction of its height (lab/fisheye_geom.py: 0.5 is the horizon,
# STOP_NEARNESS 0.794 is where the Lab 5 follower stops at a person's feet). A surface whose base
# is at BLOCK_NEARNESS or lower in the image is in the way: a backpack at the robot's feet reads
# about 0.71, a cable hanging in front 0.74, chairs a metre off 0.54.
BLOCK_NEARNESS = 0.68
NEAR_FRAC_BLOCK = 0.45  # relative: the near surface fills this much of the lower band
# circle: the fisheye's image disc (centre x, centre y, radius as a fraction of the width, radius as a
# fraction of the height): the 180 degrees span the disc's diameter, so bearings and elevations are
# measured from it (hackathon_eyes.py).
FREESPACE: dict = {"time": 0.0, "strips": [], "mode": "", "eyes_url": "", "circle": [0.5, 0.5, 0.5, 0.5], "source": ""}

# Attention: the eyes service runs at full rate only while it matters (a motion is running or
# queued, follow mode is on, or a tool was called in the last ATTENTION_S), and idles otherwise.
# Every post's reply says so, and GET /vision too, so the eyes can check cheaply. Keeps the Pi
# cool when the robot is doing nothing, which is most of the time.
ATTENTION_S = 10.0
ATTENTION: dict = {"until": 0.0}
# Who is posting free space (several eyes services may run: the Pi's own, a laptop's, the Jetson's).
# A poster that sees another fresh source stands down, so the Pi's eyes sleep while a faster one is on.
POSTERS: dict[str, float] = {}
OTHER_FRESH_S = 2.0
SCENE: dict = {"time": 0.0, "text": ""}  # what Gemma last said is around (eyes service, POST /vision/scene)

# What people call things -> what the COCO detector calls them. Anything else is matched by name.
SYNONYMS = {
    "phone": "cell phone", "mobile": "cell phone", "iphone": "cell phone", "mug": "cup", "glass": "cup",
    "water bottle": "bottle", "ball": "sports ball", "tennis ball": "sports ball", "computer": "laptop",
    "bag": "backpack", "purse": "handbag", "controller": "remote", "tv remote": "remote", "television": "tv",
    "monitor": "tv", "screen": "tv", "someone": "person", "somebody": "person", "people": "person",
    "human": "person", "man": "person", "woman": "person", "kid": "person", "seat": "chair", "sofa": "couch",
    "plant": "potted plant", "bike": "bicycle", "puppy": "dog", "kitty": "cat", "notebook": "book",
}
NOT_DETECTABLE = {"keys", "key", "wallet", "glasses", "pen", "charger", "cable", "airpods", "earbuds", "ring"}
COCO_NAMES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "boat", "bench", "bird", "cat", "dog", "backpack",
    "umbrella", "handbag", "suitcase", "sports ball", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "orange", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
}
# Classes the detector hedges between (the fisheye distorts things): asked for one, its look-alikes count too.
LOOKALIKES = {
    "backpack": {"suitcase", "handbag"}, "handbag": {"backpack", "suitcase"}, "suitcase": {"backpack", "handbag"},
    "chair": {"couch", "bench"}, "couch": {"chair", "bed"}, "cup": {"bowl", "wine glass"}, "bottle": {"cup", "vase"},
    "sports ball": {"apple", "orange", "frisbee"}, "laptop": {"tv", "keyboard"}, "tv": {"laptop"},
    "cell phone": {"remote"}, "remote": {"cell phone"}, "dog": {"cat", "teddy bear"}, "cat": {"dog", "teddy bear"},
}


# ============================================================ what the robot does: the latest frame
def update_detections(detections: list, width: int = 0, height: int = 0) -> int:
    """The browser's latest detections. Bad entries are dropped, never a crash."""
    clean = []
    for d in detections or []:
        try:
            box = [max(0.0, min(1.0, float(v))) for v in d["box"]][:4]
            if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
                continue
            clean.append({"label": str(d["label"]).lower()[:40], "score": float(d.get("score", 1.0)), "box": box})
        except (KeyError, TypeError, ValueError):
            continue
    VISION.update(time=time.time(), detections=clean, width=int(width or 0), height=int(height or 0))
    return len(clean)


def wake(seconds: float = ATTENTION_S) -> None:
    """A tool was called: the eyes should look properly for a while."""
    ATTENTION["until"] = max(ATTENTION["until"], time.time() + seconds)


def attention_now(robot=None) -> bool:
    """Should the eyes run at full rate right now?"""
    if time.time() < ATTENTION["until"]:
        return True
    if robot is None:
        return False
    try:
        return bool(robot.following or robot.current is not None or not robot.commands.empty())
    except Exception:
        return False


EYES_ALIVE_S = 15.0  # an eyes service that posted within this is alive (idle, it posts every 5 s)


def eyes_alive() -> bool:
    """Is any eyes service posting at all (idle posts count)? The data may still be stale at this
    instant: that is the gating, not a fault, and the eyes wake within half a second of a tool call."""
    return any(time.time() - t <= EYES_ALIVE_S for t in POSTERS.values())


def others_fresh(source: str) -> bool:
    """Has a different eyes service posted free space within OTHER_FRESH_S?"""
    now = time.time()
    return any(src != source and now - t <= OTHER_FRESH_S for src, t in POSTERS.items())


def _wake_hook(name: str, args: dict, res) -> None:
    wake()


try:
    from tool_registry import CALL_HOOKS as _HOOKS

    _HOOKS.append(_wake_hook)
except ImportError:
    pass


def update_freespace(strips: list, mode: str = "relative", eyes_url: str = "", circle: list | None = None,
                     frame_age_s: float = 0.0, source: str = "") -> int:
    """The eyes service's latest free-space summary. Bad strips are dropped, never a crash."""
    clean = []
    for st in strips or []:
        try:
            name = str(st["name"])
            if name not in STRIPS:
                continue
            closest = st.get("closest_m")
            near = st.get("near_frac")
            nearness = st.get("nearness")
            clean.append({"name": name,
                          "closest_m": None if closest is None else max(0.0, float(closest)),
                          "near_frac": None if near is None else max(0.0, min(1.0, float(near))),
                          "nearness": None if nearness is None else max(0.0, min(1.0, float(nearness)))})
        except (KeyError, TypeError, ValueError):
            continue
    try:
        age = max(0.0, min(30.0, float(frame_age_s or 0.0)))
    except (TypeError, ValueError):
        age = 0.0
    source = str(source or "")[:40]
    FREESPACE.update(time=time.time() - age, strips=clean, mode=str(mode or "relative")[:16], source=source)
    POSTERS[source] = time.time() - age
    if eyes_url:  # an eyes service without Gemma (the Pi's own) posts none, so a laptop's stays known
        FREESPACE["eyes_url"] = str(eyes_url)[:200]
    try:
        vals = [float(v) for v in (circle or [])]
        cx, cy, r = vals[:3]
        rh = vals[3] if len(vals) > 3 else r  # the radius as a fraction of the height, if the eyes service says
        if 0.2 <= cx <= 0.8 and 0.2 <= cy <= 0.8 and 0.15 <= r <= 0.5 and 0.15 <= rh <= 0.6:
            FREESPACE["circle"] = [cx, cy, r, rh]
    except (TypeError, ValueError):
        pass
    return len(clean)


def camera_height_m(robot) -> float:
    """How high the camera is right now: lying (motors idle) or standing (anything else)."""
    from robot_api import MOTORS_IDLE

    try:
        lying = robot.state_now is MOTORS_IDLE
    except Exception:
        lying = False
    return CAM_HEIGHT_LYING_M if lying else CAM_HEIGHT_STANDING_M


def fresh_freespace() -> list[dict] | None:
    """The strips if the depth summary is recent, else None (no depth sense: no rule applies)."""
    if time.time() - FREESPACE["time"] > FREESPACE_FRESH_S:
        return None
    return FREESPACE["strips"]


def strip_blocked(st: dict) -> str | None:
    """Why this strip counts as blocked, or None. The eyes service posts nearness (Lab 5's row of
    the base, calibrated); closest_m is the metric form the fake tests use."""
    n = st.get("nearness")
    if n is not None:
        from fisheye_geom import STOP_NEARNESS

        if n >= STOP_NEARNESS:
            return "right at my feet"
        return "very close" if n >= BLOCK_NEARNESS else None
    if st.get("closest_m") is not None:
        return f"about {st['closest_m']:.1f} m away" if st["closest_m"] <= STOP_M else None
    if st.get("near_frac") is not None and st["near_frac"] >= NEAR_FRAC_BLOCK:
        return "very close (low, by my depth sense)"
    return None


def freespace_blocked(vx: float, vy: float) -> str | None:
    """None if the depth sense shows nothing close in the way of a move at (vx, vy), else why not.
    Class-agnostic: it doesn't know what the thing is, only that it is there."""
    strips = fresh_freespace()
    if not strips:
        return None
    wanted: set[str] = set()
    if vx > 0:
        wanted |= AHEAD
    if vy > 0:
        wanted |= LEFT
    if vy < 0:
        wanted |= RIGHT
    hits = [(st, why) for st in strips if st["name"] in wanted for why in [strip_blocked(st)] if why]
    if not hits:
        return None
    st, why = max(hits, key=lambda h: (h[0].get("nearness") or 0.0, -(h[0]["closest_m"] if h[0]["closest_m"] is not None else 9.0)))
    where = "right in front of me" if st["name"] == "center" else f"ahead on my {'left' if st['name'] in LEFT else 'right'}" \
        if st["name"] in AHEAD else f"on my {'left' if st['name'] in LEFT else 'right'}"
    return (f"blocked: something low is {where}, {why}; my depth sense sees it even though I can't name it"
            " (a chair leg, a cable, a shoe?). I won't walk into it. Turn away from it, step the other way,"
            " or ask the person to move it")


def how_near(n: float | None) -> str:
    """Lab 5's row of a thing's base, in words a person understands."""
    from fisheye_geom import SLOW_NEARNESS, STOP_NEARNESS

    if n is None:
        return "at an unknown distance"
    if n >= STOP_NEARNESS:
        return "right at my feet"
    if n >= BLOCK_NEARNESS:
        return "very close"
    if n >= SLOW_NEARNESS:
        return "close"
    return "far away" if n < 0.55 else "a few steps away"


# ---------------------------------------------------------------- follow mode's guard
# Follow mode is the Lab 5 person follower, a ROS node that drives the legs itself and never
# asks the robot server, so without this it walks into chair legs. The robot server runs
# follow_guard_loop: while following, the depth sense is checked ten times a second, and
# something in the way pauses the follower (it is switched off); once the way has been clear for
# CLEAR_S it is switched back on, unless a stop or another motion came in meanwhile, or the way
# stays blocked for PAUSE_MAX_S.
FOLLOW_SPEED = 0.3  # m/s: about how fast the follower walks; what "in the way" is judged against
CLEAR_S = 1.0
PAUSE_MAX_S = 20.0
FOLLOW_GUARD: dict = {"paused_since": 0.0, "clear_since": 0.0, "stops": 0, "why": ""}


def _unpause() -> None:
    FOLLOW_GUARD.update(paused_since=0.0, clear_since=0.0, why="")


async def follow_guard_tick(robot) -> str | None:
    """One check. Returns what it did ("paused", "resumed", "gave up") or None."""
    if robot.following:
        _unpause()
        blocked = path_blocked(FOLLOW_SPEED, 0.0)
        if not blocked:
            return None
        await robot.stop_following()
        robot.send_speed(0, 0, 0)
        FOLLOW_GUARD.update(paused_since=robot.now(), clear_since=0.0, stops=robot.stops, why=blocked)
        EVENTS.emit("obstacle", f"paused following: {blocked}", for_gemini=(
            f"[robot, not the person speaking] You paused following the person: {blocked}. Say so in one short"
            f" sentence. You carry on by yourself once the way has been clear for a moment (you give up after"
            f" {PAUSE_MAX_S:.0f} s and say so)."))
        return "paused"
    if not FOLLOW_GUARD["paused_since"]:
        return None
    if robot.stops != FOLLOW_GUARD["stops"] or robot.current is not None or not robot.commands.empty():
        _unpause()  # a stop or another motion took over: following is over
        return None
    if robot.now() - FOLLOW_GUARD["paused_since"] > PAUSE_MAX_S:
        why = FOLLOW_GUARD["why"]
        _unpause()
        EVENTS.emit("follow", "stopped following: the way stayed blocked", for_gemini=(
            f"[robot, not the person speaking] You stopped following: the way stayed blocked for {PAUSE_MAX_S:.0f} s"
            f" ({why.split(';')[0]}). Say so in one short sentence; the person can ask you to follow again."))
        return "gave up"
    if path_blocked(FOLLOW_SPEED, 0.0):
        FOLLOW_GUARD["clear_since"] = 0.0
        return None
    if not FOLLOW_GUARD["clear_since"]:
        FOLLOW_GUARD["clear_since"] = robot.now()
        return None
    if robot.now() - FOLLOW_GUARD["clear_since"] < CLEAR_S:
        return None
    res = await robot.set_following(True)
    _unpause()
    if res.ok:
        EVENTS.emit("follow", "following again: the way is clear", for_gemini=(
            "[robot, not the person speaking] The way is clear and you are following the person again. Say so in a few words."))
        return "resumed"
    EVENTS.emit("follow", f"could not resume following: {res.message}", for_gemini=(
        f"[robot, not the person speaking] You could not start following again: {res.message}. Say so in one short sentence."))
    return "gave up"


async def follow_guard_loop(robot) -> None:
    """Background task in the robot server: ten times a second."""
    while True:
        try:
            await follow_guard_tick(robot)
        except Exception:  # a bug here must not stop the server
            log.exception("follow guard failed")
        await robot.sleep(0.1)


def _follow_tool_hook(name: str, args: dict, res) -> None:
    """follow(enabled=false) during a pause means: don't resume."""
    if name == "follow":
        _unpause()


try:
    from tool_registry import CALL_HOOKS

    CALL_HOOKS.append(_follow_tool_hook)
except ImportError:
    pass


def update_scene(text: str) -> int:
    SCENE.update(time=time.time(), text=" ".join(str(text or "").split())[:600])
    return len(SCENE["text"])


def fresh() -> list[dict] | None:
    """The detections if they are recent, else None (blind: no rule applies)."""
    if time.time() - VISION["time"] > FRESH_S:
        return None
    return [d for d in VISION["detections"] if d["score"] >= MIN_SCORE]


def side_of(d: dict) -> str:
    from fisheye_geom import box_bearing_deg

    b = box_bearing_deg(d["box"])
    return "to the left" if b < -12 else "to the right" if b > 12 else "straight ahead"


def how_close(d: dict) -> str:
    """How near a detected thing is: Lab 5's rule on its box's bottom edge (its base on the floor),
    with the box's size as the fallback for a box the lens model can't place."""
    from fisheye_geom import box_nearness

    n = box_nearness(d["box"])
    if n is not None:
        return how_near(n)
    x, y, w, h = d["box"]
    if h >= 0.6 or w >= 0.7:
        return "very close"
    if h >= 0.4 or (y + h >= 0.9 and w * h >= 0.08):
        return "close"
    return "a few steps away" if h >= 0.2 else "far away"


def is_close(d: dict) -> bool:
    """In the way if walked into: its base at BLOCK_NEARNESS or lower in the image (the row that
    blocks a walk), or, for a box the lens model can't place, big in the frame."""
    from fisheye_geom import box_nearness

    n = box_nearness(d["box"])
    if n is not None:
        return n >= BLOCK_NEARNESS
    return how_close(d) in ("very close", "close")


def describe(d: dict) -> str:
    return f"{d['label']} {side_of(d)}, {how_close(d)}"


def path_blocked(vx: float, vy: float) -> str | None:
    """None if nothing close is in the way of a move at (vx, vy), else why not.

    Two senses, both enforced here: the detector names what it knows (a chair, a person);
    the depth sense catches anything else (chair legs, cables) without naming it."""
    named = detection_blocked(vx, vy)
    if named:
        return named
    return freespace_blocked(vx, vy)


def detection_blocked(vx: float, vy: float) -> str | None:
    """The detector's rule: a known thing, close, in the way (or None)."""
    dets = fresh()
    if not dets:
        return None
    for d in sorted(dets, key=lambda d: -d["box"][3]):  # the biggest (closest) first
        if not is_close(d):
            continue
        cx = d["box"][0] + d["box"][2] / 2
        ahead = 0.25 <= cx <= 0.75
        if (vx > 0 and ahead) or (vy > 0 and cx < 0.4) or (vy < 0 and cx > 0.6):
            where = "right in front of me" if ahead else f"on my {'left' if cx < 0.5 else 'right'}"
            return (f"blocked: a {d['label']} is {where}, {how_close(d)}; I won't walk into it."
                    " Turn away from it, step the other way, or ask the person to move it")
    return None


def _matches(wanted: str, label: str) -> bool:
    wanted = SYNONYMS.get(wanted, wanted)
    if label in LOOKALIKES.get(wanted, set()):
        return True
    return wanted == label or wanted in label.split() or label in wanted.split() or wanted == label.replace(" ", "")


async def seen_now(name: str, since: float = 0.0) -> dict | None:
    """Is the thing in view right now? The detector's word (fresh detections), else the eyes
    service's /exists: Gemma for anything the detector has no class for, or misses. None when not
    seen, or when there are no eyes. (find asks this at every stop.)"""
    hit = visible(name, since)
    if hit:
        return {"source": "detector", "box": hit["box"], "label": hit["label"]}
    import hackathon_eyes  # here, not at the top: hackathon_eyes imports this file

    url = hackathon_eyes.eyes_url()
    if not url:
        return None
    try:
        reply = await asyncio.to_thread(hackathon_eyes.ask_eyes, url, "/exists", {"name": name}, 20.0)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        log.warning("eyes /exists failed: %s", e)
        return None
    if reply.get("found"):
        return {"source": reply.get("source", "eyes"), "box": reply.get("box"), "label": name}
    return None


def visible(name: str, since: float = 0.0) -> dict | None:
    """The best fresh detection of ``name`` (seen after ``since``), or None."""
    if VISION["time"] < since:
        return None
    dets = fresh() or []
    hits = [d for d in dets if _matches(name, d["label"])]
    return max(hits, key=lambda d: d["score"]) if hits else None


def freespace_words() -> str:
    """The depth sense in a sentence: which strips are blocked, or 'clear ahead'."""
    strips = fresh_freespace()
    if strips is None:
        return "no depth sense right now (the eyes service isn't posting)"
    blocked = [f"{st['name']} ({why})" for st in strips for why in [strip_blocked(st)] if why]
    if not blocked:
        return "depth sense: the floor ahead is clear"
    return "depth sense: something close in the " + ", ".join(blocked) + " of the view"


def what_i_see() -> Result:
    dets = fresh()
    if dets is None:
        return Result(True, "my object detector has no recent frame (is the app running with object detection on?);"
                            " use what you see in the camera yourself. " + freespace_words())
    if not dets:
        return Result(True, "the detector sees nothing it knows right now. " + freespace_words())
    return Result(True, "I detect: " + "; ".join(describe(d) for d in sorted(dets, key=lambda d: -d["score"])[:8])
                  + ". " + freespace_words())


def _openness(st: dict) -> float:
    """0 (blocked) .. 1 (nothing near) for one strip."""
    if strip_blocked(st):
        return 0.0
    if st.get("nearness") is not None:
        return 1.0 - st["nearness"]
    if st.get("closest_m") is not None:
        return min(1.0, st["closest_m"] / 2.0)
    return 1.0


def clearer_side() -> str | None:
    """'left' or 'right': the more open way round something ahead, or None if both are blocked
    or there is no fresh depth sense (then the walk stops). A side's openness is its
    worst strip's; the detector's close things block their side too."""
    strips = fresh_freespace()
    if not strips:
        return None
    score = {}
    for side, names in (("left", LEFT), ("right", RIGHT)):
        vals = [_openness(st) for st in strips if st["name"] in names]
        score[side] = min(vals) if vals else 1.0
    for d in fresh() or []:
        if is_close(d):
            cx = d["box"][0] + d["box"][2] / 2
            score["left" if cx < 0.5 else "right"] = 0.0
    side = max(score, key=score.get)
    return side if score[side] > 0.0 else None
