"""Hackathon, part E: eyes that understand. Walk to things by sight, and ask Gemma about the frame.

The brain (Gemini Live today, a local Gemma tomorrow) is slow: about one decision a second.
The legs need a speed twenty times a second. So the brain never steers; it names a goal, and a
loop on the robot closes the gap, the way the Lab 5 person follower does for people:

  go_to(what)        the brain names a target in plain words ("the red mug", "the chair leg").
                     The eyes service (vision/eyes.py) grounds it: it asks Gemma (or the COCO
                     detector, when the name is one of its classes) where the thing is in the
                     frame and posts the box to POST /vision/target a few times a second. A job
                     here, shaped like walk(), reads the latest box every tick, turns toward it
                     and walks, slows near it, and stops when it is close, lost, blocked or out
                     of time. The result comes back as an event, so the brain hears it.
  look(question)     one question about the current frame, answered by Gemma through the eyes
                     service ("is anyone waving?", "what is on the floor ahead?").
  what_is_around     the scene description Gemma posts while Pupper is idle, plus the depth
                     sense and the detector: what the brain reads before planning a move.

The obstacle rules in hackathon_vision.py (detector + depth) run inside the go_to loop too.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import urllib.error
import urllib.request

import fisheye_geom
import hackathon_memory as memory
import hackathon_vision as vision
from hackathon_alerts import EVENTS, show
from part3_follow import prepare_to_move
from robot_api import FELL_MESSAGE, LLM_CMD_RATE, LOCOMOTION, LOCOMOTION_CONTROLLER, Result, Robot, with_note
from tool_registry import Param, tool

log = logging.getLogger("vision")

HFOV_DEG = 180.0  # Pupper's camera is a fisheye (about 180 degrees across): with its equidistant projection,
# the bearing of a thing is close to proportional to how far from the centre column it is, so a box at the
# frame's edge is about 90 degrees off and one a quarter of the way in is about 45
KP_TURN = 1.5  # deg/s of turn per degree of bearing
AIM_DEG = 5.0  # facing it well enough: don't turn
WALK_WITHIN_DEG = 20.0  # walk forward only when roughly facing it; otherwise turn first
# Arrival is Lab 5's rule, calibrated on the robot (lab/fisheye_geom.py): the row of the thing's base in
# the undistorted image. STOP_NEARNESS (0.794, Lab 5's STOP_BOTTOM 454 of 572) is the follower's polite
# distance at a person's feet; NEAR_NEARNESS is "about there": a thing whose base is this low while the
# way ahead is blocked at the robot's feet counts as reached (a backpack at the feet reads 0.71).
ARRIVE_NEARNESS = fisheye_geom.STOP_NEARNESS
NEAR_NEARNESS = vision.BLOCK_NEARNESS - 0.03  # about the row that blocks a walk (a chair's box bottom sits a little above its legs)
SLOW_NEARNESS = fisheye_geom.SLOW_NEARNESS  # Lab 5's SLOWDOWN_PX before the stop
ARRIVE_TICKS = 8  # ... for this many ticks in a row (0.4 s, two depth frames): one noisy frame is not an arrival
SLOW_M = 0.9  # from here on, half speed (a bottom-unknown target, ranged by the depth strip in metres)
# Gemma takes 2-3 s per fix, and the eyes service wants two fixes that agree before it posts the first
# box, so the first box takes 5-8 s and then one comes every 2-3 s.
# Over the Pupper's slow WiFi, a Gemma sighting takes several seconds, and the robot steers only once
# two agree: 10 s to see, 4 s fresh and 6 s lost gave up just before the box came (a pink can,
# found by find, then "can't see it (waited 10 s)" 3 s before the eyes posted its box).
SEE_S = 30.0  # how long to stand and wait for the eyes to find the thing at all
LOST_S = 12.0  # how long to keep going on the last box before giving up
TARGET_FRESH_S = 8.0  # a box older than this is not steered on
SCENE_FRESH_S = 90.0
EYES_TIMEOUT_S = 25.0

# What go_to wants tracked (the eyes service polls GET /vision/wanted), and the latest box
# (POST /vision/target). box = [x, y, w, h] as fractions of the frame, or None (looked, not there).
WANTED: dict = {"name": "", "since": 0.0}
TARGET: dict = {"time": 0.0, "name": "", "box": None, "score": 0.0, "source": "", "bottom_known": True}


# ============================================================ what the robot does: the target
def update_target(name: str, box: list | None, score: float = 1.0, source: str = "eyes", bottom_known: bool = True) -> bool:
    """The eyes service's latest fix on the wanted thing (box None: it looked and didn't see it)."""
    clean = None
    if box is not None:
        try:
            clean = [max(0.0, min(1.0, float(v))) for v in box][:4]
            if len(clean) != 4 or clean[2] <= 0 or clean[3] <= 0:
                clean = None
        except (TypeError, ValueError):
            clean = None
    TARGET.update(time=time.time(), name=str(name or "").lower()[:60], box=clean,
                  score=float(score or 0.0), source=str(source or "")[:20], bottom_known=bool(bottom_known))
    return clean is not None


def target_box(name: str) -> tuple[list, str, bool] | None:
    """The freshest box for ``name``: the eyes service's, else the COCO detector's.
    (box, source, bottom_known) or None. A detector's box always has a real bottom edge."""
    if TARGET["box"] is not None and TARGET["name"] == name and time.time() - TARGET["time"] <= TARGET_FRESH_S:
        return TARGET["box"], TARGET["source"] or "eyes", bool(TARGET.get("bottom_known", True))
    hit = vision.visible(name)
    if hit:
        return hit["box"], "detector", True
    return None


def _det(box: list) -> dict:
    """A box as the detection dict hackathon_vision's side_of / how_close / is_close read."""
    return {"label": "target", "score": 1.0, "box": box}


def circle() -> tuple[float, float, float, float]:
    """The fisheye's disc: centre x, centre y, radius (of the width), radius (of the height). Found by
    the eyes service (vision.FREESPACE['circle']); without one, the whole frame is taken to be it."""
    c = list(vision.FREESPACE.get("circle") or [0.5, 0.5, 0.5, 0.5])
    return (c[0], c[1], c[2], c[3] if len(c) > 3 else c[2])


def bearing_deg(box: list) -> float:
    """Where the box is, in degrees from straight ahead; positive is to the right: the longitude of
    its centre through Lab 5's calibrated lens model (lab/fisheye_geom.py)."""
    return fisheye_geom.box_bearing_deg(box)


def strip_at(bearing: float) -> str:
    """Which free-space strip a bearing falls in (the eyes service cuts them at -45, -15, 15, 45)."""
    return ("left" if bearing < -45 else "center-left" if bearing < -15 else "center" if bearing <= 15
            else "center-right" if bearing <= 45 else "right")


def box_distance_m(box: list, cam_height_m: float) -> float | None:
    """The range from the box alone, in metres, for words only: Lab 5's row of its bottom edge turned
    into a distance for a level camera cam_height_m up. The rules use the row itself (nearness)."""
    n = fisheye_geom.box_nearness(box)
    if n is None:
        return None
    lat = n * 180.0 - 90.0
    return None if lat < 1.0 else min(9.99, cam_height_m / math.tan(math.radians(lat)))


def strip_range_m(bearing: float) -> float | None:
    """The depth sense's range in the strip at ``bearing``: the nearest surface there, which is a
    chair leg as often as the thing aimed at."""
    for st in vision.fresh_freespace() or []:
        if st["name"] == strip_at(bearing) and st.get("closest_m") is not None:
            return st["closest_m"]
    return None


def distance_m(box: list, cam_height_m: float, aim_deg: float | None = None, bottom_known: bool = True) -> tuple[float | None, bool]:
    """How far away the thing in the box is: (metres or None, from_the_thing_itself).

    Lab 5's rule: the bottom edge of the thing's box is where it meets the floor, and in a level
    fisheye a point e degrees below the horizon is cam_height / tan(e) away. That is the only range
    that is about the thing, so only it can say "arrived" (bottom_known: a detector's box, or a
    model's box with a real size; a model that only pointed gives a bearing, not a range). The depth
    sense's strip in that direction is a stand-in for slowing down when the bottom is unknown.
    ``aim_deg``: the thing's bearing now (dead-reckoned between fixes), else the box's."""
    if bottom_known:
        return box_distance_m(box, cam_height_m), True
    return strip_range_m(bearing_deg(box) if aim_deg is None else aim_deg), False


def steer(b: float, walk: float, turn: float, wz_min: float, vx_min: float, dist: float | None = None,
          near: float | None = None) -> tuple[float, float]:
    """The speed command (vx, wz_deg) toward a thing at bearing ``b`` (degrees, + right), ``dist``
    metres away (a depth-strip range) or at Lab 5 row ``near``, within the state's speeds."""
    if abs(b) <= AIM_DEG:
        wz = 0.0
    else:
        wz = max(-turn, min(turn, -KP_TURN * b))  # a target on the right (b > 0) needs a right turn (wz < 0)
        if abs(wz) < wz_min:
            wz = math.copysign(wz_min, wz)
    vx = walk if abs(b) <= WALK_WITHIN_DEG else 0.0
    if vx and ((near is not None and near >= SLOW_NEARNESS) or (near is None and dist is not None and dist <= SLOW_M)):
        vx = max(vx_min, walk / 2)
    return vx, wz


async def go_to(robot: Robot, what: str) -> Result:
    name = " ".join(what.split()).lower().strip(".!?")
    for lead in ("the ", "a ", "an ", "my ", "that ", "this "):
        if name.startswith(lead):
            name = name[len(lead):]
    if not name:
        return Result(False, "what should I walk to? e.g. go_to(what='the red mug')")
    from part2_walk import check_move  # here, not at the top: part2_walk imports hackathon_vision

    speeds = robot.speeds()
    limits = speeds.limits
    max_s = min(limits.duration_max, limits.distance_max / speeds.walk)  # within one move's time and distance
    refused = check_move(speeds.walk, 0.0, speeds.turn, max_s, limits)
    if refused:
        return Result(False, f"can't walk there: {refused}{speeds.where}")
    note = await prepare_to_move(robot)
    if robot.fallen:
        return Result(False, FELL_MESSAGE)
    state = robot.state_after_queue
    walker = robot.walking_controller(state)
    then = state if walker != LOCOMOTION_CONTROLLER else LOCOMOTION
    WANTED.update(name=name, since=time.time())

    def done(ok: bool, text: str, *, where: str | None = None) -> Result:
        robot.send_speed(0, 0, 0)
        WANTED.update(name="", since=0.0)
        if ok and where:
            memory.remember_object(name, where)
        show(name.upper(), "reached" if ok else text[:60], "green" if ok else "orange", 12, kind="goto", tell_gemini=(
            f"[robot, not the person speaking] go_to result: {text}. Tell the person in one short sentence,"
            " nothing else about what you see."))
        return Result(ok, text)

    async def job() -> Result:
        if robot.fallen:
            return Result(False, FELL_MESSAGE)
        res = await robot.switch_to(walker)
        if not res.ok:
            WANTED.update(name="", since=0.0)
            return res
        start = robot.now()
        # One move's time (max_s) counts from the first sighting: standing still while the eyes look
        # (up to SEE_S) is not moving, and over the Pupper's slow WiFi it can take a while.
        deadline = start + SEE_S + max_s
        walked = 0.0
        last_box: list | None = None
        last_seen: float | None = None
        source = ""
        bottom_known = True
        fix_time = -1.0  # when the box in hand was taken; a fix is only new when this changes
        fix_at = 0.0  # robot time of that fix
        walked_since = 0.0  # metres walked since that fix: the fix's row is pushed down the image by it (dead reckoning)
        aim = 0.0  # the thing's bearing now: a fix sets it, and the turns sent since move it (dead reckoning)
        near_ticks = 0  # how many ticks in a row the thing's own bottom edge has put it within arm's length
        near_fixes: set = set()  # ... and from how many different fixes (one fix can be wrong; two in a row less so)
        while robot.now() < deadline - 1e-6:
            if robot.fallen:
                robot.send_speed(0, 0, 0)
                WANTED.update(name="", since=0.0)
                return Result(False, FELL_MESSAGE)
            hit = target_box(name)
            if hit:
                if last_seen is None:
                    deadline = robot.now() + max_s
                last_box, source, bottom_known = hit
                last_seen = robot.now()
                taken = TARGET["time"] if source != "detector" else vision.VISION["time"]
                if taken != fix_time:  # a new fix: aim at it
                    fix_time, fix_at, walked_since = taken, robot.now(), 0.0
                    aim = bearing_deg(last_box)
            if last_seen is None:  # nothing yet: stand and give the eyes a moment
                if robot.now() - start >= SEE_S:
                    return done(False, f"I can't see the {name} (waited {SEE_S:.0f} s). Call find to look around,"
                                       " or ask where it is")
                robot.send_speed(0, 0, 0)
                await robot.sleep(1.0 / LLM_CMD_RATE)
                continue
            if robot.now() - last_seen > LOST_S:
                return done(False, f"lost sight of the {name} for {LOST_S:.0f} s; stopped where I am")
            assert last_box is not None
            near = fisheye_geom.box_nearness(last_box) if bottom_known else None  # Lab 5's row of its base, at the fix
            if near is not None and walked_since > 0.0:  # pushed down the image by the walking since the fix
                near = fisheye_geom.row_after_walking(near, walked_since)
            dist = None if bottom_known else strip_range_m(aim)  # a pointed-at thing: the depth strip, for slowing only
            # Arrived: the thing's own bottom edge is at Lab 5's stop row, on several ticks and two
            # different fixes. Never from the depth strips: they see the nearest surface in that
            # direction, a chair leg as often as the thing.
            if near is not None and near >= ARRIVE_NEARNESS:
                near_ticks += 1
                near_fixes.add(fix_time)
            else:
                near_ticks, near_fixes = 0, set()
            if near_ticks >= ARRIVE_TICKS and len(near_fixes) >= 2:
                return done(True, f"reached the {name}: it is right in front of me, {vision.how_near(near)}",
                            where=f"where I walked to it ({vision.side_of(_det(last_box))})")
            if near_ticks:  # hold still until the next fix confirms it
                robot.send_speed(0, 0, 0)
                await robot.sleep(1.0 / LLM_CMD_RATE)
                continue
            vx, wz = steer(aim, speeds.walk, speeds.turn, limits.wz_min, limits.vx_min, dist, near)
            if vx:
                blocked = vision.path_blocked(vx, 0.0)
                if blocked:
                    # The thing in the way is the target only if the target itself is about there: its
                    # own base is at the blocking row; or it is dead ahead and its last fix, now stale (the
                    # robot has walked on since), already had it close; or, with no base known, it is dead
                    # ahead. A far target (its base up near the horizon) behind a near obstacle is a stop.
                    about_there = (near is not None and near >= NEAR_NEARNESS) or (near is None and abs(aim) <= AIM_DEG * 2)
                    if abs(aim) <= WALK_WITHIN_DEG and about_there:
                        return done(True, f"reached the {name}: stopped right in front of it ({blocked.split(';')[0]})",
                                    where="right in front of me, where I stopped")
                    return done(False, f"stopped on the way to the {name}: {blocked}")
            if walked + vx / LLM_CMD_RATE > limits.distance_max:
                return done(False, f"gave up after {walked:.1f} m (at most {limits.distance_max} m in one go): the {name}"
                                   " is still ahead. Ask me again to keep going")
            robot.send_speed(vx, 0.0, wz)
            walked += vx / LLM_CMD_RATE
            walked_since += vx / LLM_CMD_RATE
            aim += wz / LLM_CMD_RATE  # a left turn (wz > 0) swings a thing on the left (aim < 0) toward ahead
            await robot.sleep(1.0 / LLM_CMD_RATE)
        return done(False, f"gave up after {max_s:.0f} s: the {name} is still ahead but I didn't reach it"
                           " (seen through " + (source or "nothing") + "). Ask me again to keep going")

    seen = target_box(name)
    if seen:
        box, src, known = seen
        how = vision.how_near(fisheye_geom.box_nearness(box)) if known else "at an unknown distance"
        intro = f"Walking to the {name}: I see it {vision.side_of(_det(box))} ({bearing_deg(box):+.0f} degrees), {how} ({src})."
    else:
        intro = f"Walking to the {name} once my eyes find it (I don't see it yet; I wait up to {SEE_S:.0f} s)."
    intro += (f" I turn toward it and walk at up to {speeds.walk:g} m/s, stop at arm's length or if something"
              f" is in the way, and give up after {max_s:.0f} s. You will be told the result.")
    return with_note(f"{note} {intro}", await robot.queue(f"go to {name}", job, 10.0, then=then))


# ============================================================ what the robot does: asking Gemma
def eyes_url() -> str:
    """Where the eyes service is: $PUPPER_EYES_URL, else what it said when it last posted free space."""
    return os.environ.get("PUPPER_EYES_URL", "").rstrip("/") or vision.FREESPACE.get("eyes_url", "").rstrip("/")


def ask_eyes(url: str, path: str, payload: dict, timeout: float = EYES_TIMEOUT_S) -> dict:
    """POST json to the eyes service and return its json (blocking: call it in a thread)."""
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


async def look(robot: Robot, question: str) -> Result:
    question = " ".join(question.split())
    if not question:
        return Result(False, "what should I look for? e.g. look(question='what is on the floor ahead of me?')")
    url = eyes_url()
    if not url:
        return Result(False, "my eyes service isn't running (vision/eyes.py), so I can't ask Gemma about the frame;"
                             " use what you see in the camera yourself")
    try:
        reply = await asyncio.to_thread(ask_eyes, url, "/look", {"question": question})
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        return Result(False, f"my eyes service didn't answer ({type(e).__name__}); use what you see in the camera yourself")
    answer = " ".join(str(reply.get("answer", "")).split())
    if not answer:
        return Result(False, "my eyes service gave no answer; use what you see in the camera yourself")
    return Result(True, f"Gemma, looking at the frame: {answer}")


def around() -> Result:
    parts = []
    if vision.SCENE["text"] and time.time() - vision.SCENE["time"] <= SCENE_FRESH_S:
        parts.append(f"scene ({time.time() - vision.SCENE['time']:.0f} s ago): {vision.SCENE['text']}")
    else:
        parts.append("no fresh scene description (the eyes service posts one while I stand still; call look to ask now)")
    parts.append(vision.what_i_see().message)
    return Result(True, " ".join(parts))


# ============================================================ what Gemini reads
@tool(
    "go_to",
    "Walk to a thing you can name, even one the detector doesn't know: 'the red mug', 'the chair leg',"
    " 'the door', 'Devesh'. Your eyes find it in the camera, and Pupper turns toward it, walks, slows"
    " down and stops at arm's length, or stops early if something is in the way or it loses sight of"
    " it. You are told the result (a [robot] message). It must be in view: if it isn't, call find"
    " instead. Use it for 'come here', 'go to the couch', 'walk over to the ball'."
    " If follow mode is on, this ends it: say so first.",
    what=Param("string", "What to walk to, in a few words: 'the red mug', 'the person in blue', 'the chair'."),
)
async def go_to_tool(robot: Robot, what: str) -> Result:
    return await go_to(robot, what)


@tool(
    "look",
    "Ask your own eyes (a local Gemma vision model) one question about the camera frame right now,"
    " and get a sentence back: 'what is on the floor ahead?', 'is anyone waving?', 'what color is"
    " the mug?', 'is the way to the door clear?'. Slower than the detector (a few seconds) but it"
    " understands anything. Use it before walking somewhere new, or when asked what you see.",
    question=Param("string", "The question, in one sentence."),
)
async def look_tool(robot: Robot, question: str) -> Result:
    return await look(robot, question)


@tool(
    "what_is_around",
    "What your eyes last noted about the room (Gemma's scene description: furniture, people, things"
    " on the floor, left/right/ahead), plus the detector and the depth sense right now. Read it before"
    " planning a walk, or when asked what is around you. It costs nothing: no new picture is taken."
    " It is for deciding, not for telling: the people with you see the room too, so don't read it out.",
)
async def what_is_around_tool(robot: Robot) -> Result:
    return around()
