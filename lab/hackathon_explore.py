"""Hackathon, part F: finding things. "Where is my cup?", "look around for the whiteboard with code on it."

One tool, find(what), does the whole job, cheapest step first:

  1. look: is it in view right now? The eyes say (the detector's word for a thing with a class,
     else Gemma's yes or no, through the eyes service). Memory says where it was last seen.
  2. sweep: turn in place in 60-degree steps, looking after each, a full circle.
  3. roam: Pupper has no map and no odometry worth the name, so this is a smart Roomba, not a
     Matic. At each stop it reads the depth sense's five strips, scores each direction by how
     open it is and whether it leads somewhere new (a coarse grid of visited cells, kept by dead
     reckoning from the speeds sent), turns that way and walks one leg, at most LEG_M, with the
     obstacle rules running every tick, and looks again. Until found, out of time, or boxed in.
  4. found: say so, remember where, and queue a go_to to walk up to it.

It only ever sends the speeds walk and turn use, within check_move's limits; a stop cancels it
like any job; anything in the way stops a leg the way it stops a walk. Dead reckoning drifts, so
the grid is a preference ("go somewhere new"), never a claim about where the robot is.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
import urllib.error

import hackathon_eyes as eyes
import hackathon_memory as memory
import hackathon_vision as vision
from hackathon_alerts import EVENTS, show
from part3_follow import prepare_to_move
from robot_api import FELL_MESSAGE, LLM_CMD_RATE, LOCOMOTION, LOCOMOTION_CONTROLLER, Result, Robot, with_note
from tool_registry import Param, tool

log = logging.getLogger("vision")

LEG_M = 1.5  # one leg at most; then stop and look again
CELL_M = 1.0  # the visited grid's cell
MAX_S_DEFAULT = 120.0
MAX_S_LIMIT = 300.0
LOOK_S = 0.6  # stand still at each stop so the depth sense (3-5 Hz) and detector settle
OPEN_M = 2.0  # a strip with nothing in it counts as this open
BLOCKED_M = vision.STOP_M + 0.1  # a direction this short is not worth turning to
VISITED_WEIGHT = 0.35  # a direction leading to a visited cell keeps this much of its score
STRAIGHT_BONUS = 1.15  # carrying on is a little preferred (longer legs, fewer turns)
TURN_AROUND_DEG = 150.0  # when nothing ahead is worth it: turn about, with a little randomness
STUCK_LEGS = 4  # legs in a row that could not start: boxed in, give up
SWEEP_STEP_DEG = 60.0  # the sweep in place: a full circle in steps, a look after each
SWEEP_STEPS = 5  # 5 turns + the look before the first one = 6 looks around the circle
SECTORS = [("left", -60.0), ("center-left", -30.0), ("center", 0.0), ("center-right", 30.0), ("right", 60.0)]
MIN_LEG_M = 0.1  # a leg shorter than this did not start

EXPLORING: dict = {"name": "", "since": 0.0, "started": False}  # started: the queued search began
rng = random.Random()


# ============================================================ what the robot does
class DeadReckoning:
    """Where Pupper roughly is, from the speeds sent: x, y in metres, heading in degrees (+ left),
    and the grid cells it has been in."""

    def __init__(self) -> None:
        self.x = self.y = 0.0
        self.heading = 0.0
        self.visited: set[tuple[int, int]] = {(0, 0)}

    def move(self, vx: float, wz_deg: float, dt: float) -> None:
        self.heading = (self.heading + wz_deg * dt + 180.0) % 360.0 - 180.0
        self.x += vx * math.cos(math.radians(self.heading)) * dt
        self.y += vx * math.sin(math.radians(self.heading)) * dt
        self.visited.add(self.cell(self.x, self.y))

    @staticmethod
    def cell(x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / CELL_M + 0.5), math.floor(y / CELL_M + 0.5))

    def ahead(self, bearing_deg: float, dist: float) -> tuple[int, int]:
        """The cell ``dist`` metres away at ``bearing_deg`` (+ right of the heading)."""
        h = math.radians(self.heading - bearing_deg)
        return self.cell(self.x + dist * math.cos(h), self.y + dist * math.sin(h))


def sector_ranges(strips: list[dict] | None) -> dict[str, float]:
    """How far each strip is open, in metres (OPEN_M when nothing is in it; 0 without a depth sense)."""
    out = {}
    for name, _ in SECTORS:
        st = next((s for s in (strips or []) if s["name"] == name), None)
        if st is None:
            out[name] = 0.0 if strips is None else OPEN_M
        elif st.get("nearness") is not None:  # Lab 5's row of the nearest base: metres for a level camera 0.22 m up
            lat = st["nearness"] * 180.0 - 90.0
            out[name] = OPEN_M if lat < 1.0 else min(OPEN_M, 0.22 / math.tan(math.radians(lat)))
        elif st.get("closest_m") is not None:
            out[name] = min(OPEN_M, st["closest_m"])
        elif st.get("near_frac") is not None and st["near_frac"] >= vision.NEAR_FRAC_BLOCK:
            out[name] = 0.0
        else:
            out[name] = OPEN_M
    return out


def choose_direction(ranges: dict[str, float], pose: DeadReckoning, chooser: random.Random = rng) -> tuple[float, float] | None:
    """(bearing to turn to, how far to walk) or None when nothing ahead is worth it.

    Score = openness (0..1) x novelty (1 for a cell not yet visited, VISITED_WEIGHT otherwise)
    x a bonus for straight on x a little noise, so two runs don't trace the same path."""
    best = None
    for name, bearing in SECTORS:
        r = ranges.get(name, 0.0)
        if r <= BLOCKED_M:
            continue
        leg = min(LEG_M, r - vision.STOP_M)
        novelty = 1.0 if pose.ahead(bearing, leg) not in pose.visited else VISITED_WEIGHT
        score = (r / OPEN_M) * novelty * (STRAIGHT_BONUS if bearing == 0.0 else 1.0) * chooser.uniform(0.85, 1.15)
        if best is None or score > best[0]:
            best = (score, bearing, leg)
    if best is None or best[0] < 0.12:
        return None
    return best[1], best[2]


async def seen(name: str) -> dict | None:
    """Is the thing in view right now? The detector's word (fresh detections), else the eyes
    service's /exists (Gemma for a thing without a class). None when not seen, or no eyes."""
    hit = vision.visible(name)
    if hit:
        return {"source": "detector", "box": hit["box"]}
    url = eyes.eyes_url()
    if not url:
        return None
    try:
        reply = await asyncio.to_thread(eyes.ask_eyes, url, "/exists", {"name": name}, 20.0)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        log.warning("explore: eyes /exists failed: %s", e)
        return None
    if reply.get("found"):
        return {"source": reply.get("source", "eyes"), "box": reply.get("box")}
    return None


async def find(robot: Robot, looking_for: str, max_seconds: float = MAX_S_DEFAULT, approach: bool = True) -> Result:
    name = " ".join(looking_for.split()).lower().strip(".!?")
    for lead in ("the ", "a ", "an ", "my ", "some "):
        if name.startswith(lead):
            name = name[len(lead):]
    if not name:
        return Result(False, "what should I look for? e.g. find(what='the red mug')")
    try:
        max_s = float(max_seconds)
    except (TypeError, ValueError):
        return Result(False, f"max_seconds must be a number of seconds, at most {MAX_S_LIMIT:g}")
    if not 5.0 <= max_s <= MAX_S_LIMIT:
        return Result(False, f"max_seconds must be between 5 and {MAX_S_LIMIT:g} s (asked {max_s:g})")
    # The same search again while one is queued or running would roam twice. A queued search that a
    # stop dropped never starts, so it only counts while something is still queued or running.
    status = robot.status()
    if EXPLORING["name"] == name and (EXPLORING["started"] or status["queued"] or status["running"]):
        return Result(True, f"already looking for the {name}: you'll get a [search result] when that search ends")
    from part2_walk import check_move  # here, not at the top: part2_walk imports hackathon_vision

    speeds = robot.speeds()
    limits = speeds.limits
    refused = check_move(speeds.walk, 0.0, speeds.turn, LEG_M / speeds.walk, limits)
    if refused:
        return Result(False, f"can't roam: {refused}{speeds.where}")
    note = await prepare_to_move(robot)
    if robot.fallen:
        return Result(False, FELL_MESSAGE)
    if not vision.eyes_alive():
        note = f"{note} Warning: my eyes service is not running (vision/eyes.py), so I have no depth sense and will only ever turn, not walk."
    state = robot.state_after_queue
    walker = robot.walking_controller(state)
    then = state if walker != LOCOMOTION_CONTROLLER else LOCOMOTION
    EXPLORING.update(name=name, since=time.time(), started=False)
    last = memory.where_is(name)
    cannot = (f" My detector doesn't know '{name}'; my eyes ask Gemma for it at each stop, and your eyes on the camera matter too."
              if not any(vision._matches(name, c) for c in vision.COCO_NAMES) else "")

    def done(ok: bool, text: str, *, found: dict | None = None) -> Result:
        robot.send_speed(0, 0, 0)
        EXPLORING.update(name="", since=0.0, started=False)
        if found:
            where = (f"{vision.side_of({'box': found['box']})}, {vision.how_close({'box': found['box']})}" if found.get("box")
                     else "in view from where I stopped")
            memory.remember_object(name, f"{where} (found while roaming)")
            show(name.upper(), "found", "green", 12, kind="found", tell_gemini=(
                f"[search result, not the person speaking] Roaming, you found the {name}: {text}."
                + (" You are now walking up to it; you will be told when you get there." if approach else "")
                + " Tell the person in one short sentence."))
        else:
            EVENTS.emit("search", text, for_gemini=(
                f"[search result, not the person speaking] Roaming ended: {text}. Tell the person in one short sentence."))
        return Result(ok, text)

    async def tick(vx: float, wz: float, pose: DeadReckoning) -> None:
        robot.send_speed(vx, 0.0, wz)
        pose.move(vx, wz, 1.0 / LLM_CMD_RATE)
        await robot.sleep(1.0 / LLM_CMD_RATE)

    async def job() -> Result:
        pose = DeadReckoning()
        EXPLORING["started"] = True
        try:
            if robot.fallen:
                return Result(False, FELL_MESSAGE)
            res = await robot.switch_to(walker)
            if not res.ok:
                return res
            start = robot.now()
            legs = failed = sweeps = 0
            while robot.now() - start < max_s:
                # 1. stop and look
                end = robot.now() + LOOK_S
                while robot.now() < end - 1e-6:
                    if robot.fallen:
                        return done(False, FELL_MESSAGE)
                    await tick(0.0, 0.0, pose)
                hit = await seen(name)
                if hit:
                    bearing = eyes.bearing_deg(hit["box"]) if hit.get("box") else None
                    where = f"{bearing:+.0f} degrees from straight ahead" if bearing is not None else "in view"
                    how = (f"after turning {sweeps * SWEEP_STEP_DEG:.0f} degrees" if legs == 0 and sweeps else
                           f"after {legs} legs" if legs else "right away")
                    res = done(True, f"found the {name} {how}: {where} ({hit['source']})", found=hit)
                    if approach:
                        await eyes.go_to(robot, name)  # queued: it runs right after this job
                    return res
                # 2. the sweep first: a full circle in place, looking after each step (cheap, and most things are near)
                if sweeps < SWEEP_STEPS:
                    sweeps += 1
                    end = robot.now() + SWEEP_STEP_DEG / speeds.turn
                    while robot.now() < end - 1e-6:
                        if robot.fallen:
                            return done(False, FELL_MESSAGE)
                        await tick(0.0, speeds.turn, pose)  # left, like turn()
                    robot.send_speed(0, 0, 0)
                    continue
                # 3. then roam: choose a direction
                choice = choose_direction(sector_ranges(vision.fresh_freespace()), pose)
                if choice is None:  # nothing ahead worth it: turn about and look again (a dead end counts as a failed leg)
                    bearing, leg = (TURN_AROUND_DEG + rng.uniform(-30.0, 30.0)) * rng.choice((-1.0, 1.0)), 0.0
                    failed += 1
                    if failed >= STUCK_LEGS:
                        return done(False, f"boxed in: nothing open in any direction {STUCK_LEGS} times in a row; no {name}"
                                           f" seen in {legs} legs. Ask the person where to look")
                else:
                    bearing, leg = choice
                # 4. turn to it (turning in place is always allowed)
                turn_s = abs(bearing) / speeds.turn
                wz = math.copysign(speeds.turn, -bearing)  # a thing on the right (bearing > 0) needs a right turn (wz < 0)
                end = robot.now() + turn_s
                while robot.now() < end - 1e-6:
                    if robot.fallen:
                        return done(False, FELL_MESSAGE)
                    await tick(0.0, wz, pose)
                if leg <= 0.0:
                    continue  # turned about: look again before walking
                # 5. walk the leg, with the obstacle rules every tick, looking out for the thing cheaply
                walked = 0.0
                t = 0
                while walked < leg:
                    if robot.fallen:
                        return done(False, FELL_MESSAGE)
                    blocked = vision.path_blocked(speeds.walk, 0.0)
                    if blocked:
                        break
                    await tick(speeds.walk, 0.0, pose)
                    walked += speeds.walk / LLM_CMD_RATE
                    t += 1
                    if t % 10 == 0 and vision.visible(name):  # the detector sees it mid-leg: stop and look properly
                        break
                robot.send_speed(0, 0, 0)
                legs += 1
                failed = failed + 1 if walked < MIN_LEG_M else 0
                if failed >= STUCK_LEGS:
                    return done(False, f"boxed in: {STUCK_LEGS} legs in a row could not start; no {name} seen in {legs} legs."
                                       " Ask the person where to look")
            return done(False, f"no {name} found in {max_s:.0f} s and {legs} legs; I have seen about"
                               f" {len(pose.visited)} square metres. Ask me to keep looking, or say where it might be")
        except asyncio.CancelledError:
            EXPLORING.update(name="", since=0.0, started=False)
            raise

    intro = (f"Finding the {name}:" + (f" last noted {last}." if last else "") + " I look, then turn a full circle in"
             f" {SWEEP_STEP_DEG:.0f}-degree steps looking after each, then roam: the most open direction that leads"
             f" somewhere new, up to {LEG_M:g} m at a time, for up to {max_s:.0f} s in all. Anything in the way stops a leg."
             + (" When I find it I walk up to it." if approach else "") + " You will be told the result." + cannot)
    return with_note(f"{note} {intro}", await robot.queue(f"find {name}", job, max_s, then=then))


# ============================================================ what Gemini reads
@tool(
    "find",
    "Find a thing and walk up to it: 'where is my cup', 'find the ball', 'look for the white container',"
    " 'look around for the whiteboard with code on it', 'go find Devesh'. First Pupper looks (its detector"
    " for everyday things, Gemma for anything you can name, and its memory of where things were), then"
    " turns a full circle in steps looking after each, then roams the room, stopping for anything in the"
    " way, for up to max_seconds. Found: you are told where (a [search result]) and Pupper walks up to it."
    " Not found: you are told that too. Only when a person asks. For a thing you can already see, go_to is"
    " quicker." + " If follow mode is on, this ends it: say so first.",
    what=Param("string", "What to find, in a few words: 'the red mug', 'a chair', 'the white container', 'a person'."),
    max_seconds=Param("number", f"How long at most, 5 to {MAX_S_LIMIT:g} s (default {MAX_S_DEFAULT:g}); a turn in place is about 20 s of it.", required=False),
)
async def find_tool(robot: Robot, what: str, max_seconds: float = MAX_S_DEFAULT) -> Result:
    return await find(robot, what, max_seconds)
