"""Part 2: Walk. TODO 2 (check_move), TODO 3 (the turn and move tools), then TODO 4 (arc).

Read it top to bottom: check_move (yours) decides whether a move is allowed;
walk (finished: read it) does a move on the robot; turn, move and arc (yours)
are what Gemini calls, and they call walk.

    "walk forward" -> move (TODO 3) -> walk -> check_move (TODO 2) -> the job, later
"""

from __future__ import annotations

import math

from basics import ENDS_FOLLOW
from hackathon_alerts import EVENTS
from hackathon_vision import clearer_side, path_blocked  # hackathon: the on-device detector's "something is in the way"
from part3_follow import prepare_to_move
from robot_api import FELL_MESSAGE, LIMITS, LLM_CMD_RATE, LOCOMOTION, LOCOMOTION_CONTROLLER, MoveLimits, Result, Robot, not_written, with_note  # noqa: F401
from tool_registry import Param, tool

# "Run" means at least this fast, in m/s (the locomotion policy switches to its
# running gait above about 0.5 m/s).
RUN_MIN_SPEED = 1.0
RUN_DEFAULT_SPEED = 1.2
# The default walking and turning speeds come from robot.speeds(): 0.4 m/s and
# 60 deg/s in locomotion mode, and a walking trick's own (the spider's are slower).


# ============================================================ what the robot does
def check_move(vx: float, vy: float, wz_deg: float, duration: float, limits: MoveLimits = LIMITS) -> str | None:
    """None if the robot may make this move, else the reason it may not.

    vx forward m/s, vy left m/s, wz_deg counter-clockwise deg/s, duration in s.

    TODO 2 (Part 2) ------------------------------------------------------------
    Gemini reads:        the string you return, word for word, as the tool's
                         reply. "too fast" leaves Gemini guessing; "too fast:
                         |vx|<=1.5 m/s" lets it retry at a speed that works.
    Must always be true: refuse anything too fast, too slow to move at all, a
                         duration that isn't more than 0 and at most
                         duration_max, and a move that would travel farther
                         than distance_max. Look at every field of MoveLimits
                         (robot/robot_api.py).
    Why here:            the tool descriptions ask Gemini to stay within the
                         limits; this makes sure of it.
    Tests check:         - good moves (even exactly at a limit) -> None
                         - each bad move -> a non-empty reason
                         - the reason contains the limit's number, as Python
                           prints it: vx_max "1.5", vy_max "0.5", wz_max
                           "114.0", duration_max "20.0", distance_max "7.0".
                           Passes: f"... at most {limits.duration_max} s"
                           Fails:  "too long", or "{limits.duration_max:.0f}" ("20")
    Test:                python3 lab/test.py todo2
    """
    # ========== TODO 2: YOUR CODE HERE ==========
    if abs(vx) > limits.vx_max or abs(vy) > limits.vy_max or abs(wz_deg) > limits.wz_max:
        return (f"too fast: limits are |vx|<={limits.vx_max} m/s, |vy|<={limits.vy_max} m/s,"
                f" |wz|<={limits.wz_max} deg/s (asked vx={vx:g}, vy={vy:g}, wz={wz_deg:.3g})")
    if abs(vx) < limits.vx_min and abs(vy) < limits.vy_min and abs(wz_deg) < limits.wz_min:
        return (f"too slow to move: use at least |vx|>={limits.vx_min} m/s, |vy|>={limits.vy_min} m/s,"
                f" or |wz|>={limits.wz_min} deg/s")
    if not 0 < duration <= limits.duration_max:
        return f"bad duration {duration:g} s: it must be more than 0 and at most {limits.duration_max} s"
    distance = (vx**2 + vy**2) ** 0.5 * duration
    if distance > limits.distance_max:
        return (f"too far: {distance:.2f} m in one move; at most {limits.distance_max} m."
                f" Use a shorter duration or several moves")
    return None
    # ====================================


async def walk(robot: Robot, vx: float, vy: float, wz_deg: float, duration: float) -> Result:
    """Walk at (vx, vy, wz_deg) for duration seconds, then stand still (finished: read it).

    vx forward m/s, vy left m/s, wz_deg counter-clockwise deg/s. Your turn and
    move tools (TODO 3) call this. It is stand_up's shape (basics.py), with a
    loop in the job. Two things to notice:

    - Why send the speed again and again? The locomotion policy doesn't take
      "walk for 2 seconds"; it takes a speed, over and over. In between sits
      cmd_vel_mux (the node that lets the joystick win over Gemini), and it
      forgets a speed it hasn't heard for 0.5 s, so a crashed program can't
      leave Pupper walking forever. So the job sends it LLM_CMD_RATE (20) times
      a second for the whole walk, then a zero to stand still.
    - Why check robot.fallen three times? The robot may already have fallen when
      Gemini asks (1), may fall while this job waits in line (2), or while it
      walks (3). Each time, Gemini should hear FELL_MESSAGE, not "moved".
    - Who walks, and how fast? Usually the locomotion policy. But in a trick
      that walks in its own stance (the spider, Part 6), its own walking policy
      does, with its own limits (robot.speeds(): from its trick.yaml), so a move
      that is too fast for it is refused with its numbers. Both depend on the
      state Pupper will be in when the job runs: robot.state_after_queue, the
      same now-question as Part 5.
    """
    # NOW, before anything is queued:
    speeds = robot.speeds()  # the limits where Pupper will be (a walking trick has its own)
    refused = check_move(vx, vy, wz_deg, duration, speeds.limits)  # your TODO 2: is this move allowed?
    if refused:
        return Result(False, refused + speeds.where)  # Gemini hears why; nothing is queued
    blocked = path_blocked(vx, vy)  # hackathon: something close in that direction? (hackathon_vision.py)
    if blocked:
        return Result(False, blocked + speeds.where)
    note = await prepare_to_move(robot)  # get ready: does nothing until your TODO 5 (Part 3)
    state = robot.state_after_queue  # where Pupper will be when this job runs
    walker = robot.walking_controller(state)  # the locomotion policy, or a walking trick's own
    then = state if walker != LOCOMOTION_CONTROLLER else LOCOMOTION  # a walking trick stays in its stance
    if robot.fallen:  # (1)
        return Result(False, FELL_MESSAGE)

    async def job() -> Result:  # LATER, when the queue gets to it
        if robot.fallen:  # (2)
            return Result(False, FELL_MESSAGE)
        res = await robot.switch_to(walker)  # the policy that walks takes the legs (if it hasn't already)
        if not res.ok:
            return res
        end = robot.now() + duration
        detours = 0
        while robot.now() < end - 1e-6:  # keep the mux input alive (1e-6: no extra step from rounding)
            if robot.fallen:  # (3)
                robot.send_speed(0, 0, 0)
                return Result(False, FELL_MESSAGE)
            blocked = path_blocked(vx, vy)  # hackathon: something stepped in the way mid-walk
            if blocked and vx > 0 and vy == 0 and detours < MAX_DETOURS:  # hackathon: go round it, not stop
                t0 = robot.now()
                went_round = await detour(robot, vx, wz_deg, speeds.turn)
                end += robot.now() - t0  # a detour doesn't eat the move's own time
                if went_round:
                    detours += 1
                    continue
            if blocked:
                robot.send_speed(0, 0, 0)
                walked = duration - (end - robot.now())
                EVENTS.emit("obstacle", f"stopped a move after {walked:.1f} s: {blocked}", for_gemini=(
                    f"[robot, not the person speaking] You stopped walking early, after {walked:.1f} s of"
                    f" {duration:.1f}: {blocked}. Tell the person in one sentence and wait for them."))
                return Result(False, f"stopped after {walked:.1f} s of {duration:.1f} s: {blocked}")
            robot.send_speed(vx, vy, wz_deg)
            await robot.sleep(1.0 / LLM_CMD_RATE)  # never time.sleep: it would freeze the server
        robot.send_speed(0, 0, 0)  # stand still
        return Result(True, f"moved {what} for {duration:.2f} s" + (f", going round {detours} obstacle(s)" if detours else ""))  # the outcome (debug log)

    # What Gemini hears NOW: the job was accepted.
    what = f"vx={vx:g} vy={vy:g} wz={wz_deg:.3g}"  # 45.836623610465864 -> 45.8: Gemini reads this
    return with_note(note, await robot.queue(f"move {what} {duration:.3g}s", job, duration, then=then))



# hackathon: automatic obstacle avoidance. Something ahead on a straight walk forward: turn toward
# the more open side until the way ahead is clear (at most DETOUR_MAX_DEG), walk PASS_M past it,
# turn back by the same angle, and carry on. Every step keeps the same checks as the walk, so
# anything still in the way ends in the usual stop. The camera can't see beside the robot, so
# PASS_M is a guess at "past it", not a measurement.
MAX_DETOURS = 3
DETOUR_MAX_DEG = 90.0
PASS_M = 0.5


async def _spin(robot: Robot, wz: float, until_clear: float | None, max_deg: float) -> float:
    """Turn at wz deg/s up to max_deg; with until_clear (a vx), stop once that way is clear.
    Returns the degrees turned (signed); -1e9 if the robot fell."""
    turned, dt = 0.0, 1.0 / LLM_CMD_RATE
    while abs(turned) < max_deg - 1e-6:
        if robot.fallen:
            return -1e9
        if until_clear is not None and not path_blocked(until_clear, 0):
            break
        robot.send_speed(0, 0, wz)
        await robot.sleep(dt)
        turned += wz * dt
    robot.send_speed(0, 0, 0)
    return turned


async def detour(robot: Robot, vx: float, wz_deg: float, turn_speed: float) -> bool:
    """Go round something ahead. True if past it and facing its first heading again; False to stop."""
    side = clearer_side()
    if side is None:
        return False
    wz = turn_speed if side == "left" else -turn_speed
    turned = await _spin(robot, wz, vx, DETOUR_MAX_DEG)
    if turned < -1e8:
        return False
    if path_blocked(vx, 0):  # no way round within the turn: face the first heading again, then stop
        await _spin(robot, -wz, None, abs(turned))
        return False
    end = robot.now() + PASS_M / vx
    while robot.now() < end - 1e-6:
        if robot.fallen or path_blocked(vx, 0):
            robot.send_speed(0, 0, 0)
            return False
        robot.send_speed(vx, 0, wz_deg)
        await robot.sleep(1.0 / LLM_CMD_RATE)
    robot.send_speed(0, 0, 0)
    back = await _spin(robot, -wz, None, abs(turned))
    if back < -1e8:
        return False
    EVENTS.emit("obstacle", f"went {side} round something in the way", for_gemini="")
    return True

# ============================================================ what Gemini reads
# TODO 3: write turn first, then move: what Gemini reads (the description and every
# parameter's description) and their bodies.
#   - turn is the warm-up: one number to work out (which way), one call. It shows
#     you the whole path: Gemini's words -> numbers -> walk -> the robot.
#   - move is the real challenge: four parameters, defaults, and the "run" rule.
#   - Put the robot's real limits in the words: LIMITS.vx_max and the others
#     (robot/robot_api.py), RUN_MIN_SPEED, and the defaults. Add ENDS_FOLLOW.
#   Tests check:
#   - each tool's description is 20+ characters, and every parameter is described
#   - turn left sends +wz, right -wz, about the angle asked for, with vx = vy = 0
#   - move's words contain the number LIMITS.vx_max ("1.5") and the word "follow"
#     (from ENDS_FOLLOW). Passes: f"... up to {LIMITS.vx_max} m/s" + ENDS_FOLLOW
#   - the move examples below send exactly those speeds
#   - move("right", run=True) says it walked: the reply says "walk".
#     Passes: "Sideways never runs; walking instead. move vx=0 vy=-0.4 ... queued"
#     Fails:  "move vx=0 vy=-0.4 ... queued" (Gemini then tells them it's running)
#   - a move check_move refuses comes back as a failure with check_move's reason
#   Test: python3 lab/test.py todo3  (turn is checked first, on its own)
# ========== TODO 3: YOUR CODE HERE ==========
@tool(
    "turn",
    "Turn in place, left (counter-clockwise) or right, by an angle in degrees, at about 60 deg/s."
    " Default 90 deg; 180 turns around; 360 is a full spin. Example: 'turn around' ->"
    " turn(direction='right', angle=180)." + ENDS_FOLLOW,
    direction=Param("string", "Which way to turn: left (counter-clockwise) or right.", enum=["left", "right"]),
    angle=Param("number", "How far to turn, in degrees (positive). Default 90.", required=False),
)
async def turn_tool(robot: Robot, direction: str, angle: float = 90.0) -> Result:
    turn_speed = robot.speeds().turn
    wz_deg = turn_speed if direction == "left" else -turn_speed
    seconds = abs(angle) / turn_speed
    return await walk(robot, 0.0, 0.0, wz_deg, seconds)


@tool(
    "move",
    "Walk (or run) straight forward, backward, or sideways left/right for a while, then stand still."
    f" Walking defaults to 0.4 m/s for 1.5 s. Limits: forward/backward up to {LIMITS.vx_max} m/s,"
    f" sideways up to {LIMITS.vy_max} m/s, at most {LIMITS.duration_max} s and {LIMITS.distance_max} m"
    f" in one move. run=true means at least {RUN_MIN_SPEED} m/s (default {RUN_DEFAULT_SPEED} m/s),"
    " forward or backward only: sideways never runs. Examples: 'walk forward a bit' ->"
    " move(direction='forward'); 'run back for 2 seconds' -> move(direction='backward', run=true,"
    " duration=2); 'step left slowly' -> move(direction='left', speed=0.25)." + ENDS_FOLLOW,
    direction=Param("string", "Which way: forward, backward, left (sideways) or right (sideways).",
                    enum=["forward", "backward", "left", "right"]),
    run=Param("boolean", f"true to run (at least {RUN_MIN_SPEED} m/s; forward/backward only). Default false.",
              required=False),
    speed=Param("number", f"Speed in m/s. Default 0.4 walking, {RUN_DEFAULT_SPEED} running."
                f" Max {LIMITS.vx_max} forward/backward, {LIMITS.vy_max} sideways.", required=False),
    duration=Param("number", f"How long to move, in seconds. Default 1.5, max {LIMITS.duration_max}.",
                   required=False),
)
async def move_tool(
    robot: Robot,
    direction: str,
    run: bool = False,
    speed: float | None = None,
    duration: float = 1.5,
) -> Result:
    # Gemini may leave out run, speed and duration (then they have the defaults above:
    # speed is None when Gemini gave none). Work out ONE speed, then split it into vx and vy.
    # 1. The speed:
    #    - running (run is True AND the direction is forward or backward): at least
    #      RUN_MIN_SPEED; if Gemini gave no speed, RUN_DEFAULT_SPEED.
    #      (max(a, b) is the bigger of two numbers: max(0.6, RUN_MIN_SPEED) is 1.0.)
    #    - otherwise (walking, or any sideways move): speed, or robot.speeds().walk if
    #      Gemini gave none (0.4 m/s walking; a walking trick has its own).
    # 2. Split it by direction (the robot's convention: +x forward, +y LEFT):
    #        forward: vx = +speed, vy = 0      backward: vx = -speed, vy = 0
    #        left:    vx = 0, vy = +speed      right:    vx = 0, vy = -speed
    # 3. If Gemini asked to run sideways, you walk instead: say so, or Gemini will
    #    tell the person it's running. Start the reply with a note:
    #      note = "Sideways never runs; walking instead." (or "" otherwise)
    #      return with_note(note, await walk(robot, vx, vy, 0.0, duration))
    straight = direction in ("forward", "backward")
    if run and straight:
        speed = max(speed if speed is not None else RUN_DEFAULT_SPEED, RUN_MIN_SPEED)
    elif speed is None:
        speed = robot.speeds().walk
    vx, vy = {"forward": (speed, 0.0), "backward": (-speed, 0.0),
              "left": (0.0, speed), "right": (0.0, -speed)}[direction]
    note = "Sideways never runs; walking instead." if run and not straight else ""
    return with_note(note, await walk(robot, vx, vy, 0.0, duration))
# ====================================


# TODO 4: the arc tool: walk forward and turn at the same time, along part of a
# circle. turn sends only a turning speed and move only a straight one, so
# "go in a circle" was move, stop, turn, move. walk takes vx and wz_deg together,
# and the locomotion policy can do both at once: arc is how Gemini asks for it.
#   Now:                 work out vx and wz_deg from the radius and the speed, then
#                        walk (walk checks the move with check_move and queues it).
#   Gemini reads:        the words, to pick a radius the robot can do; then the
#                        reply: walk's, or why there is no arc (a negative radius,
#                        a speed of 0 or less, or too tight a circle for that
#                        speed), or check_move's reason (too long, too far).
#   Must always be true: an arc the robot can't do never moves it and never crashes.
#   Radius 0 is a sharp corner, not an error: Gemini plans a whole shape and sends
#   every call at once, and it often asks for corners as arc(radius=0). Refuse
#   them, and the sides still run: a rectangle becomes a straight line. So radius
#   0 turns in place (like turn), and the reply says so.
#   The math: around a circle of radius r (m) at speed v (m/s),
#     wz_deg = math.degrees(v / r)   (+ to the left, - to the right)
#     seconds = abs(angle) / abs(wz_deg)
#     e.g. 90 deg at 0.4 m/s around r = 0.5 m: wz = 45.8 deg/s for 1.96 s (0.79 m)
#   Put the real limits in the words: the robot turns at most LIMITS.wz_max deg/s,
#   so at speed v the radius must be at least v / math.radians(LIMITS.wz_max); one
#   arc goes at most LIMITS.distance_max m. Add ENDS_FOLLOW.
#   Tests check:
#   - arc's description is 20+ characters, and every parameter is described
#   - its words contain LIMITS.distance_max ("7.0") and "follow" (from ENDS_FOLLOW).
#     Passes: f"... One arc goes at most {LIMITS.distance_max} m." + ENDS_FOLLOW
#     Fails:  "Walk in a curve." (Gemini then asks for circles the robot refuses)
#   - arc("left", angle=90, radius=0.5) sends vx = +0.4, vy = 0 and wz = +45.8 deg/s,
#     and turns about 90 deg; arc("right", ...) sends a negative wz, still forward
#   - arc("left", speed=0.6, radius=1.0) sends vx = 0.6 and wz = +34.4 deg/s
#   - arc("left", angle=90, radius=0) turns in place, left, about 90 deg, and the
#     reply says "place".
#     Passes: "Radius 0: turned in place (a sharp corner). move vx=0 ... queued"
#     Fails:  a crash (division by zero), or a refusal
#   - arc("left", radius=-1) is a failure that says "radius", and sends nothing
#   - arc("left", radius=0.1) is too tight (229 deg/s): a failure that says "radius"
#     and gives the smallest radius that works at that speed (0.2-0.3 m at 0.4 m/s).
#     Passes: "too tight: at 0.4 m/s the radius must be at least 0.25 m"
#     Fails:  check_move's "too fast: ... |wz|<=114.0 deg/s" (Gemini can't tell the
#             radius is the problem, and sends the same arc again), or a success
#             that quietly turned slower
#   Test: python3 lab/test.py todo4
# ========== TODO 4: YOUR CODE HERE ==========
@tool(
    "arc",
    "Walk forward while turning, along part of a circle: angle 90 is a quarter circle, 360 a full"
    " circle. radius in meters (default 0.5), speed in m/s (default 0.4). The robot turns at most"
    f" {LIMITS.wz_max} deg/s, so at 0.4 m/s the radius must be at least"
    f" {math.ceil(0.4 / math.radians(LIMITS.wz_max) * 100) / 100} m. One arc goes at most"
    f" {LIMITS.distance_max} m and {LIMITS.duration_max} s. radius 0 is a sharp corner (turns in place)."
    " Shapes with straight sides (square, triangle) need move for the sides and turn for the corners."
    " Examples: 'walk in a circle' -> arc(direction='left', angle=360); 'curve right' ->"
    " arc(direction='right', angle=90, radius=1.0)." + ENDS_FOLLOW,
    direction=Param("string", "Which way to curve: left (counter-clockwise) or right.", enum=["left", "right"]),
    angle=Param("number", "How much of the circle to walk, in degrees: 90 quarter, 180 half, 360 full."),
    radius=Param("number", "Circle radius in meters, 0 or more. Default 0.5; 0 turns in place.", required=False),
    speed=Param("number", "Walking speed along the arc in m/s, more than 0. Default 0.4.", required=False),
)
async def arc_tool(
    robot: Robot,
    direction: str,
    angle: float,
    radius: float = 0.5,
    speed: float | None = None,
) -> Result:
    # speed is None when Gemini gave none: use robot.speeds().walk (0.4 m/s walking).
    # 1. No circle to follow: if radius < 0 or speed <= 0, return Result(False, ...)
    #    saying which, and what works: "radius must be 0 m or more".
    #    Radius 0 is a sharp corner: turn in place like your turn tool (+-turn_speed,
    #    for abs(angle) / turn_speed s), and say so in front of walk's reply:
    #      return with_note("Radius 0: turned in place (a sharp corner).", await walk(...))
    # 2. Too tight: if math.degrees(speed / radius) > wz_max, return a failure
    #    with the smallest radius that works: speed / math.radians(wz_max), where
    #    wz_max = robot.speeds().limits.wz_max (LIMITS.wz_max walking).
    #    (check_move would refuse it too, but "|wz|<=114.0" doesn't tell Gemini that
    #    the radius is the problem.) Round up: f"{math.ceil(r * 100) / 100} m"
    # 3. wz_deg = math.degrees(speed / radius): + to the left, - to the right.
    # 4. seconds = abs(angle) / abs(wz_deg)
    # 5. return await walk(robot, speed, 0.0, wz_deg, seconds)  (walk checks it)
    # angle has no default: Gemini must say how much of the circle (left out, it
    # walked a quarter when asked for a circle). The tool registry refuses a call without it.
    speeds = robot.speeds()
    if speed is None:
        speed = speeds.walk
    if radius < 0:
        return Result(False, f"no arc: radius {radius:g} m is negative; radius must be 0 m or more")
    if speed <= 0:
        return Result(False, f"no arc: speed {speed:g} m/s; speed must be more than 0 m/s")
    sign = 1 if direction == "left" else -1
    if radius == 0:
        seconds = abs(angle) / speeds.turn
        return with_note("Radius 0: turned in place (a sharp corner).",
                         await walk(robot, 0.0, 0.0, sign * speeds.turn, seconds))
    wz_max = speeds.limits.wz_max
    if math.degrees(speed / radius) > wz_max:
        min_radius = math.ceil(speed / math.radians(wz_max) * 100) / 100
        return Result(False, f"too tight: at {speed:g} m/s the radius must be at least {min_radius} m"
                             f" (the robot turns at most {wz_max} deg/s); asked {radius:g} m")
    wz_deg = sign * math.degrees(speed / radius)
    seconds = abs(angle) / abs(wz_deg)
    return await walk(robot, speed, 0.0, wz_deg, seconds)
# ====================================
