#!/usr/bin/env python3
"""Desk tests for Gemini Pupper: no robot, no ROS, no Gemini needed.

Run them after every change to a file in lab/ (lab/test.py runs this file):

    python3 lab/test.py            # everything
    python3 lab/test.py todo3      # one TODO (todo1 ... todo11)
    python3 lab/test.py part2      # one part of the lab (part1 ... part7; part6, the spider, is the notebook)
    python3 lab/test.py robot      # the provided robot (should always pass)
    python3 lab/test.py optional_jump   # the optional jump

A fake robot stands in for ROS. It keeps track of which controllers are on,
records every message the robot sends, and plays trick clips instantly. Time
is fake too, so a 10 s walk takes no real time.

The tests check what each tool does and what Gemini is told, not how you wrote
the code. When a test fails, it prints what was asked and what happened.

Passing these tests does not mean the robot is ready. That is what the real
Pupper, the e-stop and a TA are for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ----------------------------------------------------------------------------- fake ROS
class _Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Quat:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x, self.y, self.z, self.w = x, y, z, w


class _Twist:
    def __init__(self):
        self.linear, self.angular = _Vec(), _Vec()


class _DataMsg:
    def __init__(self, data=None):
        self.data = data


class _Srv:
    Request = _Msg


class _SwitchController:
    class Request(_Msg):
        STRICT = 2
        BEST_EFFORT = 1

        def __init__(self):
            super().__init__(activate_controllers=[], deactivate_controllers=[], strictness=1)


class FakeRobot:
    """The controller manager, the topics and the person follower, all in one."""

    def __init__(self, clock: "VirtualClock"):
        self.clock = clock
        self.controllers: dict[str, str] = {}
        self.published: list[tuple[float, str, object]] = []  # (time, topic, msg)
        self.switches: list[tuple[float, list[str], list[str]]] = []  # (time, deactivated, activated)
        self.subs: dict[str, list] = {}
        self.follower_on = False
        self.follow_calls: list[str] = []
        self.start_clips: dict[str, list[str]] = {}  # controller -> start_clip values set, in order

    def active(self) -> set[str]:
        return {n for n, s in self.controllers.items() if s == "active"}

    def twists(self) -> list[tuple[float, _Twist]]:
        return [(t, m) for t, topic, m in self.published if topic == "/llm_cmd_vel"]

    def clips(self, controller: str) -> list[str]:
        return [m.data for _, topic, m in self.published if topic == f"/{controller}/clip"]

    def activations(self, controller: str) -> list[float]:
        return [t for t, _, on in self.switches if controller in on]

    def deactivations(self, controller: str) -> list[float]:
        return [t for t, off, _ in self.switches if controller in off]

    def send(self, topic: str, msg) -> None:
        """Deliver a message to the robot's subscribers."""
        for cb in self.subs.get(topic, []):
            cb(msg)

    def handle(self, service: str, req):
        if service.endswith("list_controllers"):
            return _Msg(controller=[_Msg(name=n, state=s) for n, s in self.controllers.items()])
        if service.endswith("switch_controller"):
            for c in req.deactivate_controllers:
                self.controllers[c] = "inactive"
            for c in req.activate_controllers:
                self.controllers[c] = "active"
            self.switches.append((self.clock.now, list(req.deactivate_controllers), list(req.activate_controllers)))
            return _Msg(ok=True)
        if service.endswith("/set_parameters"):
            controller = service.split("/")[1]
            for prm in req.parameters:
                if prm.name == "start_clip":
                    self.start_clips.setdefault(controller, []).append(prm.value.string_value)
            return _Msg(results=[_Msg(successful=True)])
        if service == "activate_person_following":
            self.follower_on = True
            self.follow_calls.append("on")
            return _Msg(success=True)
        if service == "deactivate_person_following":
            self.follower_on = False
            self.follow_calls.append("off")
            return _Msg(success=True)
        raise AssertionError(f"unexpected service {service}")

    def publish(self, topic: str, msg) -> None:
        self.published.append((self.clock.now, topic, msg))
        if topic.endswith("/clip"):
            controller = topic.split("/")[1]
            if self.controllers.get(controller) == "active":  # a stopped controller hears nothing
                self.send(f"/{controller}/clip_status", _DataMsg(f"{msg.data} holding frame 0 queued 0"))


ROBOT: FakeRobot | None = None


class _Future:
    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        return self._result


class _Client:
    def __init__(self, name):
        self.name = name

    def wait_for_service(self, timeout_sec=None):
        return True

    def call_async(self, req):
        return _Future(ROBOT.handle(self.name, req))


class _Pub:
    def __init__(self, topic):
        self.topic = topic

    def publish(self, msg):
        ROBOT.publish(self.topic, msg)


class _Node:
    def __init__(self, name):
        pass

    def create_client(self, srv_type, name):
        return _Client(name)

    def create_publisher(self, msg_type, topic, qos):
        return _Pub(topic)

    def create_subscription(self, msg_type, topic, cb, qos):
        ROBOT.subs.setdefault(topic, []).append(cb)


class _Executor:
    def add_node(self, node):
        pass

    def spin(self):
        pass


def _install_fake_ros() -> None:
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m

    mod("rclpy", ok=lambda: True, init=lambda: None)
    mod("rclpy.node", Node=_Node)
    mod("rclpy.executors", SingleThreadedExecutor=_Executor)
    mod("controller_manager_msgs")
    mod("controller_manager_msgs.srv", ListControllers=_Srv, SwitchController=_SwitchController)
    mod("geometry_msgs")
    mod("geometry_msgs.msg", Twist=_Twist)
    mod("rcl_interfaces")
    mod("rcl_interfaces.msg", Log=_Msg, Parameter=_Msg, ParameterValue=_Msg,
        ParameterType=types.SimpleNamespace(PARAMETER_STRING=4))
    mod("rcl_interfaces.srv", SetParameters=_Srv)
    mod("sensor_msgs")
    mod("sensor_msgs.msg", CompressedImage=_Msg, Imu=_Msg, JointState=_Msg)
    mod("std_msgs")
    mod("std_msgs.msg", Empty=_DataMsg, Float32MultiArray=_DataMsg, Float64MultiArray=_DataMsg, String=_DataMsg)
    mod("std_srvs")
    mod("std_srvs.srv", Trigger=_Srv)


_install_fake_ros()
import robot_api  # noqa: E402
import tool_registry as tools  # noqa: E402  (call and declarations: what the web app uses)
behaviors = tools.load_lab()  # noqa: E402  (everything in lab/: lab.walk, lab.check_move, ...)
check_move = behaviors.check_move
from robot_api import LIMITS, LLM_CMD_RATE, LOCOMOTION, LOCOMOTION_CONTROLLER, WALK_READY_S, Robot  # noqa: E402
from tricks import Trick, WalkConfig  # noqa: E402

_real_sleep = asyncio.sleep


class VirtualClock:
    """The robot's time.monotonic() and asyncio.sleep(), with no real waiting."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    async def sleep(self, seconds, result=None):
        self.now += max(0.0, float(seconds))
        await _real_sleep(0)
        return result


class _AsyncioShim(types.ModuleType):
    def __init__(self, clock):
        super().__init__("asyncio")
        self._clock = clock

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def sleep(self, seconds, result=None):
        return await self._clock.sleep(seconds, result)


# ----------------------------------------------------------------------------- fixtures
def _clip(name, start, end, loop=False, next=None):
    return {"name": name, "start": start, "end": end, "loop": loop, "next": next}


_REFERENCE = Path(tempfile.gettempdir()) / "gemini_pupper_test_hop.json"


def fixture_tricks() -> dict[str, Trick]:
    """Made-up tricks: a policy that ends standing, one that ends lying down, a clip
    trick that holds a pose, and a reference trick. Clip lengths are in frames at 50 fps."""
    _REFERENCE.write_text(json.dumps({"playback_s": 1.0, "active_s": 0.8, "cross_fade_s": 0.1,
                                      "table": [[0.0] * 12] * 3}))
    return {
        "bow": Trick(name="bow", description="Bows.", kind="policy", path=Path("bow.json"), duration=4.0),
        "roll_over": Trick(name="roll_over", description="Rolls over and stays lying down.", kind="policy",
                           path=Path("roll_over.json"), duration=6.0, start_pose="lying", end_pose="lying"),
        "hold_pose": Trick(
            name="hold_pose", description="Gets into a pose and holds it.", kind="policy",
            path=Path("hold_pose.json"), duration=12.0, end_pose="sitting",
            clips=[
                _clip("enter", 0, 150, next="idle"),
                _clip("idle", 149, 200, loop=True),
                _clip("wave_left", 199, 300, next="idle"),
                _clip("wave_right", 299, 400, next="idle"),
                _clip("leave", 399, 550),
            ],
            actions={"wave_left": "Waves the left paw.", "wave_right": "Waves the right paw.",
                     "leave": "Stands back up."},
            exit_clip="leave",
        ),
        "hop": Trick(name="hop", description="Hops in place.", kind="reference", path=_REFERENCE, duration=1.0),
    }


class Rig:
    """A fresh Robot wired to a fresh fake robot, standing in locomotion mode."""

    def __init__(self, tricks=None):
        global ROBOT
        self.clock = VirtualClock()
        ROBOT = self.robot = FakeRobot(self.clock)
        made = tricks or fixture_tricks
        robot_api.time = types.SimpleNamespace(monotonic=self.clock.monotonic)
        robot_api.asyncio = _AsyncioShim(self.clock)
        robot_api.load_tricks = made
        names = (set(robot_api.LEG_CONTROLLERS) | {t.controller for t in made().values()}
                 | {t.walk_controller for t in made().values() if t.walks})
        self.robot.controllers = {n: "inactive" for n in names}
        self.robot.controllers[LOCOMOTION_CONTROLLER] = "active"
        self.records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = self.records.append
        for name in ("robot_api", "tools", "behaviors"):
            lg = logging.getLogger(name)
            lg.handlers[:] = [handler]
            lg.setLevel(logging.INFO)
            lg.propagate = False
        self.ts = Robot()

    async def settle(self, limit: int = 200_000) -> bool:
        """Let the queue run until it is empty. False if it never empties."""
        for _ in range(limit):
            await _real_sleep(0)
            if self.ts.commands.empty() and self.ts.current is None:
                await _real_sleep(0)
                if self.ts.commands.empty() and self.ts.current is None:
                    return True
        return False

    async def call(self, name: str, **args):
        """Call a tool the way Gemini does, and let the queue run it."""
        res = await tools.call(self.ts, name, args)
        await self.settle()
        return res

    def outcomes(self, start: int = 0) -> list[str]:
        """What each queued job ended with, as the queue logged it."""
        out = []
        for r in self.records[start:]:
            msg = r.getMessage()
            if r.exc_info:
                exc = r.exc_info[1]
                msg += f" -- raised {type(exc).__name__}: {exc}"
            out.append(msg)
        return out

    def failures(self, start: int = 0) -> list[str]:
        return [r.getMessage() for r in self.records[start:] if r.levelno >= logging.WARNING]

    def imu(self, tilt_deg: float, spin: float = 0.0, seconds: float = 1.5) -> None:
        """Feed IMU readings: tilted tilt_deg about x, turning at spin rad/s."""
        h = math.radians(tilt_deg) / 2
        for _ in range(int(seconds * 100)):
            self.robot.send("/imu_sensor_broadcaster/imu", _Msg(
                orientation=_Quat(x=math.sin(h), w=math.cos(h)), angular_velocity=_Vec(x=spin)))
            self.clock.now += 0.01

    def close(self):
        self.ts.worker.cancel()


# ----------------------------------------------------------------------------- checks
PASS = FAIL = TODO = 0


def check(name: str, ok: bool, detail: str = "", passes: str = "") -> bool:
    """One check. On a failure, print what we got (detail) and an example that passes."""
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f": {detail}" if detail else ""))
        if passes:
            print(f"         passes, e.g.: {passes}")
    return ok


def unwritten(res) -> str | None:
    """'TODO n' if a result (or a check_move reason) says that TODO isn't written yet."""
    text = res.message if hasattr(res, "message") else str(res or "")
    m = re.search(r"(TODO \d+) is not written yet", text)
    return m.group(1) if m else None


def skip(what: str) -> None:
    global TODO
    TODO += 1
    print(f"  --   {what}")


def walk_twists(rig: Rig):
    return [m for _, m in rig.robot.twists() if abs(m.linear.x) + abs(m.linear.y) + abs(m.angular.z) > 1e-9]


def described(rig: Rig, names: list[str], todo: str) -> bool:
    """Check the declarations of some tools. False (and a skip) if one still says TODO."""
    decls = {d["name"]: d for d in tools.declarations(rig.ts)}
    missing = [n for n in names if n not in decls]
    if missing:
        check(f"lab/ declares {', '.join(names)}", False, f"missing {missing}; declared: {sorted(decls)}")
        return False
    if any("TODO" in json.dumps(decls[n]) for n in names):
        skip(f"{todo}: the descriptions of {', '.join(names)} still say TODO")
        return False
    for n in names:
        d = decls[n]
        check(f"{n} has a real description (at least 20 characters)", len(d.get("description", "")) >= 20,
              f"description: {d.get('description')!r}",
              passes='"Turn in place to the left or right." (a bare "turn" fails)')
        params = d.get("parameters", {}).get("properties", {})
        no_doc = [p for p, s in params.items() if len(s.get("description", "")) < 3]
        check(f"... and every parameter of {n} is described", not no_doc, f"missing: {no_doc}",
              passes='angle=Param("number", "Degrees to turn, e.g. 90. Default 90.")')
    return True


def system_prompt() -> str:
    text = (HERE.parent / "lab" / "prompt.md").read_text()
    return re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()


# ----------------------------------------------------------------------------- Part 1
def prompt_sections(*headings: str) -> str:
    """The text under some of prompt.md's '# ' headings."""
    parts, keep = [], False
    for line in system_prompt().splitlines():
        if line.startswith("# "):
            keep = line[2:].strip().lower() in {h.lower() for h in headings}
        elif keep:
            parts.append(line)
    return "\n".join(parts).strip()


async def todo1() -> None:
    print("TODO 1: prompt.md, first pass")
    text = prompt_sections("Who you are", "How you move", "When a tool says no")
    if "TODO" in text or not text:
        skip("TODO 1: prompt.md still says TODO under its first three headings")
        return
    words = len(text.split())
    check("the prompt says who Pupper is and how it moves (at least 60 words in the three sections)", words >= 60,
          f"{words} words", passes="a few bullet points under each heading, about 60 words or more in all")
    check('it tells Gemini it moves only by calling tools (says "tool")', "tool" in text.lower(),
          passes='"You can only move by calling a tool." (fails: "You can walk and turn.")')


# ----------------------------------------------------------------------------- Part 2
async def todo2() -> None:
    print("TODO 2: check_move refuses what the robot can't or shouldn't do")
    probe = check_move(0.4, 0.0, 0.0, 2.0)
    if unwritten(probe):
        skip("TODO 2 (check_move) is not written yet")
        return
    L = LIMITS
    fine = [
        ("walk forward 0.4 m/s for 2 s", (0.4, 0.0, 0.0, 2.0)),
        ("walk backward 0.4 m/s for 2 s", (-0.4, 0.0, 0.0, 2.0)),
        ("sidestep 0.3 m/s", (0.0, 0.3, 0.0, 2.0)),
        ("turn in place 60 deg/s", (0.0, 0.0, 60.0, 1.5)),
        ("run 1.2 m/s for 4 s (4.8 m)", (1.2, 0.0, 0.0, 4.0)),
        ("the fastest allowed forward speed", (L.vx_max, 0.0, 0.0, 1.0)),
    ]
    for name, args in fine:
        got = check_move(*args)
        check(f"allows: {name}", got is None, f"check_move{args} refused it: {got!r}",
              passes="return None for a move inside every limit (exactly at a limit is still allowed)")
    bad = [
        ("forward faster than vx_max", (L.vx_max + 1.0, 0.0, 0.0, 1.0), str(L.vx_max)),
        ("backward faster than vx_max", (-(L.vx_max + 1.0), 0.0, 0.0, 1.0), str(L.vx_max)),
        ("sideways faster than vy_max", (0.0, L.vy_max + 0.3, 0.0, 1.0), str(L.vy_max)),
        ("turning faster than wz_max", (0.0, 0.0, L.wz_max + 50, 1.0), str(L.wz_max)),
        ("too slow to move at all", (0.05, 0.05, 5.0, 2.0), None),
        ("a zero duration", (0.4, 0.0, 0.0, 0.0), None),
        ("a negative duration", (0.4, 0.0, 0.0, -1.0), None),
        ("longer than duration_max", (0.4, 0.0, 0.0, L.duration_max + 1), str(L.duration_max)),
        ("farther than distance_max in one move", (1.5, 0.0, 0.0, 6.0), str(L.distance_max)),
    ]
    for name, args, limit in bad:
        got = check_move(*args)
        if check(f"refuses: {name}", isinstance(got, str) and got != "", f"check_move{args} returned {got!r}",
                 passes='return a sentence saying what is wrong, e.g. "duration must be more than 0"'):
            if limit is not None:
                check(f"  ... and the reason includes the limit {limit} (so Gemini knows what to try instead)",
                      limit in got, f"reason was {got!r}: add the limit's number",
                      passes=f'"... {limit} ..." from {{limits.<field>}} in an f-string; fails: a reason with no number'
                      + (f', or "{limit[:-2]}" ({limit} rounded)' if limit.endswith(".0") else ""))
    await provided_walk()


async def provided_walk() -> None:
    print("  (provided) walk, with your check_move: sends the speed for the whole move, then stands still")
    rig = Rig()
    try:
        res = await behaviors.walk(rig.ts, 5.0, 0.0, 0.0, 1.0)
        if unwritten(res):
            skip(f"{unwritten(res)} is not written yet")
            return
        check("an impossible speed comes back as a failure", res.ok is False, f"got {res}")
        check("... and Gemini hears your check_move reason, limit included (1.5)", "1.5" in res.message,
              f"got {res.message!r}", passes='"too fast: limits are |vx|<=1.5 m/s, ..." (fix it in check_move: TODO 2)')
        check("... and nothing was queued or published",
              rig.ts.commands.empty() and not rig.robot.twists(), f"published {len(rig.robot.twists())} commands")

        rig.robot.controllers[LOCOMOTION_CONTROLLER] = "inactive"  # lying down
        start = len(rig.records)
        res = await behaviors.walk(rig.ts, 0.4, 0.0, 0.0, 2.0)
        check("a good move is accepted right away (queued)", res.ok is True and "queued" in res.message, f"got {res}")
        if not await rig.settle():
            check("the queue finishes", False, "the move never ended")
            return
        crashed = [o for o in rig.outcomes(start) if "raised" in o]
        check("the job ran without crashing", not crashed, "; ".join(crashed))
        check("the locomotion policy took the legs", LOCOMOTION_CONTROLLER in rig.robot.active())
        twists = rig.robot.twists()
        moving = [m for _, m in twists if abs(m.linear.x) > 1e-6]
        check("it sent the forward speed", bool(moving) and all(abs(m.linear.x - 0.4) < 1e-6 for m in moving),
              f"forward speeds sent: {sorted({round(m.linear.x, 3) for m in moving})}")
        if twists:
            span = twists[-1][0] - twists[0][0]
            check("for about the requested 2 s", 1.8 <= span <= 2.5, f"commands spanned {span:.2f} s")
            gaps = [b[0] - a[0] for a, b in zip(twists, twists[1:])]
            check("often enough that cmd_vel_mux never drops it (every < 0.5 s)",
                  bool(gaps) and max(gaps) < 0.5, f"longest gap {max(gaps) if gaps else 'n/a'} s")
            last = twists[-1][1]
            check("and ends with a zero command (stand still)",
                  abs(last.linear.x) + abs(last.linear.y) + abs(last.angular.z) < 1e-9)
            check("at roughly LLM_CMD_RATE", len(twists) >= 0.8 * 2.0 * LLM_CMD_RATE, f"{len(twists)} commands in 2 s")
        check("the outcome (in the debug log) says what it did", any("moved" in o for o in rig.outcomes(start)),
              f"log: {rig.outcomes(start)}")
        rig.robot.published.clear()
        await behaviors.walk(rig.ts, 0.0, 0.0, 60.0, 1.0)
        await rig.settle()
        turns = [m for _, m in rig.robot.twists() if abs(m.angular.z) > 1e-6]
        check("a turn sends its turn rate in rad/s", bool(turns) and abs(turns[0].angular.z - 60 / 57.2958) < 1e-3,
              f"turn rates sent: {sorted({round(m.angular.z, 3) for m in turns})}")

        # Falling over in the middle of a walk.
        rig.robot.published.clear()
        start = len(rig.records)
        await behaviors.walk(rig.ts, 0.4, 0.0, 0.0, 4.0)
        t0 = rig.clock.now
        while rig.clock.now < t0 + 1.0:
            await _real_sleep(0)
        rig.ts.fallen = LOCOMOTION_CONTROLLER  # as if the tilt e-stop had just tripped
        await rig.settle()
        twists = rig.robot.twists()
        check("if the robot falls while walking, it stops sending the speed",
              bool(twists) and twists[-1][0] - t0 < 2.0, f"kept sending until {twists[-1][0] - t0 if twists else 0:.1f} s")
        check("... ends on a zero command", bool(twists) and abs(twists[-1][1].linear.x) < 1e-9)
        check("... and the outcome says it fell", any("fell" in o for o in rig.outcomes(start)),
              f"log: {rig.outcomes(start)}")
        res = await behaviors.walk(rig.ts, 0.4, 0.0, 0.0, 1.0)
        check("a walk asked for while fallen is refused right away, saying so",
              not res.ok and "fell" in res.message, f"got {res}")
    finally:
        rig.close()


async def todo3() -> None:
    print("TODO 3: the turn tool, then the move tool")
    rig = Rig()
    try:
        # turn first: the warm-up.
        described(rig, ["turn"], "TODO 3 (turn)")
        res = await rig.call("turn", direction="left", angle=90)
        if unwritten(res):
            skip(f"{unwritten(res)} is not written yet (the turn tool needs TODO 2 and TODO 3)")
            return
        twists = rig.robot.twists()
        yaw = sum(m.angular.z * (b[0] - a[0]) for (a, b), m in zip(zip(twists, twists[1:]), (t[1] for t in twists)))
        check("turn left turns counter-clockwise (+z)", any(m.angular.z > 0 for _, m in twists),
              f"result {res}; turn rates sent: {sorted({round(m.angular.z, 2) for _, m in twists})}",
              passes="walk(robot, 0.0, 0.0, +TURN_SPEED, ...) for left (fails: -TURN_SPEED)")
        check("... by about the angle asked for (90 deg)", 70 <= math.degrees(yaw) <= 110,
              f"turned about {math.degrees(yaw):.0f} deg", passes="seconds = abs(angle) / TURN_SPEED: 90 deg -> 1.5 s")
        check("... in place (no forward or sideways speed)",
              all(abs(m.linear.x) + abs(m.linear.y) < 1e-9 for _, m in twists), passes="vx = vy = 0.0")
        rig.robot.published.clear()
        await rig.call("turn", direction="right", angle=90)
        twists = rig.robot.twists()
        check("turn right turns clockwise (-z)", any(m.angular.z < 0 for _, m in twists),
              f"turn rates sent: {sorted({round(m.angular.z, 2) for _, m in twists})}", passes="wz_deg = -TURN_SPEED")

        # move: the real challenge.
        if described(rig, ["move"], "TODO 3 (move)"):
            text = json.dumps({d["name"]: d for d in tools.declarations(rig.ts)}["move"])
            check(f"move's words give the real speed limit (LIMITS.vx_max, {LIMITS.vx_max})", str(LIMITS.vx_max) in text,
                  passes='f"... running {RUN_MIN_SPEED}-{LIMITS.vx_max} m/s ..." (fails: "fast", or a typed-in 1.5)')
            check('... and say that it ends follow mode (says "follow")', "follow" in text.lower(),
                  passes='"Walk or run ..." + ENDS_FOLLOW (fails: leaving ENDS_FOLLOW out)')
        rig.robot.published.clear()
        res = await rig.call("move", direction="forward", duration=1.0)
        if unwritten(res):
            skip(f"{unwritten(res)} is not written yet (the move tool)")
            return

        async def sent(**args):
            rig.robot.published.clear()
            res = await rig.call("move", duration=1.0, **args)
            return res, walk_twists(rig)

        res, ts = await sent(direction="forward")
        check('move("forward") walks forward at the default 0.4 m/s',
              bool(ts) and all(abs(m.linear.x - 0.4) < 1e-6 and abs(m.linear.y) < 1e-9 for m in ts),
              f"result {res}; sent vx {sorted({round(m.linear.x, 2) for m in ts})}", passes="vx = 0.4, vy = 0")
        res, ts = await sent(direction="backward", speed=0.3)
        check('move("backward", speed=0.3) sends vx = -0.3', bool(ts) and all(abs(m.linear.x + 0.3) < 1e-6 for m in ts),
              f"sent vx {sorted({round(m.linear.x, 2) for m in ts})}", passes="vx = -speed (fails: +0.3)")
        res, ts = await sent(direction="left", speed=0.3)
        check('move("left", speed=0.3) sends vy = +0.3 (the robot\'s left is +y)',
              bool(ts) and all(abs(m.linear.y - 0.3) < 1e-6 and abs(m.linear.x) < 1e-9 for m in ts),
              f"sent vy {sorted({round(m.linear.y, 2) for m in ts})}", passes="vy = +speed, vx = 0 (fails: vy = -0.3)")
        res, ts = await sent(direction="backward", run=True)
        check('move("backward", run=True) runs at RUN_DEFAULT_SPEED (vx = -1.2)',
              bool(ts) and all(abs(m.linear.x + 1.2) < 1e-6 for m in ts), f"sent vx {sorted({round(m.linear.x, 2) for m in ts})}",
              passes="vx = -1.2 (fails: -0.4, the walking default)")
        res, ts = await sent(direction="forward", run=True, speed=0.6)
        check('move("forward", run=True, speed=0.6) still runs: at least 1.0 m/s',
              bool(ts) and min(m.linear.x for m in ts) >= 1.0 - 1e-9, f"sent vx {sorted({round(m.linear.x, 2) for m in ts})}",
              passes="vx = max(0.6, RUN_MIN_SPEED) = 1.0 (fails: 0.6)")
        res, ts = await sent(direction="right", run=True)
        check('move("right", run=True) sidesteps at walking speed (vy = -0.4): sideways never runs',
              bool(ts) and all(abs(m.linear.y + 0.4) < 1e-6 for m in ts), f"result {res}; sent vy {sorted({round(m.linear.y, 2) for m in ts})}",
              passes="vy = -0.4 (fails: -1.2, running sideways)")
        check('... and says so: the reply says "walk"', res.ok and "walk" in res.message.lower(), f"got {res}",
              passes='"Sideways never runs; walking instead. move ... queued" (fails: just "move ... queued")')
        res = await rig.call("move", direction="forward", speed=1.5, duration=10.0)
        check("a move the robot refuses comes back as a failure with the reason",
              res.ok is False and "m" in res.message, f"got {res}",
              passes='Result(False, "that would travel 15.00 m, more than 7.0 m in one move: split it into shorter moves") from walk -> check_move')
    finally:
        rig.close()


async def todo4() -> None:
    print("TODO 4: the arc tool (walk and turn at the same time)")
    rig = Rig()
    try:
        if described(rig, ["arc"], "TODO 4"):
            text = json.dumps({d["name"]: d for d in tools.declarations(rig.ts)}["arc"])
            check(f"arc's words give how far one arc can go (LIMITS.distance_max, {LIMITS.distance_max})",
                  str(LIMITS.distance_max) in text,
                  passes='f"... One arc goes at most {LIMITS.distance_max} m." (fails: "Walk in a curve.")')
            check('... and say that it ends follow mode (says "follow")', "follow" in text.lower(),
                  passes='"Walk forward and turn ..." + ENDS_FOLLOW (fails: leaving ENDS_FOLLOW out)')
        res = await rig.call("arc", direction="left", angle=90, radius=0.5)
        if unwritten(res):
            skip(f"{unwritten(res)} is not written yet (arc needs TODO 2 and TODO 4)")
            return

        def swept(ts):
            return math.degrees(sum(m.angular.z * (b[0] - a[0]) for (a, b), m in zip(zip(ts, ts[1:]), (t[1] for t in ts))))

        twists = rig.robot.twists()
        moving = walk_twists(rig)
        wz = 45.84  # math.degrees(0.4 / 0.5)
        check('arc("left", angle=90, radius=0.5) goes forward at the default 0.4 m/s and turns left',
              bool(moving) and all(abs(m.linear.x - 0.4) < 1e-6 and abs(m.linear.y) < 1e-9 for m in moving)
              and all(m.angular.z > 0 for m in moving),
              f"result {res}; sent vx {sorted({round(m.linear.x, 2) for m in moving})},"
              f" wz {sorted({round(math.degrees(m.angular.z), 1) for m in moving})} deg/s",
              passes="walk(robot, 0.4, 0.0, +45.8, ...) (fails: vx = 0, a turn in place)")
        check(f"... at wz = speed / radius = {wz:.1f} deg/s", bool(moving) and all(abs(math.degrees(m.angular.z) - wz) < 0.5 for m in moving),
              f"sent wz {sorted({round(math.degrees(m.angular.z), 1) for m in moving})} deg/s",
              passes="wz_deg = math.degrees(speed / radius) (fails: speed / radius = 0.8, which is rad/s)")
        check("... for about 90 deg of the circle", 70 <= swept(twists) <= 110, f"turned about {swept(twists):.0f} deg",
              passes="seconds = abs(angle) / abs(wz_deg) = 1.96 s")
        rig.robot.published.clear()
        res = await rig.call("arc", direction="right", angle=90, radius=0.5)
        moving = walk_twists(rig)
        check('arc("right", ...) turns right (wz < 0), still going forward (vx > 0)',
              bool(moving) and all(m.angular.z < 0 and m.linear.x > 0 for m in moving),
              f"result {res}; sent vx {sorted({round(m.linear.x, 2) for m in moving})},"
              f" wz {sorted({round(math.degrees(m.angular.z), 1) for m in moving})}",
              passes="wz_deg = -45.8, vx = +0.4 (fails: vx = -0.4, which backs up)")
        rig.robot.published.clear()
        res = await rig.call("arc", direction="left", angle=90, speed=0.6, radius=1.0)
        moving = walk_twists(rig)
        check('arc("left", speed=0.6, radius=1.0) sends vx = 0.6, wz = +34.4 deg/s',
              bool(moving) and all(abs(m.linear.x - 0.6) < 1e-6 and abs(math.degrees(m.angular.z) - 34.38) < 0.5 for m in moving),
              f"result {res}; sent vx {sorted({round(m.linear.x, 2) for m in moving})},"
              f" wz {sorted({round(math.degrees(m.angular.z), 1) for m in moving})}",
              passes="vx = speed, wz_deg = math.degrees(0.6 / 1.0)")
        rig.robot.published.clear()
        res = await rig.call("arc", direction="left", angle=90, radius=0)
        twists = rig.robot.twists()
        moving = walk_twists(rig)
        check('arc("left", angle=90, radius=0) turns in place, left, about 90 deg (a sharp corner)',
              getattr(res, "ok", False) and bool(moving) and all(abs(m.linear.x) + abs(m.linear.y) < 1e-9 and m.angular.z > 0 for m in moving)
              and 70 <= swept(twists) <= 110, f"got {res}; turned about {swept(twists):.0f} deg",
              passes="walk(robot, 0.0, 0.0, +TURN_SPEED, abs(angle) / TURN_SPEED) (fails: a crash dividing by 0, or a refusal)")
        check('... and the reply says so (says "place")', getattr(res, "ok", False) and "place" in res.message.lower(), f"got {res}",
              passes='"Radius 0: turned in place (a sharp corner). move vx=0 ... queued"')
        rig.robot.published.clear()
        res = await rig.call("arc", direction="left", angle=90, radius=-1)
        check('arc("left", radius=-1) is a failure that says "radius", and sends nothing',
              res.ok is False and "radius" in res.message.lower() and not walk_twists(rig), f"got {res}",
              passes='Result(False, "no arc: radius must be 0 m or more ...")')
        rig.robot.published.clear()
        res = await rig.call("arc", direction="left", angle=90, radius=0.1)
        nums = [float(x) for x in re.findall(r"\d+\.\d+|\d+", res.message)] if not res.ok else []
        check('a too-tight arc (radius 0.1 m at 0.4 m/s) fails, says "radius", and gives the smallest radius (0.2-0.3 m)',
              res.ok is False and "radius" in res.message.lower() and any(0.2 <= n <= 0.3 for n in nums) and not walk_twists(rig),
              f"got {res}",
              passes='"too tight: at 0.4 m/s the radius must be at least 0.25 m" (fails: check_move\'s'
                     ' "too fast: ... |wz|<=114.0": Gemini can\'t tell the radius is the problem)')
    finally:
        rig.close()


# ----------------------------------------------------------------------------- Part 3
async def todo5() -> None:
    print("TODO 5: any motion ends follow mode right away, and Gemini hears about it")
    rig = Rig()
    try:
        res = await rig.ts.set_following(True)
        await rig.settle()
        check("(provided) following starts", rig.ts.following and rig.robot.follower_on, f"got {res}")
        res = await behaviors.stand_up(rig.ts)
        stopped_now = not rig.ts.following and not rig.robot.follower_on
        if not stopped_now and rig.robot.follow_calls == ["on"] and "follow" not in res.message.lower():
            skip("TODO 5 (prepare_to_move) is not written yet")
            return
        check("standing up ends follow mode right away (before the queue reaches it)", stopped_now,
              passes="await robot.stop_following() in prepare_to_move (fails: doing it inside a job)")
        check('the tool result tells Gemini it stopped following (says "follow")', "follow" in res.message.lower(),
              f"result was {res.message!r}",
              passes='"Stopped following the person first. stand up queued; ..." (fails: "stand up queued; ...")')
        await rig.settle()
        await rig.ts.set_following(True)
        await rig.settle()
        res = await behaviors.lie_down(rig.ts)
        check("lying down ends follow mode too", not rig.ts.following and not rig.robot.follower_on)
        check('... and says so (says "follow")', "follow" in res.message.lower(), f"result was {res.message!r}",
              passes='"Stopped following the person first. lie down queued; ..."')
        await rig.settle()
        await rig.ts.set_following(True)
        await rig.settle()
        res = await behaviors.walk(rig.ts, 0.4, 0.0, 0.0, 1.0)
        if unwritten(res):
            print(f"  --   (skipping the walking case: {unwritten(res)} is not written yet)")
        else:
            check("walking ends follow mode", not rig.ts.following and not rig.robot.follower_on)
            check('... and says so (says "follow")', "follow" in res.message.lower(), f"result was {res.message!r}",
                  passes='"Stopped following the person first. move vx=0.4 ... queued; ..."')
        await rig.settle()
        res = await behaviors.stand_up(rig.ts)
        check("with follow mode off, results don't mention following", "follow" not in res.message.lower(),
              f"result was {res.message!r}",
              passes='"stand up queued; ..." (fails: "Stopped following ..." when it wasn\'t following: return "")')
        await rig.settle()
        await rig.ts.set_following(True)
        await rig.settle()
        await rig.ts.mode()
        check("asking what mode it is in keeps following", rig.ts.following and rig.robot.follower_on,
              passes="only motions call prepare_to_move; questions (robot_status) don't")
    finally:
        rig.close()


async def todo6() -> None:
    print("TODO 6: the follow tool")
    rig = Rig()
    try:
        if described(rig, ["follow"], "TODO 6"):
            text = json.dumps({d["name"]: d for d in tools.declarations(rig.ts)}["follow"]).lower()
            check('follow\'s description says what ends follow mode (says "end" or "stop")', "end" in text or "stop" in text,
                  passes='"... Talking keeps following; any motion tool ends it." (fails: "Follow the person.")')
        follow_text = prompt_sections("Follow mode")
        if "TODO" in follow_text or not follow_text:
            skip("TODO 6: prompt.md still says TODO under Follow mode")
        else:
            check('the prompt\'s Follow mode section says what ends following (says "end" or "stop")',
                  "end" in follow_text.lower() or "stop" in follow_text.lower(), follow_text,
                  passes='"- Any motion ends follow mode. Before you call that tool, say so."')
        res = await rig.call("follow", enabled=True)
        if unwritten(res):
            skip(f"{unwritten(res)} is not written yet")
            return
        check("follow(enabled=True) starts following", rig.robot.follower_on, f"got {res}",
              passes="return await robot.set_following(enabled)")
        res = await rig.call("follow", enabled=False)
        check("follow(enabled=False) stops it", not rig.robot.follower_on, f"got {res}",
              passes="return await robot.set_following(enabled) (fails: always passing True)")
    finally:
        rig.close()


# ----------------------------------------------------------------------------- Part 4
async def todo7() -> None:
    print("TODO 7: play_trick (the fake tricks: bow, roll_over, hold_pose)")
    rig = Rig()
    try:
        res = await behaviors.play_trick(rig.ts, "backflip")
        await rig.settle()
        if unwritten(res):
            skip("TODO 7 (play_trick) is not written yet")
            return
        check("an unknown trick fails", not res.ok, f"got {res}", passes="Result(False, ...) when robot.tricks.get(name) is None")
        check("... and the message lists the tricks that exist", "bow" in res.message, f"got {res.message!r}",
              passes='"unknown trick \'backflip\'; available: [\'bow\', \'roll_over\', ...]" (fails: "unknown trick")')

        bow = rig.ts.tricks["bow"]
        start = len(rig.records)
        res = await behaviors.play_trick(rig.ts, "bow")
        check("a trick is accepted right away (queued)", res.ok and "queued" in res.message, f"got {res}",
              passes='return what robot.queue(...) returns: "trick bow queued; ..." (fails: awaiting the job yourself)')
        await rig.settle()
        if any(unwritten(_Msg(message=o)) for o in rig.outcomes(start)):
            skip("TODO 7 (play_policy_trick, the job) is not written yet")
            return
        on, off = rig.robot.activations(bow.controller), rig.robot.deactivations(bow.controller)
        check("the trick's controller took the legs", bool(on), f"switches: {rig.robot.switches}",
              passes="job = lambda: play_policy_trick(robot, trick), queued")
        check("... for the whole trick (trick.total_time)", bool(on and off) and off[-1] - on[-1] >= bow.total_time,
              f"held it {off[-1] - on[-1] if on and off else 0:.1f} s of {bow.total_time:.1f} s",
              passes="the provided play_policy_trick as the job (it waits trick.total_time)")
        check("a trick that ends standing hands the legs back to walking",
              LOCOMOTION_CONTROLLER in rig.robot.active() and bow.controller not in rig.robot.active(),
              f"active: {sorted(rig.robot.active())}", passes="the provided play_policy_trick as the job")
        check('... and the outcome says so (says "locomotion")', any("locomotion" in o for o in rig.outcomes(start)),
              f"log: {rig.outcomes(start)}", passes='"played bow; back in locomotion mode"')

        roll = rig.ts.tricks["roll_over"]
        start = len(rig.records)
        await behaviors.play_trick(rig.ts, "roll_over")
        await rig.settle()
        check("a trick that ends lying down keeps its controller holding the pose",
              roll.controller in rig.robot.active() and LOCOMOTION_CONTROLLER not in rig.robot.active(),
              f"active: {sorted(rig.robot.active())}", passes="the provided play_policy_trick as the job")
        check('... and the outcome says it holds the final pose (says "hold" or "pose")',
              any("hold" in o or "pose" in o for o in rig.outcomes(start)), f"log: {rig.outcomes(start)}",
              passes='"played roll_over; it holds its final pose (lying)"')
        n_on = len(rig.robot.activations(roll.controller))
        await behaviors.play_trick(rig.ts, "roll_over")
        await rig.settle()
        check("playing it again restarts its controller (relax, then switch to it)",
              len(rig.robot.activations(roll.controller)) == n_on + 1 and len(rig.robot.deactivations(roll.controller)) >= 1,
              f"switches: {rig.robot.switches[-3:]}", passes="the provided play_policy_trick as the job (its step 2)")

        hold = rig.ts.tricks["hold_pose"]
        start = len(rig.records)
        res = await behaviors.play_trick(rig.ts, "hold_pose")
        await rig.settle()
        check("an interactive trick: Pupper stays in it", hold.controller in rig.robot.active() and rig.ts.state_now is hold,
              f"active: {sorted(rig.robot.active())}; state_now: {rig.ts.state_now}",
              passes="the provided play_policy_trick as the job")
        out = " ".join(rig.outcomes(start))
        check("... and the outcome lists its actions and its exit clip", "wave_left" in out and "leave" in out,
              f"log: {rig.outcomes(start)}",
              passes='"played hold_pose; it now stays ... Interactive actions: wave_left, wave_right. Any other motion first plays leave."')
        n_switches = len(rig.robot.switches)
        start = len(rig.records)
        await behaviors.play_trick(rig.ts, "hold_pose")
        await rig.settle()
        check("asking for the interactive trick Pupper is already in does nothing, and says so",
              len(rig.robot.switches) == n_switches and any("already" in o for o in rig.outcomes(start)),
              f"log: {rig.outcomes(start)}",
              passes='"already doing hold_pose" (needs then=trick for an interactive trick; fails: then=LOCOMOTION)')

    finally:
        rig.close()

    rig = Rig()
    try:
        await rig.ts.set_following(True)
        await rig.settle()
        res = await behaviors.play_trick(rig.ts, "bow")
        await rig.settle()
        check('a trick ends follow mode (TODO 5\'s rule), and says so (says "follow")',
              not rig.ts.following and "follow" in res.message.lower(), f"got {res}",
              passes='with_note(note, ...): "Stopped following the person first. trick bow queued; ..."')
    finally:
        rig.close()


async def todo8() -> None:
    print("TODO 8: the play_trick tool")
    rig = Rig()
    try:
        if not described(rig, ["play_trick"], "TODO 8"):
            return
        decl = {d["name"]: d for d in tools.declarations(rig.ts)}["play_trick"]
        check("its description lists every trick with its own description (has {tricks})",
              all(t.description in decl["description"] for t in rig.ts.tricks.values()), decl["description"],
              passes='"Play a trick. ..." + ENDS_FOLLOW + "\\nTricks:\\n{tricks}" (fails: listing the tricks by hand)')
        params = decl.get("parameters", {}).get("properties", {})
        enums = [v for p in params.values() for v in p.get("enum", [])]
        check("its choices are the installed tricks", set(rig.ts.tricks) <= set(enums), f"choices: {enums}",
              passes="name=Param(\"string\", ..., enum=TRICK_NAMES) (keep the enum)")
        check('... and it says a trick keeps its final pose (says "pose")',
              "pose" in decl["description"].lower(), decl["description"],
              passes='"A trick keeps its final pose until the next motion or stand_up."')
    finally:
        rig.close()


async def optional_jump() -> None:
    print("Optional: the jump (the fake reference trick 'hop')")
    rig = Rig()
    try:
        res = await behaviors.play_trick(rig.ts, "hop")
        await rig.settle()
        if unwritten(res):
            skip("TODO 7 (play_trick) is not written yet")
            return
        refs = [m for _, topic, m in rig.robot.published if topic.endswith("/reference_trick")]
        check("a reference trick plays through the locomotion policy (robot.play_reference_trick)",
              bool(refs) and LOCOMOTION_CONTROLLER in rig.robot.active(),
              f"got {res}; log: {rig.outcomes()}; active: {sorted(rig.robot.active())}",
              passes='trick.kind == "reference" -> job = lambda: robot.play_reference_trick(trick)')
    finally:
        rig.close()


# ----------------------------------------------------------------------------- Part 5
async def todo9() -> None:
    print("TODO 9: interactive actions (the fake interactive trick 'hold_pose': wave_left, wave_right, leave)")
    rig = Rig()
    try:
        ctrl = rig.ts.tricks["hold_pose"].controller
        res = await behaviors.interactive_action(rig.ts, "wave_left")
        await rig.settle()
        if unwritten(res):
            skip("TODO 9 (interactive_action) is not written yet")
            return
        probe = await behaviors.play_trick(rig.ts, "bow")
        await rig.settle()
        if unwritten(probe) or any(unwritten(_Msg(message=o)) for o in rig.outcomes()):
            skip("these checks need TODO 7 (play_trick) first")
            return

        start = len(rig.records)
        res = await behaviors.interactive_action(rig.ts, "fetch_the_ball")
        await rig.settle()
        fails = ([] if res.ok else [res.message]) + rig.failures(start)
        check("an unknown action fails", bool(fails), f"got {res}",
              passes="Result(False, ...) when robot.action_ids().get(action) is None")
        check("... and the message lists the actions that exist", any("wave_left" in f for f in fails),
              f"messages: {fails}",
              passes='"unknown interactive action fetch_the_ball; available: [\'wave_left\', ...]" (fails: "unknown action")')

        start = len(rig.records)
        rig.robot.published.clear()
        res = await behaviors.interactive_action(rig.ts, "wave_left")  # hold_pose is not running
        await rig.settle()
        check("an action while its trick is not running fails right away", not res.ok, f"got {res}",
              passes="refuse now if robot.state_after_queue is not trick (fails: only checking in the job)")
        check("... the message says which trick has to run first", "hold_pose" in res.message, f"got {res.message!r}",
              passes='"wave_left only works while hold_pose is running; play it first (play_trick hold_pose)"')
        check("... and nothing was sent to the trick", not rig.robot.clips(ctrl), passes="return before robot.queue")

        await behaviors.play_trick(rig.ts, "hold_pose")
        await rig.settle()
        start = len(rig.records)
        res = await behaviors.interactive_action(rig.ts, "wave_left")
        await rig.settle()
        check("an action while the trick holds its pose is accepted", res.ok, f"got {res}",
              passes='"hold_pose wave_left queued; ..."')
        check("... and plays that clip", rig.robot.clips(ctrl)[-1:] == ["wave_left"], f"clips sent: {rig.robot.clips(ctrl)}",
              passes="await robot.play_clip(trick, clip) in the job")
        check("... without a failure in the queue", not rig.failures(start), f"{rig.failures(start)}",
              passes="the job returns play_clip's Result")
        check("... and Pupper stays in the trick afterwards",
              ctrl in rig.robot.active() and LOCOMOTION_CONTROLLER not in rig.robot.active(),
              passes="no switch_to after a clip that isn't the exit clip")

        await behaviors.interactive_action(rig.ts, "leave")
        await rig.settle()
        check("the exit clip plays", rig.robot.clips(ctrl)[-1:] == ["leave"], f"clips sent: {rig.robot.clips(ctrl)}",
              passes="await robot.switch_to(LOCOMOTION_CONTROLLER): it plays the exit clip on the way out")
        check("... then the locomotion policy takes the legs back",
              LOCOMOTION_CONTROLLER in rig.robot.active() and ctrl not in rig.robot.active(),
              f"active: {sorted(rig.robot.active())}",
              passes="for the exit clip: await robot.switch_to(LOCOMOTION_CONTROLLER)")
        check("... and Pupper is in locomotion mode again", rig.ts.state_now == LOCOMOTION,
              f"state_now: {rig.ts.state_now}", passes="switch_to keeps robot.state_now up to date")
        res = await behaviors.interactive_action(rig.ts, "wave_left")
        check("after the exit clip, actions are refused again", not res.ok, f"got {res}",
              passes="then=LOCOMOTION when queuing the exit clip (fails: then=trick)")
    finally:
        rig.close()

    rig = Rig()
    try:
        ctrl = rig.ts.tricks["hold_pose"].controller
        # "Sit and give me your paw", in one breath: two tool calls, back to back.
        await behaviors.play_trick(rig.ts, "hold_pose")
        start = len(rig.records)
        res = await behaviors.interactive_action(rig.ts, "wave_right")
        await rig.settle()
        check("an action queued right behind its trick is accepted and plays",
              res.ok and "wave_right" in rig.robot.clips(ctrl),
              f"result {res}; queue: {rig.failures(start)}; clips sent: {rig.robot.clips(ctrl)}",
              passes="the now-check asks robot.state_after_queue (fails: asking robot.state_now or is_active now)")

        # Queued behind the trick, but the trick was relaxed before the queue got there.
        await behaviors.play_trick(rig.ts, "hold_pose")
        await rig.settle()
        await behaviors.lie_down(rig.ts)  # queued: relaxes the motors
        rig.ts.state_after_queue = rig.ts.tricks["hold_pose"]  # as if it were still coming
        rig.robot.published.clear()
        start = len(rig.records)
        res = await behaviors.interactive_action(rig.ts, "wave_left")
        await rig.settle()
        check("when the queue gets to an action, it checks again that its trick is running",
              not rig.robot.clips(ctrl) and bool(rig.failures(start)),
              f"clips sent: {rig.robot.clips(ctrl)}; log: {rig.outcomes(start)}",
              passes="in the job, first: if robot.state_now is not trick: return Result(False, ...)")
    finally:
        rig.close()


async def todo10() -> None:
    print("TODO 10: the interactive_action tool")
    rig = Rig()
    try:
        if described(rig, ["interactive_action"], "TODO 10"):
            decl = {d["name"]: d for d in tools.declarations(rig.ts)}["interactive_action"]
            check("its description lists the actions (has {actions})", "Waves the left paw." in decl["description"],
                  decl["description"], passes='"Do something inside a trick ... Actions:\\n{actions}"')
            params = decl.get("parameters", {}).get("properties", {})
            enums = [v for p in params.values() for v in p.get("enum", [])]
            check("its choices are the installed actions", {"wave_left", "wave_right", "leave"} <= set(enums),
                  f"choices: {enums}", passes="action=Param(\"string\", ..., enum=ACTION_IDS) (keep the enum)")
        ctrl = rig.ts.tricks["hold_pose"].controller
        await behaviors.play_trick(rig.ts, "hold_pose")
        await rig.settle()
        res = await rig.call("interactive_action", action="wave_left")
        if unwritten(res):
            skip(f"{unwritten(res)} is not written yet")
            return
        check("calling the interactive_action tool plays the clip", res.ok and rig.robot.clips(ctrl)[-1:] == ["wave_left"],
              f"result {res}; clips sent: {rig.robot.clips(ctrl)}", passes="return await interactive_action(robot, action)")
    finally:
        rig.close()
    shake = HERE.parent / "tricks" / "sit_stay_shake" / "trick.yaml"
    if shake.exists():
        import yaml

        spec = yaml.safe_load(shake.read_text()) or {}
        text = json.dumps(spec)
        if "TODO" in text:
            skip("TODO 10: tricks/sit_stay_shake/trick.yaml still says TODO")
        else:
            check("sit_stay_shake/trick.yaml describes the trick (at least 20 characters)",
                  len(str(spec.get("description", ""))) >= 20, f"description: {spec.get('description')!r}",
                  passes='description: "Sit and stay sitting. Use it when someone says sit."')
            check("... and each of its actions (at least 10 characters each)",
                  all(len(str(v)) >= 10 for v in (spec.get("actions") or {}).values())
                  and bool(spec.get("actions")), f"actions: {spec.get('actions')}",
                  passes='shake_left: "Shakes the left paw, then keeps sitting." (fails: shake_left: "left")')
    else:
        print("  --   (no tricks/sit_stay_shake yet: fetch your policy with get_trick.py, Part 5)")


# ----------------------------------------------------------------------------- Part 7
async def todo11() -> None:
    print("TODO 11: prompt.md, second pass")
    text = prompt_sections("Tricks", "Safety").lower()
    if "todo" in text or not text:
        skip("TODO 11: prompt.md still says TODO under Tricks or Safety")
        return
    check('the prompt says what to do when someone says stop (says "stop")', "stop" in text,
          passes='"- If someone says stop, call stop right away."')
    check('... and when Pupper falls over (says "fall" or "fell")', "fall" in text or "fell" in text,
          passes='"- If you fall over, ask a person to set you upright, then call stand_up."')
    check('... and how to handle several requests in a row (says "order", "row" or "sequence")',
          "order" in text or "row" in text or "sequence" in text,
          passes='"- If the person asks for several things in a row, call the tools in that order."')


# ----------------------------------------------------------------------------- provided
async def provided() -> None:
    print("The provided robot (these should pass before you change anything)")
    rig = Rig()
    try:
        bad = await tools.call(rig.ts, "stand_up", {"speed": 3})
        check("a tool call with a parameter the tool doesn't have is refused", not bad.ok, f"got {bad}")
        bad = await tools.call(rig.ts, "play_trick", {"name": "backflip"})
        check("a value outside a tool's choices is refused, listing the choices",
              not bad.ok and "bow" in bad.message, f"got {bad}")

        rig.robot.controllers[LOCOMOTION_CONTROLLER] = "inactive"
        rig.ts.walk_since = None
        await behaviors.stand_up(rig.ts)
        on_at = rig.clock.now
        await rig.settle()
        check("stand_up hands the legs to the locomotion policy", LOCOMOTION_CONTROLLER in rig.robot.active())
        check("... and waits until the policy is ready before the next job starts",
              rig.clock.now - on_at >= WALK_READY_S - 0.01, f"ready after {rig.clock.now - on_at:.2f} s")
        await behaviors.lie_down(rig.ts)
        await rig.settle()
        check("lie_down relaxes every leg controller", not (rig.robot.active() & rig.ts.leg_controllers),
              f"active: {sorted(rig.robot.active())}")
        await behaviors.stand_up(rig.ts)
        await rig.settle()

        # A tilt e-stop: the controller says so on /rosout.
        rig.robot.send("/rosout", _Msg(name="gemini_walk", msg=robot_api.ESTOP_LOG))
        await rig.settle()
        res = await tools.call(rig.ts, "robot_status", {})
        check("after a fall, robot_status says it fell over", "fall" in res.message.lower() or "fell" in res.message.lower(),
              f"got {res.message!r}")
        rig.imu(tilt_deg=70)
        res = await rig.ts.switch_to(LOCOMOTION_CONTROLLER)
        check("standing up while still tipped over is refused, asking for help",
              not res.ok and "upright" in res.message, f"got {res}")
        rig.imu(tilt_deg=3)
        res = await rig.ts.switch_to(LOCOMOTION_CONTROLLER)
        check("once upright and still, standing up works again", res.ok and rig.ts.fallen is None, f"got {res}")

        await rig.ts.wait(5.0)
        res = await tools.call(rig.ts, "stop", {})
        check("stop clears the queue", rig.ts.commands.empty() and "stopped" in res.message, f"got {res}")
    finally:
        rig.close()

    # The server restarted while the robot sat in an interactive trick.
    rig = Rig()
    try:
        hold = fixture_tricks()["hold_pose"]
        rig.robot.controllers[LOCOMOTION_CONTROLLER] = "inactive"
        rig.robot.controllers[hold.controller] = "active"
        rig.ts.worker.cancel()
        rig.ts = Robot()  # a fresh server, the robot still sitting
        await rig.settle()
        check("a restarted server notices the robot is still in an interactive trick",
              getattr(rig.ts.state_now, "name", None) == "hold_pose", f"state_now: {rig.ts.state_now}")
        await behaviors.stand_up(rig.ts)
        await rig.settle()
        check("... so standing up plays the trick's exit clip first",
              rig.robot.clips(hold.controller)[-1:] == ["leave"] and LOCOMOTION_CONTROLLER in rig.robot.active(),
              f"clips sent: {rig.robot.clips(hold.controller)}; active: {sorted(rig.robot.active())}")
    finally:
        rig.close()

    # lab/chat.py's fake-robot mode tells Gemini exactly what the web app would.
    import gemini_chat
    desk = gemini_chat.DeskBackend()
    prompt, decls = desk.setup()
    try:
        # The robot server appends what Pupper remembers (hackathon_memory.memory_prompt) to prompt.md.
        check("chat.py gives Gemini the same prompt as the robot server",
              prompt == system_prompt() + "\n\n" + behaviors.memory_prompt())
        check("... and the same tools", decls == tools.declarations(desk.rig.ts),
              f"chat.py: {[d['name'] for d in decls]}")
    finally:
        desk.rig.close()

    # The stop tool, in the middle of things.
    rig = Rig()
    try:
        bow, hold = rig.ts.tricks["bow"], rig.ts.tricks["hold_pose"]
        await rig.ts.queue("trick bow", lambda: behaviors.play_policy_trick(rig.ts, bow), bow.total_time, then=LOCOMOTION)
        t0 = rig.clock.now
        while rig.clock.now < t0 + 2.0:  # two seconds into the bow
            await _real_sleep(0)
        check("(setup) the bow is playing", bow.controller in rig.robot.active())
        res = await tools.call(rig.ts, "stop", {})
        await rig.settle()
        check("stop in the middle of a trick relaxes the motors (a trick policy plays on by itself)",
              not (rig.robot.active() & rig.ts.leg_controllers), f"active: {sorted(rig.robot.active())}")
        check("... and the reply says so", res.ok and "relaxed" in res.message, f"got {res}")

        async def walking_job():
            await rig.ts.switch_to(LOCOMOTION_CONTROLLER)
            await rig.ts.sleep(5.0)
            return robot_api.Result(True, "walked")

        await rig.ts.queue("walk", walking_job, 5.0, then=LOCOMOTION)
        t0 = rig.clock.now
        while rig.clock.now < t0 + 6.0:  # standing up (4.2 s), then walking
            await _real_sleep(0)
        res = await tools.call(rig.ts, "stop", {})
        await rig.settle()
        check("stop during a walk just stands still: the locomotion policy keeps the legs",
              LOCOMOTION_CONTROLLER in rig.robot.active() and "relaxed" not in res.message, f"got {res}")

        await rig.ts.queue("trick hold_pose", lambda: behaviors.play_policy_trick(rig.ts, hold), hold.total_time,
                           then=hold)
        await rig.settle()
        res = await tools.call(rig.ts, "stop", {})
        await rig.settle()
        check("stop while a trick only holds its pose leaves it holding (nothing is moving)",
              hold.controller in rig.robot.active() and "relaxed" not in res.message, f"got {res}")
    finally:
        rig.close()

    # stop is immediate and clears everything, even requests still being handled.
    rig = Rig()
    try:
        bow = rig.ts.tricks["bow"]

        async def long_walk():
            await rig.ts.switch_to(LOCOMOTION_CONTROLLER)
            while True:
                rig.ts.send_speed(0.4, 0, 0)
                await rig.ts.sleep(0.05)

        await rig.ts.queue("walk", long_walk, 10.0, then=LOCOMOTION)
        await rig.ts.queue("trick bow", lambda: behaviors.play_policy_trick(rig.ts, bow), bow.total_time, then=LOCOMOTION)
        t0 = rig.clock.now
        for _ in range(100000):  # walking for a second
            if rig.clock.now >= t0 + 1.0:
                break
            await _real_sleep(0)
        rig.robot.published.clear()
        n_switches, t_stop = len(rig.robot.switches), rig.clock.now
        res = await tools.call(rig.ts, "stop", {})
        sent = rig.robot.twists()
        check("stop sends a zero speed at once, before anything else",
              bool(sent) and sent[0][0] == t_stop and sent[0][1].linear.x == 0.0,
              f"first speeds sent: {[(round(t - t_stop, 2), m.linear.x) for t, m in sent[:3]]}")
        check("... and empties the queue right away (the running job is cancelled)",
              rig.ts.commands.empty() and (rig.ts.current is None or rig.ts.current.done()),
              f"queued: {rig.ts.commands.qsize()}; running: {rig.ts.current}")
        await rig.settle()
        later = [sw for sw in rig.robot.switches[n_switches:] if bow.controller in str(sw)]
        check("... so nothing that was queued behind runs (the bow never plays)", not later, f"switches: {later}")
        check("... and the walk's speeds stop", all(m.linear.x == 0.0 for _, m in rig.robot.twists()),
              f"speeds after the stop: {sorted({m.linear.x for _, m in rig.robot.twists()})}")

        # follow(true) waits while the legs switch; a stop in that time wins.
        await behaviors.lie_down(rig.ts)
        await rig.settle()
        follow = asyncio.ensure_future(tools.call(rig.ts, "follow", {"enabled": True}))
        t0 = rig.clock.now
        while rig.clock.now < t0 + 1.0 and not follow.done():  # still standing up (4.2 s)
            await _real_sleep(0)
        if follow.done() and unwritten(follow.result()):
            skip(f"the follow-and-stop check needs {unwritten(follow.result())} first")
        else:
            await tools.call(rig.ts, "stop", {})
            await rig.settle()
            res = await follow
            check("a stop while follow mode is starting: it doesn't start", not rig.ts.following and not res.ok,
                  f"following: {rig.ts.following}; follow said: {res}")

        # A request still in its *now* part when the stop came in doesn't queue its job.
        token = robot_api.CALL_STOPS.set(rig.ts.stops)
        try:
            await rig.ts.immediate_stop()
            res = await behaviors.stand_up(rig.ts)
        finally:
            robot_api.CALL_STOPS.reset(token)
        check("a request that began before a stop doesn't queue anything after it",
              not res.ok and rig.ts.commands.empty(), f"got {res}; queued: {rig.ts.commands.qsize()}")
    finally:
        rig.close()

    # A trick that walks in its own stance (the spider): its clips, then its walking policy.
    def creep_tricks() -> dict[str, Trick]:
        tricks = fixture_tricks()
        tricks["creep"] = Trick(
            name="creep", description="Crouches low and creeps.", kind="policy", path=Path("creep.json"),
            duration=8.0, end_pose="crouched",
            clips=[_clip("down", 0, 200, next="idle"), _clip("idle", 199, 250, loop=True), _clip("up", 249, 450)],
            actions={"up": "Stands back up."}, exit_clip="up",
            walk_path=Path("creep_walk.json"),
            walk=WalkConfig(speed=0.1, max_forward=0.15, max_sideways=0.1, turn=30.0, note="It creeps."),
        )
        return tricks

    rig = Rig(tricks=creep_tricks)
    try:
        creep = rig.ts.tricks["creep"]
        res = await behaviors.play_policy_trick(rig.ts, creep)
        check("a walking trick: its clips play, then its walking policy takes the legs",
              creep.walk_controller in rig.robot.active() and creep.controller not in rig.robot.active()
              and rig.robot.start_clips.get(creep.controller) == [""] and rig.ts.state_now is creep,
              f"active: {sorted(rig.robot.active())}; start_clips: {rig.robot.start_clips}; got {res}")
        check("... and the outcome says it walks in its stance", res.ok and "walk" in res.message, f"got {res}")

        rig.ts.state_after_queue = creep  # as play_trick's then= leaves it
        check("robot.speeds() in a walking trick: its own defaults and limits",
              rig.ts.speeds().walk == 0.1 and rig.ts.speeds().turn == 30.0 and rig.ts.speeds().limits.vx_max == 0.15,
              f"got {rig.ts.speeds()}")
        rig.robot.published.clear()
        res = await behaviors.walk(rig.ts, 0.3, 0.0, 0.0, 2.0)
        await rig.settle()
        if unwritten(res):
            skip(f"walking in the stance needs {unwritten(res)} first")
        else:
            check("a move too fast for the trick is refused, with its limit, and nothing is sent",
                  not res.ok and "0.15" in res.message and "creep" in res.message and not rig.robot.twists(),
                  f"got {res}", passes='"too fast: |vx|<=0.15 m/s (in creep)"')
            res = await behaviors.walk(rig.ts, 0.1, 0.0, 0.0, 2.0)
            await rig.settle()
            sent = [m.linear.x for _, m in rig.robot.twists() if m.linear.x]
            check("a move within its limits drives the trick's walking policy, and Pupper stays in the trick",
                  res.ok and creep.walk_controller in rig.robot.active() and sent and abs(max(sent) - 0.1) < 1e-6
                  and abs(len(sent) / LLM_CMD_RATE - 2.0) < 0.1 and rig.ts.state_now is creep,
                  f"got {res}; active: {sorted(rig.robot.active())}; sent {len(sent) / LLM_CMD_RATE:.1f} s")
            rig.robot.published.clear()
            res = await rig.call("turn", direction="left", angle=90)
            await rig.settle()
            turned = [math.degrees(m.angular.z) for _, m in rig.robot.twists() if m.angular.z]
            if unwritten(res):
                skip(f"turning in the stance needs {unwritten(res)} first")
            else:
                check("turn in the trick turns at its speed (robot.speeds().turn), as far as asked",
                      res.ok and turned and abs(max(turned) - 30.0) < 0.1
                      and abs(sum(turned) / LLM_CMD_RATE - 90.0) < 3.0,
                      f"got {res}; top {max(turned or [0]):.1f} deg/s, {sum(turned) / LLM_CMD_RATE:.0f} deg",
                      passes="turn_speed = robot.speeds().turn")

        await behaviors.stand_up(rig.ts)
        await rig.settle()
        check("standing up leaves through the exit clip: the trick's policy, started at it",
              rig.robot.start_clips.get(creep.controller) == ["", "up"]
              and rig.robot.activations(creep.controller)[-1] > rig.robot.deactivations(creep.walk_controller)[-1] - 1e-6
              and LOCOMOTION_CONTROLLER in rig.robot.active() and rig.ts.state_now == LOCOMOTION,
              f"start_clips: {rig.robot.start_clips}; active: {sorted(rig.robot.active())}; state {rig.ts.state_now}")

        await behaviors.play_policy_trick(rig.ts, creep)
        await rig.ts.emergency_stop()
        check("the e-stop relaxes a trick's walking policy too", not (rig.robot.active() & rig.ts.leg_controllers),
              f"active: {sorted(rig.robot.active())}")
    finally:
        rig.close()

    # replaced_by: a trick steps aside while its replacement is installed.
    import tricks as tricks_mod
    with tempfile.TemporaryDirectory() as tmp:
        def add(name, extra=""):
            (Path(tmp) / name).mkdir()
            (Path(tmp) / name / "policy.json").write_text(json.dumps({"trick": {"duration_s": 1.0}}))
            (Path(tmp) / name / "trick.yaml").write_text(f"description: {name}\ntype: policy\npolicy: policy.json\n{extra}")
        add("old_sit", "replaced_by: new_sit\n")
        check("a trick with replaced_by is there while its replacement isn't",
              "old_sit" in tricks_mod.load_tricks(Path(tmp)))
        add("new_sit")
        got = tricks_mod.load_tricks(Path(tmp))
        check("... and steps aside once the replacement is installed", "old_sit" not in got and "new_sit" in got,
              f"tricks: {sorted(got)}")

    def no_clip_tricks():
        return {k: v for k, v in fixture_tricks().items() if not v.clips}

    rig = Rig(no_clip_tricks)
    try:
        decls = {d["name"]: d for d in tools.declarations(rig.ts)}
        check("interactive_action is offered to Gemini even before an interactive trick is installed", "interactive_action" in decls,
              f"declared: {sorted(decls)}")
        res = await tools.call(rig.ts, "interactive_action", {"action": "shake_left"})
        check("... and says nothing is installed yet", not res.ok and "installed" in res.message, f"got {res}")
    finally:
        rig.close()


# ----------------------------------------------------------------------------- main
async def reminders() -> None:
    """Reminders (added after the lab): robot/test_reminders.py."""
    import test_reminders
    await test_reminders.run(check, Rig, tools)


GROUPS = {
    "todo1": todo1, "todo2": todo2, "todo3": todo3, "todo4": todo4, "todo5": todo5, "todo6": todo6,
    "todo7": todo7, "todo8": todo8, "todo9": todo9, "todo10": todo10, "todo11": todo11, "robot": provided,
    "optional_jump": optional_jump, "reminders": reminders,
}
OPT_IN = {"optional_jump"}  # run only when asked for by name
PARTS = {
    "part1": ["todo1"],
    "part2": ["todo2", "todo3", "todo4"],
    "part3": ["todo5", "todo6"],
    "part4": ["todo7", "todo8"],
    "part5": ["todo9", "todo10"],
    "part7": ["todo11"],
}


def main() -> None:
    chosen: list[str] = []
    for arg in sys.argv[1:] or ["robot", *(g for g in GROUPS if g not in OPT_IN)]:
        for name in PARTS.get(arg, [arg]):
            if name not in GROUPS:
                sys.exit(f"unknown test group {arg!r}; choose from {list(GROUPS)} or {list(PARTS)}")
            if name not in chosen:
                chosen.append(name)
    for name in chosen:
        asyncio.run(GROUPS[name]())
        print()
    if FAIL:
        print(f"{FAIL} check(s) failed, {PASS} passed.")
        raise SystemExit(1)
    if TODO:
        print(f"All {PASS} checks that ran passed; {TODO} TODO(s) not written yet. Keep going!")
        raise SystemExit(1)
    print(f"All {PASS} checks passed. Now try it on the robot (with a TA's OK for new policies).")


if __name__ == "__main__":
    main()
