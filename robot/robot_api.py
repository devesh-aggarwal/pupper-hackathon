"""The robot: a command queue, the leg controllers, falls and the e-stop (provided).

You don't change this file. Your code in behaviors.py uses the Robot API below;
the table in the lab handout lists it too.

  await robot.queue(name, job, seconds, then=state)
                                  put a job in line; returns what Gemini hears now
  await robot.switch_to(controller or None)
                                  give the legs to one controller, or relax them
  robot.state_now                 the state Pupper is in right now
  robot.state_after_queue         ... once everything already queued has run
  robot.send_speed(vx, vy, wz_deg)  one speed command to the policy that walks now
  robot.walking_controller(state)  the controller that walks in a state (locomotion,
                                  or a walking trick's own walking policy)
  robot.speeds()                  default speeds and limits where Pupper will be
                                  once the queue has run (move, turn, arc use them)
  await robot.sleep(seconds), robot.now()
                                  wait without freezing the server; the time now
  robot.fallen                    which controller fell over (None if upright)
  robot.following, await robot.stop_following()
  await robot.is_active(controller)
  await robot.play_clip(trick, clip)
  await robot.play_reference_trick(trick)
  robot.tricks, robot.action_ids()

The robot is always in exactly one state, and each state is whatever holds the
legs: LOCOMOTION (the locomotion policy, the gemini_walk controller; follow mode
is on or off inside it), one trick (its trick_<name> controller: trick mode),
MOTORS_IDLE (nothing), or FALLEN. Changing state is one controller switch,
switch_to, and only switch_to (and a fall or the e-stop) changes robot.state_now.

Adapted from the Pupper monorepo's ros_tool_server.py, without the LiveKit voice
agent it was written for. Topics and controllers used (all started by
launch/gemini_pupper.launch.py):
  /llm_cmd_vel                      speed commands; cmd_vel_mux forwards them to
                                    /cmd_vel, and a joystick always overrides them
  trick_<name> controllers          trick policies (loaded by the launch file)
  /trick_<name>/clip, /clip_status  interactive tricks: request a clip, follow playback
  /camera/image_raw/compressed      latest camera frame
  activate/deactivate_person_following        person_follower services
  /imu_sensor_broadcaster/imu       body tilt and rotation, to check it is safe to stand up
  /rosout                           the controllers' "Emergency stop triggered" (tilt e-stop)

Falls. A leg controller that tilts past its max_body_angle latches an e-stop: it
stays "active" but only damps the joints until it is activated again. The robot
notices from /rosout, cancels the queue, and refuses motion until standing up
again, which first checks that the robot is upright and still, then restarts
the controller (its on_activate clears the latch and eases the legs to standing).
"""

from __future__ import annotations

import asyncio
import collections
import contextvars
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import Log, Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import CompressedImage, Imu
from std_msgs.msg import Empty, Float32MultiArray, String
from std_srvs.srv import Trigger

from spawn_controllers import LOADED, spawn
from tricks import TRICK_FADE_S, TRICK_INIT_S, WALK_CONTROLLER, WALK_READY_S, Trick, load_tricks, reference_message_data

log = logging.getLogger("robot_api")

# The locomotion controller Gemini drives: gemini_walk, the course default policy
# (the joystick's X button) loaded by our launch file with running allowed and
# reference tricks enabled. See tricks.py.
LOCOMOTION_CONTROLLER = WALK_CONTROLLER

# The states that aren't a trick (a trick's state is its Trick).
LOCOMOTION = "locomotion"  # the locomotion policy holds the legs (following or not)
MOTORS_IDLE = "motors idle"  # no controller holds the legs: Pupper lies down
FALLEN = "fallen"  # a tilt e-stop latched: the motors only damp until a safe stand-up
# gemini_walk holds the default pose for init_duration, then fades the policy in
# over fade_in_duration (the course default controller's 2 s + 2 s); reference
# tricks are refused until then. WALK_READY_S (tricks.py) is that wait.
# Everything that may hold the leg joints; switching modes deactivates the others.
LEG_CONTROLLERS = {
    WALK_CONTROLLER,
    "neural_controller",
    "neural_controller_default",
    "neural_controller_default_alt",
    "neural_controller_three_legged",
    "neural_controller_student",
}

LLM_CMD_RATE = 20.0  # Hz; cmd_vel_mux drops a source that is quiet for 500 ms

# Standing up again after a tilt e-stop: the body must be this level and this still
# (all IMU samples over SAFE_STILL_S) before a controller is restarted.
SAFE_TILT_DEG = 20.0
SAFE_GYRO_RAD_S = 0.5
SAFE_STILL_S = 1.0
SAFE_WAIT_S = 4.0  # how long standing up waits for the robot to settle
ESTOP_LOG = "Emergency stop triggered"
FELL_MESSAGE = (
    "the robot fell over: its tilt e-stop tripped and the motors only damp now. "
    "A person should set it upright on its feet; then call stand_up"
)


@dataclass
class MoveLimits:
    """Speeds below the minimums barely move the robot; above the maximums are unsafe."""

    vx_min: float = 0.2  # m/s
    vy_min: float = 0.2  # m/s
    wz_min: float = 20.0  # deg/s
    vx_max: float = 1.5  # m/s; above about 0.5 the default policy runs
    vy_max: float = 0.5  # m/s, the policy's trained sideways range
    wz_max: float = 114.0  # deg/s, the policy's trained 2 rad/s
    duration_max: float = 20.0  # s: long enough for a 1 m circle at a walk
    distance_max: float = 7.0  # m in one move (a 1 m circle is 6.3 m), so a run can't cross the room


LIMITS = MoveLimits()


@dataclass
class Speeds:
    """How Pupper walks in a state: default speeds, and the limits check_move enforces."""

    walk: float = 0.4  # m/s: move's and arc's speed when Gemini gives none
    turn: float = 60.0  # deg/s: turn's speed (and arc's, for a sharp corner)
    limits: MoveLimits = field(default_factory=lambda: LIMITS)
    where: str = ""  # "" in locomotion; " (in <trick>)" in a walking trick, for refusals


SPEEDS = Speeds()  # locomotion mode


@dataclass
class Result:
    """What a tool tells Gemini: did it work, and what happened (or why not)."""

    ok: bool
    message: str

    def as_dict(self) -> dict:
        return {"success": self.ok, "message": self.message}


def with_note(note: str, res: Result) -> Result:
    """``res`` with ``note`` (e.g. "Stopped following the person first.") in front."""
    return Result(res.ok, f"{note.strip()} {res.message}") if note.strip() else res


def not_written(todo: str, where: str) -> Result:
    """What an unfinished TODO tells Gemini, so it says so instead of pretending."""
    return Result(False, f"{todo} is not written yet ({where})")


# then= for a queued job that doesn't move the legs (robot.wait): see Robot.queue.
UNCHANGED = object()

# How many stops had happened when the tool call being handled began (set by
# tool_registry.call). A request that was still in its *now* part when a stop
# came in must not queue its job afterwards: stop clears everything.
CALL_STOPS: contextvars.ContextVar[int | None] = contextvars.ContextVar("call_stops", default=None)


def state_name(state) -> str:
    """A state in words: "locomotion", "motors idle", "fallen", or the trick's name."""
    return state.name if isinstance(state, Trick) else str(state)

Job = Callable[[], Awaitable[Result]]


class Robot:
    """Owns a ROS node and runs queued jobs one at a time."""

    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("gemini_pupper_tools")
        self.switch_client = self.node.create_client(SwitchController, "/controller_manager/switch_controller")
        self.list_client = self.node.create_client(ListControllers, "/controller_manager/list_controllers")
        self.follow_on = self.node.create_client(Trigger, "activate_person_following")
        self.follow_off = self.node.create_client(Trigger, "deactivate_person_following")
        self.cmd_pub = self.node.create_publisher(Twist, "/llm_cmd_vel", 10)
        self.ref_pub = self.node.create_publisher(Float32MultiArray, f"/{WALK_CONTROLLER}/reference_trick", 10)
        self.walk_since = None  # monotonic time gemini_walk was last activated

        self.latest_image: queue.Queue[CompressedImage] = queue.Queue(maxsize=1)
        self.node.create_subscription(CompressedImage, "/camera/image_raw/compressed", self._on_image, 1)

        # The state Pupper is in right now (set by switch_to, a fall, the e-stop),
        # and the one it will be in once everything queued so far has run (set by
        # queue's then=): what a request made *now* can rely on.
        self.state_now: str | Trick = MOTORS_IDLE
        self.state_after_queue: str | Trick = MOTORS_IDLE
        self._running_job_then: str | Trick = MOTORS_IDLE  # ... once the running job ends
        # Interactive tricks: request clips and follow what the controller is playing.
        self.clip_status: dict[str, tuple[str, bool]] = {}  # controller -> (clip, holding)
        self.clip_pubs: dict[str, object] = {}
        for t in load_tricks().values():
            if t.interactive:
                self.clip_pubs[t.controller] = self.node.create_publisher(String, f"/{t.controller}/clip", 10)
                self.node.create_subscription(
                    String, f"/{t.controller}/clip_status", lambda m, c=t.controller: self._on_clip_status(c, m), 10
                )

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()

        self.tricks: dict[str, Trick] = load_tricks()
        self.leg_controllers = (LEG_CONTROLLERS | {t.controller for t in self.tricks.values() if t.kind == "policy"}
                                | {t.walk_controller for t in self.tricks.values() if t.walks})
        self.param_clients: dict[str, object] = {}  # controller -> its set_parameters client
        self.following = False
        self.commands: asyncio.Queue = asyncio.Queue()
        self.current: asyncio.Task | None = None
        self.busy_until = 0.0  # monotonic time when the queued work should finish
        self.outcomes: collections.deque[dict] = collections.deque(maxlen=8)
        self.outcome_id = 0
        self.stops = 0  # how many times stop (or the e-stop) has run
        self.worker = asyncio.get_event_loop().create_task(self._run_queue())
        # The gamepad e-stop (right-stick press) relaxes the course's joystick
        # controllers and publishes /emergency_stop. The controllers used here
        # (walking, tricks) are not on its list, so relax them too.
        self.loop = asyncio.get_event_loop()
        self.node.create_subscription(Empty, "/emergency_stop", self._on_estop, 10)
        self.gamepad_estop_at = 0.0  # monotonic time of the last gamepad e-stop
        # Tilt e-stops: the controller that latched one, until a safe stand-up clears it.
        self.fallen: str | None = None
        self.imu: collections.deque = collections.deque()  # (monotonic t, tilt deg, gyro rad/s)
        self.node.create_subscription(Imu, "/imu_sensor_broadcaster/imu", self._on_imu, 10)
        self.node.create_subscription(Log, "/rosout", self._on_rosout, 100)
        log.info("tool server ready: %d tricks, locomotion controller %s", len(self.tricks), LOCOMOTION_CONTROLLER)

    # ---------------------------------------------------------------- time
    def now(self) -> float:
        """Seconds on a clock that only goes forward (time.monotonic)."""
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        """Wait without freezing the server. Never use time.sleep in here: it would
        stop everything else, the stop tool included."""
        await asyncio.sleep(seconds)

    # ---------------------------------------------------------------- ROS helpers
    def _on_clip_status(self, controller: str, msg: String) -> None:
        # "<clip> playing|holding frame N queued K"
        parts = msg.data.split()
        if len(parts) >= 2:
            self.clip_status[controller] = (parts[0], parts[1] == "holding")

    def _on_image(self, msg: CompressedImage) -> None:
        try:
            self.latest_image.get_nowait()
        except queue.Empty:
            pass
        self.latest_image.put_nowait(msg)

    def _on_imu(self, msg: Imu) -> None:
        q, w = msg.orientation, msg.angular_velocity
        # Angle between the body's up axis and the world's: the controllers' tilt check.
        up = max(-1.0, min(1.0, 1.0 - 2.0 * (q.x * q.x + q.y * q.y)))
        now = time.monotonic()
        self.imu.append((now, math.degrees(math.acos(up)), math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z)))
        while self.imu and self.imu[0][0] < now - SAFE_STILL_S - 0.5:
            self.imu.popleft()

    def _on_rosout(self, msg: Log) -> None:
        name = msg.name.rsplit(".", 1)[-1]
        if msg.msg != ESTOP_LOG or name not in self.leg_controllers:
            return
        if time.monotonic() - self.gamepad_estop_at < 2.0:
            return  # the gamepad e-stop, which relaxes everything itself
        asyncio.run_coroutine_threadsafe(self._on_fall(name), self.loop)

    async def _on_fall(self, controller: str) -> None:
        if self.fallen:
            return
        self.fallen = controller
        await self.immediate_stop()
        self.state_now = self.state_after_queue = FALLEN
        log.warning("tilt e-stop in %s: the robot fell over; waiting for a safe stand-up", controller)

    def upright(self) -> tuple[bool, str]:
        """Whether the robot is level and still enough to restart a controller."""
        now = time.monotonic()
        samples = list(self.imu)  # the ROS thread keeps appending
        recent = [s for s in samples if s[0] >= now - SAFE_STILL_S]
        if not recent or now - recent[-1][0] > 0.2:
            return False, "no IMU data, so it can't check that the robot is upright"
        if now - samples[0][0] < SAFE_STILL_S:
            return False, "not enough IMU data yet"
        tilt = max(s[1] for s in recent)
        gyro = max(s[2] for s in recent)
        if tilt > SAFE_TILT_DEG:
            return False, f"the robot is tilted {tilt:.0f} degrees (needs under {SAFE_TILT_DEG:.0f}); set it upright on its feet"
        if gyro > SAFE_GYRO_RAD_S:
            return False, "the robot is still moving or being handled; let it rest on the ground"
        return True, f"upright ({tilt:.0f} degrees) and still"

    async def _wait_upright(self) -> tuple[bool, str]:
        deadline = time.monotonic() + SAFE_WAIT_S
        while True:
            ok, why = self.upright()
            if ok or time.monotonic() > deadline:
                return ok, why
            await asyncio.sleep(0.1)

    async def _call(self, client, request, timeout: float = 3.0):
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        return future.result() if future.done() else None

    async def controller_states(self) -> dict[str, str]:
        res = await self._call(self.list_client, ListControllers.Request())
        return {c.name: c.state for c in res.controller} if res else {}

    async def is_active(self, controller: str) -> bool:
        """Is this controller driving the legs right now?"""
        return (await self.controller_states()).get(controller) == "active"

    # ---------------------------------------------------------------- the legs
    async def switch_to(self, controller: str | None, start_clip: str = "") -> Result:
        """Hand the legs to one controller, or to none (relax the motors). Every other
        leg controller is deactivated in the same switch, so exactly one policy (or
        nothing) drives the legs.

        It also takes care of: standing up safely after a fall (only once the robot
        is upright and still), loading a controller the launch file missed,
        leaving an interactive trick through its exit clip before another motion
        takes the legs, waiting until the locomotion policy is ready to walk, and
        keeping robot.state_now up to date.
        """
        activate = [controller] if controller else []
        deactivate = sorted(self.leg_controllers - set(activate))
        states = await self.controller_states()
        if not states:
            return Result(False, "controller_manager is not running; is the robot stack launched?")
        recovered = ""
        if self.fallen and activate:
            ok, why = await self._wait_upright()
            if not ok:
                return Result(False, f"won't stand up: {why}. ({FELL_MESSAGE})")
            # Relax every latched controller first: activating one clears its e-stop.
            latched = [c for c in self.leg_controllers if states.get(c) == "active"]
            if latched:
                req = SwitchController.Request()
                req.deactivate_controllers = latched
                req.strictness = SwitchController.Request.BEST_EFFORT
                await self._call(self.switch_client, req)
                states = await self.controller_states()
            self.walk_since = None
            log.info("recovering from the fall in %s: %s", self.fallen, why)
            self.fallen = None
            recovered = "recovered from the fall and stood back up; "
        not_ready = [c for c in activate if states.get(c) not in LOADED]
        if not_ready:  # the launch's spawner missed it: load it now (a few seconds)
            log.warning("loading %s (state %s)", not_ready, [states.get(c, "not loaded") for c in not_ready])
            await asyncio.to_thread(spawn, not_ready)
            states = await self.controller_states()
            not_ready = [c for c in activate if states.get(c) not in LOADED]
            if not_ready:
                return Result(False, f"could not load controllers {not_ready}; see /tmp/gemini-pupper/robot.log")
        leaving = self.state_now
        if (isinstance(leaving, Trick) and leaving.interactive and leaving.exit_clip and activate
                and controller not in (leaving.controller, leaving.walk_controller if leaving.walks else None)):
            # Leave an interactive trick through its own exit clip before another motion
            # takes the legs (not before relaxing the motors: lying down from a sit is fine).
            if states.get(leaving.controller) == "active":
                res = await self.play_clip(leaving, leaving.exit_clip)
                if not res.ok:
                    log.warning("exit clip: %s", res.message)
            elif leaving.walks and states.get(leaving.walk_controller) == "active":
                # Walking in its stance: give the legs back to the trick's own policy,
                # started at its exit clip, and let it play.
                res = await self.switch_to(leaving.controller, start_clip=leaving.exit_clip)
                if res.ok:
                    await asyncio.sleep(TRICK_INIT_S + TRICK_FADE_S + leaving.clip_length(leaving.exit_clip) + 0.3)
                else:
                    log.warning("exit clip: %s", res.message)
            states = await self.controller_states()
            deactivate = sorted(self.leg_controllers - set(activate))
        trick = self.state_of(controller)
        if isinstance(trick, Trick) and trick.interactive and controller == trick.controller and states.get(controller) != "active":
            await self._set_start_clip(controller, start_clip)  # read by the controller when it activates
        req = SwitchController.Request()
        req.activate_controllers = [c for c in activate if states[c] != "active"]
        req.deactivate_controllers = [c for c in deactivate if states.get(c) == "active"]
        if states.get(LOCOMOTION_CONTROLLER) == "active" and self.walk_since is None:
            self.walk_since = time.monotonic() - WALK_READY_S  # active before we started
        if not req.activate_controllers and not req.deactivate_controllers:
            self.state_now = self.state_of(controller)
            if controller == LOCOMOTION_CONTROLLER:
                await self._wait_walk_ready()
            return Result(True, recovered + "already in that state")
        req.strictness = SwitchController.Request.STRICT
        res = await self._call(self.switch_client, req)
        if res is None or not res.ok:
            why = "no answer from controller_manager" if res is None else "refused by controller_manager"
            log.warning("switch %s -> %s: %s", req.deactivate_controllers, req.activate_controllers, why)
            return Result(False, f"controller switch failed (activate {activate}): {why}")
        self.state_now = self.state_of(controller)
        if LOCOMOTION_CONTROLLER in req.activate_controllers:
            self.walk_since = time.monotonic()
        if LOCOMOTION_CONTROLLER in req.deactivate_controllers:
            self.walk_since = None
        if controller == LOCOMOTION_CONTROLLER:
            await self._wait_walk_ready()
        return Result(True, recovered + "ok")

    def state_of(self, controller: str | None) -> str | Trick:
        """The state a controller holding the legs means (None: MOTORS_IDLE). A walking
        trick's walking policy is that trick's state too."""
        if controller is None:
            return MOTORS_IDLE
        for t in self.tricks.values():
            if t.kind == "policy" and (t.controller == controller or (t.walks and t.walk_controller == controller)):
                return t
        return LOCOMOTION

    def walking_controller(self, state) -> str:
        """The controller move, turn and arc drive in a state: a walking trick's own
        walking policy, else the locomotion policy (which they switch to first)."""
        return state.walk_controller if isinstance(state, Trick) and state.walks else LOCOMOTION_CONTROLLER

    def speeds(self, state=None) -> Speeds:
        """Default speeds and limits in a state (default: robot.state_after_queue, where
        Pupper will be when a move asked for now runs). A walking trick has its own,
        from its trick.yaml; everywhere else, the locomotion policy's."""
        state = self.state_after_queue if state is None else state
        w = state.walk if isinstance(state, Trick) and state.walks else None
        if w is None:
            return SPEEDS
        limits = MoveLimits(vx_min=min(0.05, w.max_forward), vy_min=min(0.05, w.max_sideways), wz_min=min(10.0, w.turn),
                            vx_max=w.max_forward, vy_max=w.max_sideways, wz_max=w.turn,
                            duration_max=LIMITS.duration_max, distance_max=LIMITS.distance_max)
        return Speeds(walk=w.speed, turn=w.turn, limits=limits, where=f" (in {state.name})")

    async def _set_start_clip(self, controller: str, clip: str) -> None:
        """Set an interactive trick's start_clip parameter ("" = its first clip)."""
        client = self.param_clients.get(controller)
        if client is None:
            client = self.param_clients[controller] = self.node.create_client(SetParameters, f"/{controller}/set_parameters")
        req = SetParameters.Request()
        req.parameters = [Parameter(name="start_clip", value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=clip))]
        if await self._call(client, req) is None:
            log.warning("could not set %s start_clip=%r", controller, clip)

    async def _wait_walk_ready(self) -> None:
        """Wait until gemini_walk has eased the legs to standing and faded its
        policy in. A walk sent before that runs on a half-started controller."""
        wait = (self.walk_since or time.monotonic()) + WALK_READY_S - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)

    def send_speed(self, vx: float, vy: float, wz_deg: float) -> None:
        """One speed command to the locomotion policy: vx forward m/s, vy left m/s,
        wz_deg counter-clockwise deg/s. cmd_vel_mux forgets it after 0.5 s."""
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz_deg) / 57.2958
        self.cmd_pub.publish(t)

    async def _adopt_running_state(self) -> None:
        """A restarted server finds the legs as the last one left them: start from
        that state. An interactive trick still running (the robot sits) is taken up
        again, so the next motion stands up through its exit clip instead of
        jumping out of it."""
        states = await self.controller_states()
        active = [c for c in sorted(self.leg_controllers) if states.get(c) == "active"]
        if active:
            self.state_now = self.state_of(active[0])
            if self.commands.empty():  # a request may already be queued: it set the state after it
                self.state_after_queue = self.state_now
            log.info("found %s holding the legs; starting in state %s", active[0], state_name(self.state_now))

    # ---------------------------------------------------------------- the queue
    async def _run_queue(self) -> None:
        await self._adopt_running_state()
        while True:
            name, job, then = await self.commands.get()
            self._running_job_then = then
            self.current = asyncio.create_task(job())
            res = Result(False, "cancelled")
            try:
                res = await self.current
                if not res.ok:
                    self.send_speed(0, 0, 0)
                    cleared = await self.clear_queue()
                    res = Result(False, f"{res.message}; {cleared.message}")
                (log.info if res.ok else log.warning)("%s: %s", name, res.message)
            except asyncio.CancelledError:
                self.send_speed(0, 0, 0)
                log.info("%s: cancelled", name)
            except Exception as e:
                res = Result(False, f"{type(e).__name__}: {e}")
                self.send_speed(0, 0, 0)
                await self.clear_queue()
                log.exception("%s failed", name)
            finally:
                self.outcome_id += 1
                self.outcomes.append({"id": self.outcome_id, "job": name, **res.as_dict()})
                self.current = None

    async def queue(self, name: str, job: Job, seconds: float, *, then) -> Result:
        """Put ``job`` in line behind everything already queued, and return at once.

        ``job`` is an async function with no arguments. It runs *later*, when the
        queue reaches it, and what it returns (the outcome) goes to the debug log.
        What this returns now (the job was accepted) is what Gemini hears.

        ``seconds``: about how long the job takes, for the estimate Gemini hears.
        ``then``: the state Pupper is in once this job is done (LOCOMOTION,
        MOTORS_IDLE, or a trick); it becomes robot.state_after_queue. Jobs that
        don't move the legs (wait) pass UNCHANGED.
        """
        if self.stopped_during_call():
            return Result(False, f"{name} not queued: a stop came in while this request was being handled")
        if then is not UNCHANGED:
            self.state_after_queue = then
        now = time.monotonic()
        self.busy_until = max(self.busy_until, now) + seconds
        await self.commands.put((name, job, self.state_after_queue))
        return Result(True, f"{name} queued; the queue should finish in {self.busy_until - now:.1f} s")

    def stopped_during_call(self) -> bool:
        """Did a stop come in after the tool call being handled began?"""
        began = CALL_STOPS.get()
        return began is not None and began != self.stops

    def status(self) -> dict:
        return {
            "queued": self.commands.qsize(),
            "running": self.current is not None,
            "seconds_left": round(max(0.0, self.busy_until - time.monotonic()), 1),
            "following": self.following,
            "state": state_name(self.state_now),
            "state_after_queue": state_name(self.state_after_queue),
            "outcomes": list(self.outcomes),
        }

    async def wait(self, duration: float) -> Result:
        async def job():
            await asyncio.sleep(duration)
            return Result(True, f"waited {duration} s")

        return await self.queue(f"wait {duration}s", job, duration, then=UNCHANGED)

    async def clear_queue(self) -> Result:
        n = 0
        while not self.commands.empty():
            self.commands.get_nowait()
            n += 1
        self.busy_until = time.monotonic()
        running = self.current is not None and not self.current.done()
        self.state_after_queue = self._running_job_then if running else self.state_now
        return Result(True, f"cleared {n} queued commands")

    # ---------------------------------------------------------------- tricks
    def action_ids(self) -> dict[str, tuple[Trick, str]]:
        """Interactive action ids Gemini uses -> (trick, clip). A clip name, prefixed with the
        trick's name only when two tricks share it."""
        pairs = [(t, clip) for t in self.tricks.values() for clip in t.actions]
        names = [clip for _, clip in pairs]
        return {(clip if names.count(clip) == 1 else f"{t.name}.{clip}"): (t, clip) for t, clip in pairs}

    async def play_clip(self, trick: Trick, clip: str) -> Result:
        """Request a clip and wait until the controller has played it through."""
        self.clip_status[trick.controller] = ("", False)  # forget an earlier activation's status
        self.clip_pubs[trick.controller].publish(String(data=clip))
        length = trick.clip_length(clip)
        longest = max((trick.clip_length(c["name"]) for c in trick.clips), default=0.0)
        deadline = time.monotonic() + longest + 1.0  # the current clip finishes first
        while self.clip_status.get(trick.controller, ("", False))[0] != clip:
            if time.monotonic() > deadline:
                return Result(False, f"{trick.name} did not start {clip} (is the controller running?)")
            await asyncio.sleep(0.05)
        deadline = time.monotonic() + length + 1.0
        while True:
            name, holding = self.clip_status.get(trick.controller, ("", False))
            if name != clip or holding or time.monotonic() > deadline:
                break
            await asyncio.sleep(0.05)
        return Result(True, f"played {clip}")

    async def play_reference_trick(self, trick: Trick) -> Result:
        """Feed a reference (e.g. the jump) to the running locomotion policy, standing still."""
        res = await self.switch_to(LOCOMOTION_CONTROLLER)  # waits until the policy is ready
        if not res.ok:
            return res
        self.send_speed(0, 0, 0)
        await asyncio.sleep(0.5)  # settle, standing still
        self.ref_pub.publish(Float32MultiArray(data=reference_message_data(trick)))
        await asyncio.sleep(trick.total_time)
        if self.fallen:
            return Result(False, f"{trick.name} ended in a fall: {FELL_MESSAGE}")
        return Result(True, f"played {trick.name}; standing, in locomotion mode")

    # ---------------------------------------------------------------- follow mode
    async def stop_following(self) -> None:
        """Turn the person follower off (if it is on)."""
        if self.following:
            self.following = False  # at once: nothing should count on it while the service answers
            await self._call(self.follow_off, Trigger.Request())

    async def set_following(self, enabled: bool) -> Result:
        if not enabled:
            await self.stop_following()
            self.send_speed(0, 0, 0)
            return Result(True, "stopped following")
        if self.following:
            return Result(True, "already following the nearest person")
        await self.clear_queue()
        if self.current and not self.current.done():
            self.current.cancel()
        res = await self.switch_to(LOCOMOTION_CONTROLLER)
        self.state_after_queue = LOCOMOTION
        if not res.ok:
            return res
        if self.stopped_during_call():  # stop came in while the legs were switching
            return Result(False, "stopped before follow mode started; not following")
        r = await self._call(self.follow_on, Trigger.Request())
        if r is None or not r.success:
            return Result(False, "person_follower did not respond")
        if self.stopped_during_call():  # ... or while the follower was starting
            await self._call(self.follow_off, Trigger.Request())
            return Result(False, "stopped before follow mode started; not following")
        self.following = True
        return Result(True, "following the nearest person. Any new motion (walk, turn, trick, stand, lie down) will end follow mode.")

    # ---------------------------------------------------------------- stopping
    def _on_estop(self, _msg: Empty) -> None:
        self.gamepad_estop_at = time.monotonic()
        self.fallen = None  # a person stopped it on purpose; standing up is their call
        asyncio.run_coroutine_threadsafe(self.emergency_stop(), self.loop)

    async def emergency_stop(self) -> Result:
        """Gamepad e-stop: cancel everything and relax every leg controller."""
        await self.immediate_stop()
        self.state_now = self.state_after_queue = MOTORS_IDLE  # no exit clip: the motors just relax
        states = await self.controller_states()
        active = [c for c in sorted(self.leg_controllers) if states.get(c) == "active"]
        if active:
            req = SwitchController.Request()
            req.deactivate_controllers = active
            req.strictness = SwitchController.Request.BEST_EFFORT
            await self._call(self.switch_client, req)
        self.walk_since = None
        log.warning("e-stop: relaxed %s", active or "nothing (no leg controller was active)")
        return Result(True, "e-stop: motors relaxed")

    async def immediate_stop(self, relax_playing_trick: bool = False) -> Result:
        """Clear the queue, cancel the running job, stop following and stand still.

        Cancelling a walk is enough to stop it: the locomotion policy only moves while
        it is sent speeds. A trick policy plays its motion by itself, so with
        relax_playing_trick (the stop tool) a trick that was still playing is
        stopped the only way it can be: by relaxing the motors, like the gamepad
        e-stop. A trick that finished and only holds its pose is left as it is.
        """
        self.stops += 1  # requests still being handled won't queue anything now
        self.send_speed(0, 0, 0)  # stand still first, before anything that waits
        running = self.current is not None and not self.current.done()
        cleared = await self.clear_queue()
        if running:
            self.current.cancel()
        self.state_after_queue = self.state_now  # nothing else is coming
        await self.stop_following()
        self.send_speed(0, 0, 0)
        if not (relax_playing_trick and running):
            return Result(True, f"stopped; {cleared.message}")
        await asyncio.sleep(0.1)  # a job cancelled while switching to a trick may still finish the switch
        states = await self.controller_states()
        playing = [t for t in self.tricks.values() if t.kind == "policy" and states.get(t.controller) == "active"]
        if not playing:
            return Result(True, f"stopped; {cleared.message}")
        req = SwitchController.Request()
        req.deactivate_controllers = [c for c in sorted(self.leg_controllers) if states.get(c) == "active"]
        req.strictness = SwitchController.Request.BEST_EFFORT
        await self._call(self.switch_client, req)
        self.state_now = self.state_after_queue = MOTORS_IDLE
        self.walk_since = None
        log.warning("stop: %s was still playing; relaxed the motors", playing[0].name)
        return Result(True, f"stopped; {cleared.message}. {playing[0].name} was still playing, so the motors"
                            " relaxed and the robot is lying down; call stand_up to go on")

    async def mode(self) -> Result:
        """What the robot is doing right now, in words (the robot_status tool)."""
        states = await self.controller_states()
        if not states:
            return Result(False, "controller_manager is not running")
        trick = next((t for t in self.tricks.values() if t.kind == "policy" and states.get(t.controller) == "active"), None)
        walking = next((t for t in self.tricks.values() if t.walks and states.get(t.walk_controller) == "active"), None)
        if self.fallen:
            return Result(True, f"mode: fallen over (tilt e-stop in {self.fallen}). {FELL_MESSAGE}")
        mode = (
            "locomotion, following a person" if self.following
            else "locomotion" if states.get(LOCOMOTION_CONTROLLER) == "active"
            else (f"trick {walking.name}, walking in its {walking.end_pose} stance (move, turn and arc walk slowly in it; "
                  f"actions: {', '.join(walking.actions)})") if walking
            else f"trick {trick.name} ({trick.end_pose}; actions: {', '.join(trick.actions)})" if trick and trick.actions
            else f"trick {trick.name}, holding its end pose ({trick.end_pose})" if trick
            else "motors idle (lying down)"
        )
        return Result(True, f"mode: {mode}")


# Another name for Robot, for staff scripts written against tool_server.ToolServer.
ToolServer = Robot
