#!/usr/bin/env python3
"""Desk tests for the hackathon features: memory, the screen and hearing, reminders,
seeing things, and routines learned on the fly. No robot, no ROS, no Gemini needed.

    python3 lab/test.py hackathon     # just these
    python3 lab/test.py               # the lab's 166 checks, then these

They run on test_tools.py's fake robot and fake clock, with the memory in a
temporary file (never your ~/.config). Each group says what it checks; a failure
prints what was asked and what came back.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_TMP = Path(tempfile.mkdtemp(prefix="pupper_memory_test_"))
os.environ.setdefault("PUPPER_MEMORY_FILE", str(_TMP / "memory.json"))
os.environ.setdefault("PUPPER_LOG_DIR", str(_TMP / "logs"))

import test_tools as tt  # noqa: E402  a fake ROS, and lab/ loaded
from test_tools import LOCOMOTION_CONTROLLER, Rig, check, tools  # noqa: E402

import fisheye_geom  # noqa: E402
import hackathon_alerts as alerts  # noqa: E402
import hackathon_explore as explore  # noqa: E402
import hackathon_eyes as eyes  # noqa: E402
import hackathon_learn as learn  # noqa: E402
import hackathon_log as hlog  # noqa: E402
import hackathon_memory as memory  # noqa: E402
import hackathon_vision as vision  # noqa: E402

memory.MEMORY_FILE = _TMP / "memory.json"  # whatever the environment said: never the real file


def fresh_memory() -> None:
    memory.save(memory._empty())
    alerts.clear_display()
    alerts._last_sound_alert = 0.0
    alerts.SOUND_ALERTS_ON = False  # as at every start: opt in
    vision.VISION.update(time=0.0, detections=[])
    vision.FREESPACE.update(time=0.0, strips=[], mode="", eyes_url="")
    vision.SCENE.update(time=0.0, text="")
    eyes.TARGET.update(time=0.0, name="", box=None)
    eyes.WANTED.update(name="", since=0.0)
    explore.EXPLORING.update(name="", since=0.0)
    learn.LEARNING = None


def project(x: float, y: float, z: float) -> tuple[float, float]:
    """Lab 5's forward lens model (follow_me_pupper/robot/fisheye_converter.py): a ray -> frame fractions."""
    g = fisheye_geom
    r2 = x * x + y * y
    d1 = math.sqrt(r2 + z * z)
    k2 = g.XI * d1 + z
    d2 = math.sqrt(r2 + k2 * k2)
    den = g.ALPHA * d2 + (1 - g.ALPHA) * k2
    return (g.FX * x / den + g.CX) / g.FRAME_W, (g.FY * y / den + g.CY) / g.FRAME_H


def frame_point(bearing_deg: float, lat_deg: float) -> tuple[float, float]:
    """Where a thing at bearing (+ right) and latitude (+ down) appears, as frame fractions."""
    lat, lon = math.radians(lat_deg), math.radians(bearing_deg)
    return project(math.cos(lat) * math.sin(lon), math.sin(lat), math.cos(lat) * math.cos(lon))


def floor_lat(distance_m: float, cam_height_m: float = 0.22) -> float:
    """The latitude of a point on the floor distance_m ahead, for a level camera cam_height_m up."""
    return math.degrees(math.atan2(cam_height_m, max(distance_m, 0.02)))


def row_of(distance_m: float) -> float:
    """Lab 5's row (nearness) of a floor point distance_m ahead of a level camera 0.22 m up."""
    return (floor_lat(distance_m) + 90.0) / 180.0


def box(cx: float, w: float, h: float, bottom: float = 1.0) -> list[float]:
    """A detection box from its center x, width, height and bottom edge (fractions of the frame)."""
    return [cx - w / 2, bottom - h, w, h]


def see(*dets: tuple[str, list[float]]) -> None:
    vision.update_detections([{"label": label, "score": 0.9, "box": b} for label, b in dets], 640, 480)


async def call(rig: Rig, tool: str, **args):
    """Call a tool the way Gemini does (Rig.call can't take a parameter called name)."""
    res = await tools.call(rig.ts, tool, args)
    await rig.settle()
    return res


def turns(rig: Rig) -> list:
    return [m for m in tt.walk_twists(rig) if abs(m.angular.z) > 1e-9]


# ----------------------------------------------------------------------------- memory
async def memory_tests() -> None:
    print("Hackathon A: memory (profiles, facts, where things are)")
    fresh_memory()
    rig = Rig()
    res = await call(rig, "switch_profile", name="Devesh A.")
    check("switch_profile makes a profile from the name", res.ok and "devesh_a" in res.message, res.message)
    res = await call(rig, "remember", fact="likes tennis balls")
    check("remember saves a fact", res.ok and "tennis" in res.message and "devesh_a" in res.message, res.message)
    res = await call(rig, "remember", fact="likes tennis balls")
    check("... the same fact twice is kept once", res.ok and "already" in res.message, res.message)
    res = await call(rig, "remember", fact="x")
    check("... a one-letter 'fact' is refused with an example", not res.ok and "e.g." in res.message, res.message)
    await call(rig, "found_object", name="keys", where="on the desk")
    res = await call(rig, "recall")
    check("recall says the facts and the things", res.ok and "tennis" in res.message and "keys on the desk" in res.message, res.message)
    check("where_is finds a thing by name", (memory.where_is("my keys") or "").startswith("on the desk"), str(memory.where_is("my keys")))
    prompt = memory.memory_prompt()
    check("the prompt section names the person, the fact and the thing",
          "devesh_a" in prompt and "tennis" in prompt and "keys" in prompt, prompt)
    res = await call(rig, "switch_profile", name="Guest Two")
    check("another person starts empty", res.ok and "nothing remembered" in res.message, res.message)
    check("... and recall lists the other people", "devesh_a" in (await call(rig, "recall")).message)
    await call(rig, "switch_profile", name="Devesh A.")
    res = await call(rig, "forget", what="tennis")
    check("forget by a word drops that fact", res.ok and "tennis" in res.message, res.message)
    res = await call(rig, "forget", what="tennis")
    check("... forgetting it again says there is nothing, and lists the facts", not res.ok and "nothing" in res.message, res.message)
    res = await call(rig, "forget", what="everything")
    check("forget everything clears the profile", res.ok and "forgot everything" in res.message and
          memory.active_profile(memory.load())["objects"] == {}, res.message)
    res = memory.reset_memory("all")
    check("reset all leaves an empty memory", res.ok and memory.load()["profiles"] == {}, res.message)
    memory.MEMORY_FILE.write_text("{not json")
    check("a broken memory file reads as empty, not a crash", memory.load()["profiles"] == {})
    fresh_memory()
    rig.close()


# ----------------------------------------------------------------------------- screen and events
async def screen_tests() -> None:
    print("Hackathon B: the screen and events")
    fresh_memory()
    rig = Rig()
    res = await call(rig, "show_text", text="DOORBELL", detail="14:02", color="red", seconds=5)
    check("show_text for a hearing person: not shown, say it instead (the screen has text only for the deaf)",
          not res.ok and "say it out loud" in res.message and alerts.display_state() is None and not memory.needs_text(), res.message)
    await call(rig, "remember", fact="is deaf")
    check("... remembering that the person is deaf turns screen text on", memory.needs_text())
    before = alerts.EVENTS.last_id
    res = await call(rig, "show_text", text="DOORBELL", detail="14:02", color="red", seconds=5)
    check("show_text is accepted", res.ok and "DOORBELL" in res.message, res.message)
    shown = alerts.display_state()
    check("... and it is on the screen in that color", shown is not None and shown["text"] == "DOORBELL"
          and shown["color"] == alerts.COLORS["red"], str(shown))
    evs = alerts.EVENTS.since(before)
    check("... and the app gets an event carrying the display", len(evs) == 1 and evs[0]["display"]["text"] == "DOORBELL"
          and evs[0]["kind"] == "display", str(evs))
    check("since(last) is empty: nothing is delivered twice", alerts.EVENTS.since(alerts.EVENTS.last_id) == [])
    res = await call(rig, "show_text", text="   ")
    check("empty text is refused", not res.ok, res.message)
    res = await call(rig, "show_text", text="HI", color="hotpink")
    check("an unknown color is refused by the declaration (enum)", not res.ok and "must be one of" in res.message, res.message)
    alerts.DISPLAY["until"] = time.time() - 1
    check("after its seconds the screen is clear", alerts.display_state() is None)
    rig.close()


# ----------------------------------------------------------------------------- hearing
async def sound_tests() -> None:
    print("Hackathon B: hearing for people who can't (sound alerts)")
    fresh_memory()
    rig = Rig()
    res = alerts.sound_detected(rig.ts, above_db=40.0)
    check("sound alerts are off until a person asks (opt in)", "off" in res.message, res.message)
    res = alerts.hearing_detected(rig.ts, {"cls": "bang", "score": 50.0})
    check("... for the calibrated detector too", "off" in res.message, res.message)
    res = await call(rig, "sound_alerts", enabled=True)
    check("sound_alerts(true) turns them on", res.ok and "on" in res.message, res.message)
    before = alerts.EVENTS.last_id
    res = alerts.sound_detected(rig.ts, above_db=10.0, duration_s=0.2)
    check("a sound under the threshold is ignored, and says the threshold", res.ok and "ignored" in res.message and "+18" in res.message, res.message)
    res = alerts.sound_detected(rig.ts, above_db=20.0, duration_s=0.3)
    check("a loud sound (+20 dB, medium) alerts", res.ok and res.message.startswith("LOUD SOUND"), res.message)
    evs = alerts.EVENTS.since(before)
    check("... the screen turns red with LOUD SOUND", alerts.display_state() and alerts.display_state()["text"] == "LOUD SOUND"
          and alerts.display_state()["color"] == alerts.COLORS["red"], str(alerts.display_state()))
    check("... and Gemini is NOT told (Pupper shows sounds, it doesn't talk about them)",
          len(evs) == 1 and evs[0]["kind"] == "sound" and evs[0]["for_gemini"] == "", str(evs))
    res = alerts.sound_detected(rig.ts, above_db=40.0)
    check("a second sound right after is ignored (one alert at a time)", "already" in res.message, res.message)
    alerts._last_sound_alert = 0.0
    res = alerts.sound_detected(rig.ts, above_db=40.0)
    check("+40 dB is a VERY LOUD SOUND", res.message.startswith("VERY LOUD"), res.message)
    alerts._last_sound_alert = 0.0
    rig.ts.busy_until = rig.ts.now() + 3.0  # a motion job is running
    res = alerts.sound_detected(rig.ts, above_db=40.0)
    check("while the robot moves, a loud sound is its own motors: ignored", "motors" in res.message, res.message)
    rig.ts.busy_until = rig.ts.now() - 10.0
    res = await call(rig, "sound_alerts", enabled=False)
    check("sound_alerts(false) turns them off", res.ok and "off" in res.message, res.message)
    res = alerts.sound_detected(rig.ts, above_db=40.0)
    check("... and then nothing alerts", "off" in res.message, res.message)
    res = await call(rig, "sound_alerts", enabled=True, sensitivity="high")
    check("sound_alerts(true, high) is on with a +12 dB threshold", res.ok and "12" in res.message, res.message)
    check("... and the setting survives in memory", memory.load()["settings"].get("sound_sensitivity") == "high")
    res = await call(rig, "sound_alerts", enabled=True, sensitivity="loud")
    check("an unknown sensitivity is refused with the choices", not res.ok and "low" in res.message, res.message)

    # The calibrated detector (hearing/listener.py) reports what it heard by type.
    alerts._last_sound_alert = 0.0
    before = alerts.EVENTS.last_id
    res = alerts.hearing_detected(rig.ts, {"cls": "alarm", "score": 40.0, "info": {"tone_hz": 3100.0}})
    evs = alerts.EVENTS.since(before)
    check("a detected alarm turns the screen red with ALARM, and Gemini is not told",
          res.message.startswith("ALARM") and alerts.display_state()["text"] == "ALARM"
          and alerts.display_state()["color"] == alerts.COLORS["red"] and len(evs) == 1 and evs[0]["for_gemini"] == "", res.message)
    for cls, words in (("siren", "SIREN"), ("doorbell", "DOORBELL"), ("crash", "CRASH"),
                       ("knock", "KNOCKING"), ("baby", "BABY CRYING")):
        alerts._last_sound_alert = 0.0
        alerts.hearing_detected(rig.ts, {"cls": cls, "score": 50.0, "info": {"label": "x", "confidence": 0.8}})
        check(f"a detected {cls} shows {words} in red", alerts.display_state()["text"] == words
              and alerts.display_state()["color"] == alerts.COLORS["red"], str(alerts.display_state()))
    alerts._last_sound_alert = 0.0
    alerts.clear_display()
    res = alerts.hearing_detected(rig.ts, {"cls": "shout", "score": 50.0})
    check("a shout doesn't alert (shouting is not on the urgent list)",
          "not an urgent sound" in res.message and alerts.display_state() is None, res.message)
    alerts._last_sound_alert = 0.0
    alerts.hearing_detected(rig.ts, {"cls": "bang", "score": 50.0})
    res = alerts.hearing_detected(rig.ts, {"cls": "knock", "score": 60.0, "info": {"label": "Knock", "confidence": 0.7}})
    check("a knock right after the meter's LOUD BANG replaces it with KNOCKING", alerts.display_state()["text"] == "KNOCKING", res.message)
    res = alerts.hearing_detected(rig.ts, {"cls": "siren", "score": 80.0})
    check("... but another sound right after is still one alert at a time", "already" in res.message, res.message)
    alerts._last_sound_alert = 0.0
    res = alerts.hearing_detected(rig.ts, {"cls": "speech", "score": 0.9})
    check("a sound that isn't on the urgent list doesn't alert", "not an urgent" in res.message, res.message)
    alerts._last_sound_alert = 0.0
    rig.ts.busy_until = rig.ts.now() + 3.0
    res = alerts.hearing_detected(rig.ts, {"cls": "bang", "score": 50.0})
    check("a detected bang while moving is its own motors: ignored", "motors" in res.message, res.message)
    rig.ts.busy_until = rig.ts.now() - 10.0

    class FakeHearing:  # the robot server's detector, without a mic
        enabled = False
        def __init__(self): self.calls = []
        async def set_enabled(self, on, sensitivity="medium"):
            self.calls.append((on, sensitivity)); self.enabled = on
            return True, "listening with the calibrated detector"
    rig.ts.hearing = FakeHearing()
    res = await call(rig, "sound_alerts", enabled=True, sensitivity="low")
    check("with a calibrated detector, sound_alerts(true) starts it", rig.ts.hearing.calls[-1] == (True, "low") and res.ok, res.message)
    alerts._last_sound_alert = 0.0
    res = alerts.sound_detected(rig.ts, above_db=40.0)
    check("... and the browser's own measurement is then ignored", "calibrated" in res.message, res.message)
    res = await call(rig, "sound_alerts", enabled=False)
    check("sound_alerts(false) stops the detector (the mic isn't checked)", rig.ts.hearing.calls[-1][0] is False and not rig.ts.hearing.enabled, res.message)
    del rig.ts.hearing
    rig.close()


# ----------------------------------------------------------------------------- seeing things
async def vision_tests() -> None:
    print("Hackathon D: seeing things (the obstacle rules)")
    fresh_memory()
    rig = Rig()
    real_ask, eyes.ask_eyes = eyes.ask_eyes, (lambda url, path, payload, timeout=0: {"found": False, "source": "gemma"})  # no eyes service here
    n = vision.update_detections([{"label": "chair", "score": 0.8, "box": [0.3, 0.4, 0.4, 0.6]}, {"bad": 1}, {"label": "x", "box": [0, 0, 0, 0]}])
    check("update_detections keeps the good entries and drops the bad", n == 1, str(n))
    vision.VISION["time"] = 0.0
    check("stale detections are no eyes: path_blocked is None", vision.path_blocked(0.4, 0.0) is None)
    res = await call(rig, "move", direction="forward", duration=1.0)
    check("... and a forward move is accepted", res.ok, res.message)

    see(("chair", box(0.5, 0.4, 0.6)))  # a chair fills the middle of the view, close
    res = await call(rig, "move", direction="forward", duration=1.0)
    check("a close chair straight ahead refuses a forward move, naming it", not res.ok and "blocked" in res.message
          and "chair" in res.message and "in front" in res.message, res.message)
    check("... and nothing was queued (no speeds sent)", not tt.walk_twists(rig)[len(tt.walk_twists(rig)):])
    res = await call(rig, "move", direction="backward", duration=1.0)
    check("backing away from it is fine", res.ok, res.message)
    res = await call(rig, "turn", direction="left", angle=90)
    check("turning in place is fine", res.ok, res.message)
    see(("person", box(0.15, 0.25, 0.5)))  # someone close on the left
    res = await call(rig, "move", direction="left", duration=1.0)
    check("a close person on the left refuses a sidestep left", not res.ok and "person" in res.message and "left" in res.message, res.message)
    res = await call(rig, "move", direction="right", duration=1.0)
    check("... but not a sidestep right", res.ok, res.message)
    res = await call(rig, "move", direction="forward", duration=1.0)
    check("... nor walking forward (they are off to the side)", res.ok, res.message)
    see(("cup", box(0.5, 0.1, 0.1, 0.5)))  # a small cup far ahead: its bottom edge near the horizon
    res = await call(rig, "move", direction="forward", duration=1.0)
    check("a small far cup does not block", res.ok, res.message)
    res = vision.what_i_see()
    check("what_i_see describes the frame (what_is_around reads it)", res.ok and "cup straight ahead" in res.message, res.message)

    # Something steps in the way mid-walk: the job stops early and Gemini is told.
    vision.VISION.update(time=0.0, detections=[])
    publish, sent = rig.robot.publish, [0]

    def publish_and_step_in(topic, msg):
        publish(topic, msg)
        if topic == "/llm_cmd_vel" and abs(msg.linear.x) > 0:
            sent[0] += 1
            if sent[0] == 20:  # after 1 s of walking
                see(("person", box(0.5, 0.5, 0.7)))

    rig.robot.publish = publish_and_step_in
    before, start = alerts.EVENTS.last_id, len(rig.records)
    res = await call(rig, "move", direction="forward", duration=3.0)
    rig.robot.publish = publish
    check("a 3 s walk is accepted when the way is clear", res.ok, res.message)
    outcomes = rig.outcomes(start)
    check("... but stops early when a person steps in front", any("stopped after 1.0 s of 3.0" in o and "person" in o for o in outcomes), str(outcomes))
    check("... sending at most about 1 s of speeds", sent[0] <= 21, str(sent[0]))
    evs = alerts.EVENTS.since(before)
    check("... and Gemini is told it stopped early", len(evs) == 1 and evs[0]["kind"] == "obstacle" and "stopped" in evs[0]["for_gemini"], str(evs))

    eyes.ask_eyes = real_ask
    rig.close()


# ----------------------------------------------------------------------------- depth sense and go_to
def depth(**per_strip) -> None:
    """Post a free-space summary: depth(center=0.3) in metres, or depth(center=("near", 0.6)) for relative mode."""
    strips, mode = [], "metric"
    for name in vision.STRIPS:
        v = per_strip.get(name.replace("-", "_"))
        if isinstance(v, tuple) and v[0] == "row":
            strips.append({"name": name, "closest_m": None, "near_frac": None, "nearness": v[1]})
        elif isinstance(v, tuple):
            mode = "relative"
            strips.append({"name": name, "closest_m": None, "near_frac": v[1]})
        else:
            strips.append({"name": name, "closest_m": 2.0 if v is None else v, "near_frac": 0.05 if v is None else 0.9})
    vision.update_freespace(strips, mode, "http://eyes.test:8100")


class World:
    """A kinematic stand-in for the camera: one thing at a bearing and a distance from Pupper.
    Every speed Pupper sends moves it; every tick the eyes service 'sees' it as a box."""

    H_AT_1M = 0.2  # box height (fraction of the frame) when the thing is 1 m away

    def __init__(self, rig: Rig, name: str, bearing_deg: float, distance_m: float, see: bool = True, every: int = 1):
        self.rig, self.name, self.bearing, self.distance, self.see = rig, name, bearing_deg, distance_m, see
        self.every = every  # a fix only every this many ticks (Gemma is slow); the depth strip updates every tick
        self.ticks, self.turn_ticks, self.walk_ticks, self.first_walk_bearing = 0, 0, 0, None
        self.on_tick = None
        self.obstacle: dict = {}  # something else the depth sense sees, e.g. {"center": 0.25}: a cable on the route
        self._publish = rig.robot.publish
        rig.robot.publish = self.publish
        self.update()

    def box(self) -> list[float]:
        """The thing as a box through Lab 5's lens: its base on the floor at its bearing and distance
        sets the bottom edge's centre; its height shrinks with distance."""
        h = min(0.95, self.H_AT_1M / max(self.distance, 0.05))
        u, v = frame_point(self.bearing, floor_lat(self.distance))
        return box(u, h * 0.6, h, min(0.99, v))

    def update(self) -> None:
        if self.see and self.ticks % self.every == 0:
            eyes.update_target(self.name, self.box(), 0.8, "eyes")
        if self.see:  # the depth sense sees the thing's base in its strip (and floor elsewhere), as a Lab 5 row
            strip = eyes.strip_at(self.bearing).replace("-", "_")
            depth(**{strip: ("row", round(row_of(self.distance), 3)), **self.obstacle})

    def publish(self, topic, msg):
        self._publish(topic, msg)
        if topic != "/llm_cmd_vel":
            return
        dt = 1.0 / tt.LLM_CMD_RATE if hasattr(tt, "LLM_CMD_RATE") else 0.05
        vx, wz = msg.linear.x, math.degrees(msg.angular.z)
        if abs(vx) + abs(wz) > 1e-9:
            self.ticks += 1
            if abs(wz) > 1e-9:
                self.turn_ticks += 1
            if vx > 1e-9:
                self.walk_ticks += 1
                if self.first_walk_bearing is None:
                    self.first_walk_bearing = self.bearing
            self.bearing += wz * dt  # a left turn (wz > 0) swings a thing on the left (bearing < 0) toward ahead
            self.distance -= vx * math.cos(math.radians(self.bearing)) * dt
        if self.on_tick:
            self.on_tick(self)
        self.update()

    def close(self) -> None:
        self.rig.robot.publish = self._publish


async def eyes_tests() -> None:
    print("Hackathon F: the depth sense, go_to, look and what_is_around")
    fresh_memory()
    rig = Rig()

    # --- the depth sense: class-agnostic, enforced in path_blocked
    n = vision.update_freespace([{"name": "center", "closest_m": 0.3}, {"name": "nope", "closest_m": 0.1}, {"bad": 1}], "metric")
    check("update_freespace keeps the good strips and drops the bad", n == 1, str(n))
    vision.FREESPACE["time"] = 0.0
    check("a stale depth summary is no sense: path_blocked is None", vision.path_blocked(0.4, 0.0) is None)
    vision.update_freespace([{"name": "center", "closest_m": 0.3}], "metric", frame_age_s=vision.FREESPACE_FRESH_S + 1)
    check("a post from an old frame counts as that old (a slow poster cannot make it look fresh)", vision.path_blocked(0.4, 0.0) is None)
    depth(center=0.3)
    res = await call(rig, "move", direction="forward", duration=1.0)
    check("something 0.3 m ahead refuses a forward move, without naming it", not res.ok and "blocked" in res.message
          and "in front" in res.message and "depth sense" in res.message, res.message)
    check("... and the reply says what to do instead", "Turn away" in res.message, res.message)
    res = await call(rig, "turn", direction="left", angle=90)
    check("turning in place is fine", res.ok, res.message)
    res = await call(rig, "move", direction="backward", duration=1.0)
    check("backing away is fine", res.ok, res.message)
    depth(center=0.8)
    res = await call(rig, "move", direction="forward", duration=1.0)
    check("something 0.8 m ahead does not block", res.ok, res.message)
    depth(center=0.44)
    res = await call(rig, "move", direction="forward", duration=1.0)
    check("... 0.44 m does (the stop distance has margin: the ranges are rough)", not res.ok and "blocked" in res.message, res.message)
    depth(left=0.2)
    res = await call(rig, "move", direction="left", duration=1.0)
    check("something close on the left refuses a sidestep left", not res.ok and "left" in res.message, res.message)
    res = await call(rig, "move", direction="right", duration=1.0)
    check("... but not a sidestep right", res.ok, res.message)
    res = await call(rig, "move", direction="forward", duration=1.0)
    check("... nor walking forward", res.ok, res.message)
    depth(center_left=("near", 0.6))
    res = await call(rig, "move", direction="forward", duration=1.0)
    check("relative mode: a near surface filling the lower band ahead-left blocks forward", not res.ok
          and "ahead on my left" in res.message, res.message)
    depth(center_left=("near", 0.2))
    res = await call(rig, "move", direction="forward", duration=1.0)
    check("... a low near_frac (floor only) does not", res.ok, res.message)
    depth(center=0.2)
    see(("chair", box(0.5, 0.4, 0.6)))
    res = await call(rig, "move", direction="forward", duration=1.0)
    check("when the detector can name it, the reply names it (chair) rather than 'something'", not res.ok
          and "chair" in res.message and "something low" not in res.message, res.message)
    res = vision.what_i_see()
    check("what_i_see adds the depth sense", "depth sense" in res.message and "center" in res.message, res.message)
    vision.VISION.update(time=0.0, detections=[])
    depth()
    res = vision.what_i_see()
    check("... 'clear ahead' when nothing is close", "clear" in res.message, res.message)

    # something low appears mid-walk: the depth sense stops the walk
    publish, sent = rig.robot.publish, [0]

    def publish_and_appear(topic, msg):
        publish(topic, msg)
        if topic == "/llm_cmd_vel" and abs(msg.linear.x) > 0:
            sent[0] += 1
            if sent[0] == 20:
                depth(center=0.25)

    rig.robot.publish = publish_and_appear
    before, start = alerts.EVENTS.last_id, len(rig.records)
    res = await call(rig, "move", direction="forward", duration=3.0)
    rig.robot.publish = publish
    check("a 3 s walk is accepted when the floor is clear", res.ok, res.message)
    check("... and stops after 1 s when the depth sense sees something", any("stopped after 1.0 s of 3.0" in o
          and "depth sense" in o for o in rig.outcomes(start)), str(rig.outcomes(start)))
    check("... and Gemini is told", any(e["kind"] == "obstacle" for e in alerts.EVENTS.since(before)))
    depth()

    # --- go_to: the brain names the goal, the robot closes the loop
    fresh_memory()
    depth()
    res = await call(rig, "go_to", what="")
    check("go_to with nothing is refused with an example", not res.ok and "e.g." in res.message, res.message)

    world = World(rig, "red mug", bearing_deg=-25.0, distance_m=2.0)
    before, start, t0 = alerts.EVENTS.last_id, len(rig.records), rig.clock.now
    res = await call(rig, "go_to", what="the red mug")
    world.close()
    check("go_to a mug 25 degrees to the left: accepted, says what it sees and will do", res.ok and "to the left" in res.message
          and "queued" in res.message and "arm's length" in res.message, res.message)
    outcomes = rig.outcomes(start)
    check("... the job reaches it", any("reached the red mug" in o for o in outcomes), str(outcomes))
    check("... turning left first, before walking", world.first_walk_bearing is not None and abs(world.first_walk_bearing) <= eyes.WALK_WITHIN_DEG
          and world.turn_ticks > 0, f"first walk at bearing {world.first_walk_bearing}, turn ticks {world.turn_ticks}")
    check("... ends at arm's length (under 0.65 m), facing it (under 10 degrees)", world.distance < 0.65 and abs(world.bearing) < 10,
          f"distance {world.distance:.2f} m, bearing {world.bearing:.1f} deg")
    check("... says how far it is", any("reached the red mug" in o and " m" in o for o in outcomes), str(outcomes))
    check("... in well under the 20 s limit", rig.clock.now - t0 < 15, f"{rig.clock.now - t0:.1f} s")
    twists = tt.walk_twists(rig)
    check("... never faster than the walk and turn speeds", all(m.linear.x <= 0.4 + 1e-9 and abs(math.degrees(m.angular.z)) <= 60 + 1e-6 for m in twists))
    evs = alerts.EVENTS.since(before)
    check("... Gemini hears the result", any(e["kind"] == "goto" and "reached" in e["for_gemini"] for e in evs), str(evs))
    check("... the screen shows it in green", alerts.display_state() is not None and alerts.display_state()["text"] == "RED MUG")
    check("... it is remembered", memory.where_is("red mug") is not None, str(memory.where_is("red mug")))
    check("... and the eyes service is no longer asked for it", eyes.WANTED["name"] == "")
    check("... ending in locomotion", rig.robot.controllers[LOCOMOTION_CONTROLLER] == "active")

    world = World(rig, "ball", bearing_deg=0.0, distance_m=1.0)
    start = len(rig.records)
    await call(rig, "go_to", what="ball")
    world.close()
    check("a ball straight ahead: reached without turning", any("reached the ball" in o for o in rig.outcomes(start)) and world.turn_ticks == 0,
          f"turn ticks {world.turn_ticks}; {rig.outcomes(start)}")

    # The depth sense's range for the thing's strip beats the box's bottom edge: here the box says
    # 1 m but the depth sense sees the thing's base 0.4 m away in the centre strip, so it has arrived.
    world = World(rig, "box", bearing_deg=0.0, distance_m=0.3)
    start = len(rig.records)
    await call(rig, "go_to", what="the box")
    world.close()
    depth()
    check("a thing whose base is already at the blocking row (0.3 m) is reached without walking",
          any("reached the box" in o for o in rig.outcomes(start)) and world.walk_ticks == 0,
          f"walk ticks {world.walk_ticks}; {rig.outcomes(start)}")

    world = World(rig, "cone", bearing_deg=0.0, distance_m=1.0, see=True)
    world.see = False  # no fixes and no depth from the sim ...
    eyes.update_target("cone", box(0.5, 0.03, 0.03, 0.99), 0.8, "eyes", bottom_known=False)  # ... a model that only pointed: no real bottom
    depth(center=0.4)  # and the depth sense sees something 0.4 m ahead (a chair leg, say)
    start = len(rig.records)
    await call(rig, "go_to", what="the cone")
    world.close()
    depth()
    check("a point without a real bottom edge: no 'arrived' from a range; dead ahead and blocked at arm's length counts (stopped in front)",
          any("reached the cone: stopped right in front of it" in o for o in rig.outcomes(start))
          and not any("it is right in front of me" in o for o in rig.outcomes(start)), f"walk ticks {world.walk_ticks}; {rig.outcomes(start)}")

    # The failure seen on the robot: a pointed-at wagon 2 m away, a chair base 0.5 m away in that direction.
    world = World(rig, "wagon", bearing_deg=-25.0, distance_m=2.0, see=True)
    world.see = False
    eyes.update_target("wagon", box(0.5 - 25 / eyes.HFOV_DEG, 0.03, 0.03, 0.5), 0.8, "eyes", bottom_known=False)
    world.obstacle = {"center_left": 0.5}
    depth(center_left=0.5)
    start = len(rig.records)
    await call(rig, "go_to", what="the wagon")
    world.close()
    depth()
    check("a pointed-at thing with a chair base in its direction is not 'reached' at 0.5 m",
          not any("reached the wagon" in o and "about 0.5 m" in o for o in rig.outcomes(start)), str(rig.outcomes(start)))

    world = World(rig, "cup", bearing_deg=-30.0, distance_m=1.5, every=40)  # a fix only every 2 s, as with Gemma
    start, t0 = len(rig.records), rig.clock.now
    await call(rig, "go_to", what="the cup")
    world.close()
    depth()
    check("with a fix only every 2 s it still gets there, dead-reckoning the aim between fixes",
          any("reached the cup" in o for o in rig.outcomes(start)) and world.distance < 0.65 and abs(world.bearing) < 12,
          f"distance {world.distance:.2f} m, bearing {world.bearing:.1f} deg, {rig.clock.now - t0:.1f} s; {rig.outcomes(start)}")

    world = World(rig, "person", bearing_deg=15.0, distance_m=3.0)

    def vanish(w: World):
        if w.ticks == 20:  # after 1 s it is gone (walked out of view)
            w.see = False
            eyes.TARGET["time"] = 0.0

    world.on_tick = vanish
    start = len(rig.records)
    await call(rig, "go_to", what="the person")
    world.close()
    check("losing sight of it stops the walk and says so", any("lost sight of the person" in o for o in rig.outcomes(start)), str(rig.outcomes(start)))
    check("... within a few seconds of losing it", world.ticks <= 20 + int(eyes.LOST_S * 20) + 1, str(world.ticks))

    before, start = alerts.EVENTS.last_id, len(rig.records)
    res = await call(rig, "go_to", what="the unicorn")
    check("a thing the eyes never find: accepted with 'once my eyes find it'", res.ok and "once my eyes find it" in res.message, res.message)
    check("... the job gives up after waiting, without moving", any("can't see the unicorn" in o for o in rig.outcomes(start))
          and not tt.walk_twists(rig)[len(tt.walk_twists(rig)):], str(rig.outcomes(start)))
    check("... and says what to do instead", any("Call find" in e["for_gemini"] for e in alerts.EVENTS.since(before)))

    world = World(rig, "door", bearing_deg=0.0, distance_m=3.0)

    def cable(w: World):
        if w.ticks == 20:
            w.obstacle = {"center": 0.25}  # a cable on the floor, not the door (which is 3 m off)

    world.on_tick = cable
    start = len(rig.records)
    await call(rig, "go_to", what="the door")
    world.close()
    depth()
    check("something low in the way on the route stops it, with the depth sense's reason", any("stopped on the way to the door" in o
          and "depth sense" in o for o in rig.outcomes(start)), str(rig.outcomes(start)))

    world = World(rig, "couch", bearing_deg=0.0, distance_m=0.9)

    def arrive(w: World):
        if w.distance < 0.6:
            depth(center=0.3)  # the couch itself fills the depth sense

    world.on_tick = arrive
    start = len(rig.records)
    await call(rig, "go_to", what="the couch")
    world.close()
    depth()
    check("the target itself blocking the way counts as arriving", any("reached the couch" in o for o in rig.outcomes(start)), str(rig.outcomes(start)))

    see(("cup", box(0.7, 0.2, 0.3, 0.9)))  # the COCO detector sees it; the eyes service has nothing
    res = await call(rig, "go_to", what="my mug")
    check("a thing the COCO detector knows is used when the eyes service has no fix (mug = cup, detector)",
          res.ok and "(detector)" in res.message and "degrees" in res.message, res.message)
    vision.VISION.update(time=0.0, detections=[])

    await call(rig, "follow", enabled=True)
    world = World(rig, "chair", bearing_deg=0.0, distance_m=1.0)
    res = await call(rig, "go_to", what="the chair")
    world.close()
    check("go_to ends follow mode and says so", res.ok and "follow" in res.message.lower() and not rig.ts.following, res.message)

    # --- look: one question to Gemma through the eyes service
    vision.FREESPACE["eyes_url"] = ""
    os.environ.pop("PUPPER_EYES_URL", None)
    res = await call(rig, "look", question="what is on the floor?")
    check("look without an eyes service says so and points to the camera", not res.ok and "eyes service" in res.message, res.message)
    asked = []

    def fake_ask(url, path, payload, timeout=0):
        asked.append((url, path, payload))
        return {"answer": "A black cable and a shoe, about a metre ahead."}

    real_ask, eyes.ask_eyes = eyes.ask_eyes, fake_ask
    depth()  # the eyes service posts its url with the free space
    res = await call(rig, "look", question="what is on the floor?")
    check("look asks the eyes service at the url it posted, and returns Gemma's answer", res.ok and "cable" in res.message
          and asked and asked[0][0] == "http://eyes.test:8100" and asked[0][1] == "/look", f"{res.message} {asked}")
    res = await call(rig, "look", question="  ")
    check("an empty question is refused with an example", not res.ok and "e.g." in res.message, res.message)

    def broken_ask(url, path, payload, timeout=0):
        raise TimeoutError("no answer")

    eyes.ask_eyes = broken_ask
    res = await call(rig, "look", question="who is there?")
    check("a dead eyes service: a clear failure, not a crash", not res.ok and "didn't answer" in res.message, res.message)
    eyes.ask_eyes = real_ask

    # --- what_is_around: the scene memory
    res = await call(rig, "what_is_around")
    check("no scene yet: says so", res.ok and "no fresh scene" in res.message, res.message)
    vision.update_scene("A living room. Couch on the right, two chairs with thin legs ahead-left, a doorway behind.")
    res = await call(rig, "what_is_around")
    check("with a scene: Gemma's words, then the senses", res.ok and "thin legs" in res.message and "depth sense" in res.message, res.message)
    rig.close()


# ----------------------------------------------------------------------------- roaming
class Room:
    """A kinematic room for explore: four walls, things standing in it, and the senses computed
    from the pose every tick (the depth strips by ray casting, the detector when a thing is in
    view). Every speed Pupper sends moves it; the room records how close it came to a wall."""

    EDGES = [-80.0, -45.0, -15.0, 15.0, 45.0, 80.0]  # the strips' bearings, as the eyes service cuts them
    SEE_M = 2.5  # the detector sees a thing within this range and 80 degrees of straight on
    THING_R = 0.05  # things are this big around (a chair's legs: what the depth sense hits is what the box's bottom is)

    def __init__(self, rig: Rig, w: float, h: float, x: float, y: float, heading: float, things: dict, coco: bool = True):
        self.rig, self.w, self.h, self.x, self.y, self.heading = rig, w, h, x, y, heading
        self.things, self.coco = things, coco  # name -> (x, y); coco: the detector knows them (else only Gemma)
        self.min_wall = min(x, w - x, y, h - y)
        self.cells: set = set()
        self.ticks = self.walk_ticks = 0
        self._publish = rig.robot.publish
        rig.robot.publish = self.publish
        self.sense()

    def ray(self, bearing: float) -> float:
        """Distance along a ray at ``bearing`` (+ right) to the nearest wall or thing."""
        a = math.radians(self.heading - bearing)
        dx, dy = math.cos(a), math.sin(a)
        ts = []
        for t in ((-self.x / dx) if dx else None, ((self.w - self.x) / dx) if dx else None,
                  (-self.y / dy) if dy else None, ((self.h - self.y) / dy) if dy else None):
            if t is not None and t > 0:
                ts.append(t)
        for tx, ty in self.things.values():
            fx, fy = self.x - tx, self.y - ty
            b = 2 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - self.THING_R ** 2
            disc = b * b - 4 * c
            if disc >= 0:
                t = (-b - math.sqrt(disc)) / 2
                if t > 0:
                    ts.append(t)
        return min(ts) if ts else 9.0

    def strips(self) -> dict:
        out = {}
        for name, b0, b1 in zip(vision.STRIPS, self.EDGES[:-1], self.EDGES[1:]):
            r = min(self.ray(b0 + (b1 - b0) * k / 4) for k in range(5))
            out[name.replace("-", "_")] = ("row", round(row_of(r), 3))  # walls and things as Lab 5 rows
        return out

    def bearing_to(self, name: str) -> tuple[float, float]:
        tx, ty = self.things[name]
        d = math.hypot(tx - self.x, ty - self.y)
        b = (self.heading - math.degrees(math.atan2(ty - self.y, tx - self.x)) + 180.0) % 360.0 - 180.0
        return b, d

    def visible(self, name: str) -> bool:
        b, d = self.bearing_to(name)
        return d <= self.SEE_M and abs(b) <= 80.0

    def sense(self) -> None:
        depth(**self.strips())
        dets = []
        for name in self.things:
            if self.coco and self.visible(name):
                b, d = self.bearing_to(name)
                hgt = min(0.95, 0.2 / max(d, 0.05))
                u, v = frame_point(b, floor_lat(d))
                dets.append((name, box(u, hgt * 0.6, hgt, min(0.99, v))))
        if dets:
            see(*dets)
        else:
            vision.VISION.update(time=0.0, detections=[])

    def publish(self, topic, msg):
        self._publish(topic, msg)
        if topic != "/llm_cmd_vel":
            return
        dt = 1.0 / tt.LLM_CMD_RATE
        vx, wz = msg.linear.x, math.degrees(msg.angular.z)
        self.ticks += 1
        if vx > 1e-9:
            self.walk_ticks += 1
        self.heading += wz * dt
        self.x += vx * math.cos(math.radians(self.heading)) * dt
        self.y += vx * math.sin(math.radians(self.heading)) * dt
        self.min_wall = min(self.min_wall, self.x, self.w - self.x, self.y, self.h - self.y)
        self.cells.add((int(self.x), int(self.y)))
        self.sense()

    def close(self) -> None:
        self.rig.robot.publish = self._publish


async def explore_tests() -> None:
    print("Hackathon G: finding things (find: look, sweep, roam, walk up to it)")
    fresh_memory()
    rig = Rig()
    real_ask, eyes.ask_eyes = eyes.ask_eyes, (lambda url, path, payload, timeout=0: {"found": False, "source": "gemma"})  # no eyes service here
    res = await call(rig, "find", what="")
    check("find with nothing is refused with an example", not res.ok and "e.g." in res.message, res.message)
    res = await call(rig, "find", what="a chair", max_seconds=1000)
    check("a search over the time limit is refused with the limit", not res.ok and "300" in res.message, res.message)

    # The same search again before it ends (a model may call find twice in one turn): one search,
    # not two. A stop drops it, and then the same search can start again.
    first = await tools.call(rig.ts, "find", {"what": "the red mug", "max_seconds": 10})
    second = await tools.call(rig.ts, "find", {"what": "my red mug", "max_seconds": 10})
    pending = rig.ts.status()
    check("find again while that search is pending: 'already looking', nothing more queued",
          first.ok and second.ok and "already looking for the red mug" in second.message
          and pending["queued"] + pending["running"] <= 1, f"{second.message} {pending}")
    await tools.call(rig.ts, "stop", {})
    await rig.settle()
    third = await tools.call(rig.ts, "find", {"what": "the red mug", "max_seconds": 10})
    check("... after a stop, the same search starts again", third.ok and "already" not in third.message, third.message)
    await rig.settle()

    # In view already: found right away, and a go_to follows.
    depth()
    see(("cup", box(0.8, 0.2, 0.3, 0.9)))
    before, start = alerts.EVENTS.last_id, len(rig.records)
    res = await call(rig, "find", what="my mug")
    vision.VISION.update(time=0.0, detections=[])
    check("a mug the detector sees: accepted, says it looks first", res.ok and "I look" in res.message and "queued" in res.message, res.message)
    check("... found right away, to the right, no turning", any("found the mug right away" in o and "+" in o for o in rig.outcomes(start))
          and not turns(rig)[len(turns(rig)):], str(rig.outcomes(start)))
    check("... the screen shows it in green and it is remembered", alerts.display_state() is not None and alerts.display_state()["text"] == "MUG"
          and memory.where_is("mug") is not None, str(alerts.display_state()))

    # Out of view: the sweep in place first; the ball comes into view during its second step.
    fresh_memory()
    depth()
    spins_seen = [0]
    publish = rig.robot.publish

    def publish_and_find(topic, msg):
        publish(topic, msg)
        if topic == "/llm_cmd_vel" and abs(msg.angular.z) > 0:
            spins_seen[0] += 1
            if spins_seen[0] == 40:  # during the second step: the ball comes into view
                see(("sports ball", box(0.5, 0.2, 0.25, 0.95)))

    rig.robot.publish = publish_and_find
    before, start, n_turn = alerts.EVENTS.last_id, len(rig.records), len(turns(rig))
    res = await call(rig, "find", what="the ball")
    rig.robot.publish = publish
    vision.VISION.update(time=0.0, detections=[])
    check("a ball out of view: the sweep finds it after two 60-degree steps, without roaming",
          any("found the ball after turning 120 degrees" in o for o in rig.outcomes(start)), str(rig.outcomes(start)))
    check("... Gemini hears where", any(e["kind"] == "found" and "found the ball" in e["for_gemini"] for e in alerts.EVENTS.since(before)))

    # A thing the detector has no class for: Gemma, through the eyes service, at each stop of the sweep.
    fresh_memory()
    depth()
    asked = []

    def exists_at_third_look(url, path, payload, timeout=0):
        asked.append(payload["name"])
        return {"found": len(asked) >= 3, "source": "gemma"}

    eyes.ask_eyes = exists_at_third_look
    start, n_turn = len(rig.records), len(turns(rig))
    res = await call(rig, "find", what="the white container")
    eyes.ask_eyes = lambda url, path, payload, timeout=0: {"found": False, "source": "gemma"}
    check("a 'white container' (no detector class): the reply says Gemma looks for it", res.ok and "doesn't know 'white container'" in res.message
          and "Gemma" in res.message, res.message)
    check("... and it is found at the third look (after 120 degrees), by Gemma", any("found the white container after turning 120 degrees" in o
          and "gemma" in o for o in rig.outcomes(start)) and asked[:3] == ["white container"] * 3, f"{rig.outcomes(start)} asked {asked}")
    depth()
    room = Room(rig, 6.0, 5.0, 4.5, 2.5, 0.0, {"chair": (1.0, 2.5)})  # the chair is 3.5 m behind the robot
    before, start, t0 = alerts.EVENTS.last_id, len(rig.records), rig.clock.now
    res = await call(rig, "find", what="a chair", max_seconds=120)
    room.close()
    check("find a chair behind the robot: accepted, says what it will do", res.ok and "Finding" in res.message
          and "queued" in res.message and "walk up to it" in res.message, res.message)
    outcomes = rig.outcomes(start)
    check("... finds it (3.5 m is beyond the sim's 2.5 m sight, so the sweep misses it and the roam finds it)",
          any("found the chair after" in o and "legs" in o for o in outcomes), str(outcomes))
    check("... in under 120 s", rig.clock.now - t0 < 120, f"{rig.clock.now - t0:.0f} s; {outcomes}")
    check("... never nearer a wall than 0.2 m (it stops with margin)", room.min_wall >= 0.2, f"nearest {room.min_wall:.2f} m")
    check("... never faster than walk and turn speeds", all(m.linear.x <= 0.4 + 1e-9 and abs(math.degrees(m.angular.z)) <= 60 + 1e-6 for m in tt.walk_twists(rig)))
    evs = alerts.EVENTS.since(before)
    check("... Gemini hears it was found, and that Pupper is walking up to it", any(e["kind"] == "found" and "found the chair" in e["for_gemini"]
          and "walking up to it" in e["for_gemini"] for e in evs), str([e["for_gemini"][:80] for e in evs]))
    check("... then a go_to to it ran and reached it", any("reached the chair" in o for o in outcomes), str(outcomes))
    check("... and the chair is remembered (by the roam, then by the walk to it)", memory.where_is("chair") is not None, str(memory.where_is("chair")))
    check("... ending in locomotion with nothing wanted", rig.robot.controllers[LOCOMOTION_CONTROLLER] == "active"
          and explore.EXPLORING["name"] == "" and eyes.WANTED["name"] == "")

    # A thing without a class: the eyes service's /exists (Gemma) at each stop.
    fresh_memory()
    depth()
    room = Room(rig, 6.0, 5.0, 1.0, 1.0, 90.0, {"whiteboard with code": (5.0, 4.0)}, coco=False)
    asked = []

    def fake_exists(url, path, payload, timeout=0):
        asked.append(path)
        return {"found": room.visible(payload["name"]), "source": "gemma"}

    eyes.ask_eyes = fake_exists
    start = len(rig.records)
    res = await call(rig, "find", what="the whiteboard with code", max_seconds=200)
    eyes.ask_eyes = lambda url, path, payload, timeout=0: {"found": False, "source": "gemma"}
    room.close()
    outcomes = rig.outcomes(start)
    check("a thing only Gemma can recognise, across the room: found by asking the eyes at each stop",
          any("found the whiteboard with code" in o and "gemma" in o for o in outcomes) and asked and all(p == "/exists" for p in asked),
          f"{outcomes} asked {len(asked)}")
    check("... never nearer a wall than 0.2 m", room.min_wall >= 0.2, f"nearest {room.min_wall:.2f} m")

    # Nothing to find: it keeps moving to new places until time is up, and says so.
    fresh_memory()
    depth()
    room = Room(rig, 6.0, 5.0, 3.0, 2.5, 0.0, {})
    before, start = alerts.EVENTS.last_id, len(rig.records)
    await call(rig, "find", what="a unicorn", max_seconds=60)
    room.close()
    outcomes = rig.outcomes(start)
    check("nothing to find: gives up when the time is up, saying how much it covered", any("no unicorn found in 60 s" in o
          and "square metres" in o for o in outcomes), str(outcomes))
    check("... having moved through several cells, not bounced in place", len(room.cells) >= 4, f"{len(room.cells)} cells")
    check("... never nearer a wall than 0.2 m", room.min_wall >= 0.2, f"nearest {room.min_wall:.2f} m")
    check("... and Gemini hears that it ended", any(e["kind"] == "search" and "Roaming ended" in e["for_gemini"] for e in alerts.EVENTS.since(before)))

    # Boxed in: every direction blocked from the start.
    fresh_memory()
    room = Room(rig, 0.7, 0.7, 0.35, 0.35, 0.0, {})
    start = len(rig.records)
    await call(rig, "find", what="a ball", max_seconds=60)
    room.close()
    outcomes = rig.outcomes(start)
    check("boxed in: gives up after a few legs that could not start, without walking", any("boxed in" in o for o in outcomes)
          and room.walk_ticks == 0, f"walk ticks {room.walk_ticks}; {outcomes}")

    # No depth sense at all: accepted with a warning, and it only turns.
    fresh_memory()
    vision.FREESPACE.update(time=0.0, strips=[])
    vision.POSTERS.clear()
    res = await call(rig, "find", what="a chair", max_seconds=5)
    check("with no eyes service at all the reply warns it will only turn", res.ok and "eyes service is not running" in res.message, res.message)
    vision.update_freespace([{"name": "center", "closest_m": 2.0}], "metric", source="pi")
    vision.FREESPACE["time"] = 0.0  # idle: the last post is older than the trust window, but the eyes are alive
    res = await call(rig, "find", what="a chair", max_seconds=5)
    check("... but an idle eyes service (stale data, recent post) is not a fault", res.ok and "not running" not in res.message, res.message)

    depth()
    await call(rig, "follow", enabled=True)
    room = Room(rig, 6.0, 5.0, 3.0, 2.5, 0.0, {})
    res = await call(rig, "find", what="a chair", max_seconds=5)
    room.close()
    check("find ends follow mode and says so", res.ok and "follow" in res.message.lower() and not rig.ts.following, res.message)
    rig.imu(45.0)
    rig.robot.send("/rosout", tt._Msg(name="gemini_walk", msg="Emergency stop triggered"))
    await rig.settle()
    res = await call(rig, "find", what="a chair")
    check("fallen: no search", not res.ok and "fell" in res.message, res.message)
    eyes.ask_eyes = real_ask
    rig.close()


# ----------------------------------------------------------------------------- follow guard and the session log
async def guard_tests() -> None:
    print("Hackathon H: the follow guard and the session log")
    fresh_memory()
    rig = Rig()
    hlog.LOG_DIR = _TMP / "logs"
    vision._unpause()
    depth()
    res = await call(rig, "follow", enabled=True)
    check("follow mode on", res.ok and rig.ts.following, res.message)
    check("the guard leaves a clear way alone", await vision.follow_guard_tick(rig.ts) is None and rig.ts.following)
    before = alerts.EVENTS.last_id
    depth(center=0.3)
    did = await vision.follow_guard_tick(rig.ts)
    check("something in the way pauses following (the follower is switched off)", did == "paused" and not rig.ts.following, str(did))
    evs = alerts.EVENTS.since(before)
    check("... and Gemini is told, with what was in the way", len(evs) == 1 and evs[0]["kind"] == "obstacle"
          and "paused following" in evs[0]["for_gemini"] and "something low" in evs[0]["for_gemini"], str(evs))
    check("... still blocked: stays paused", await vision.follow_guard_tick(rig.ts) is None and not rig.ts.following)
    depth()
    before = alerts.EVENTS.last_id
    did = await vision.follow_guard_tick(rig.ts)
    check("clear: not yet (it waits a moment)", did is None and not rig.ts.following, str(did))
    rig.clock.now += vision.CLEAR_S + 0.1
    did = await vision.follow_guard_tick(rig.ts)
    check("clear for a second: following again, and Gemini hears it", did == "resumed" and rig.ts.following
          and any(e["kind"] == "follow" and "following again" in e["text"] for e in alerts.EVENTS.since(before)), str(did))

    depth(center=0.3)
    await vision.follow_guard_tick(rig.ts)
    check("paused again", not rig.ts.following and vision.FOLLOW_GUARD["paused_since"] > 0)
    rig.clock.now += vision.PAUSE_MAX_S + 1
    before = alerts.EVENTS.last_id
    did = await vision.follow_guard_tick(rig.ts)
    check("blocked too long: gives up and says so", did == "gave up" and not rig.ts.following and vision.FOLLOW_GUARD["paused_since"] == 0
          and any("stopped following" in e["text"] for e in alerts.EVENTS.since(before)), str(did))

    depth()
    await call(rig, "follow", enabled=True)
    depth(center=0.3)
    await vision.follow_guard_tick(rig.ts)
    res = await call(rig, "follow", enabled=False)
    depth()
    rig.clock.now += vision.CLEAR_S + 0.1
    did = await vision.follow_guard_tick(rig.ts)
    check("follow(false) during a pause means no resume", did is None and not rig.ts.following, str(did))

    await call(rig, "follow", enabled=True)
    depth(center=0.3)
    await vision.follow_guard_tick(rig.ts)
    depth()
    res = await call(rig, "turn", direction="left", angle=90)
    rig.clock.now += vision.CLEAR_S + 0.1
    did = await vision.follow_guard_tick(rig.ts)
    check("another motion during a pause means no resume either", did is None and not rig.ts.following, str(did))

    # attention: the eyes run at full rate only while it matters
    vision.ATTENTION["until"] = 0.0
    vision.POSTERS.clear()
    check("idle robot, no recent tool call: no attention", not vision.attention_now(rig.ts))
    await call(rig, "robot_status")
    check("any tool call wakes the eyes for a while", vision.attention_now(rig.ts) and vision.attention_now())
    vision.ATTENTION["until"] = 0.0
    await call(rig, "follow", enabled=True)
    vision.ATTENTION["until"] = 0.0
    check("following keeps them awake even with no recent call", vision.attention_now(rig.ts))
    await call(rig, "follow", enabled=False)
    vision.ATTENTION["until"] = 0.0
    check("... and so does a running or queued motion", not vision.attention_now(rig.ts))
    vision.update_freespace([{"name": "center", "closest_m": 2.0}], "metric", source="pi")
    check("one poster: nobody else is posting", not vision.others_fresh("pi") and vision.others_fresh("laptop"))
    vision.POSTERS["pi"] -= vision.OTHER_FRESH_S + 1
    check("... an old post no longer counts", not vision.others_fresh("laptop"))
    vision.update_freespace([{"name": "center", "closest_m": 2.0}], "metric", source="laptop")
    check("a laptop posting means the Pi's eyes should stand down", vision.others_fresh("pi") and not vision.others_fresh("laptop"))
    vision.POSTERS.clear()

    # the session log
    e = hlog.record("tool", name="move", args={"direction": "forward"}, ok=False, message="blocked: something low")
    hlog.record("speech", who="person", text="go to the chair")
    path = hlog.save_frame(b"\xff\xd8not really a jpeg", "move")
    entries = hlog.tail(5)
    check("record writes json lines with a clock", entries and entries[-2]["kind"] == "tool" and entries[-2]["name"] == "move"
          and "clock" in entries[-2] and entries[-1]["text"] == "go to the chair", str(entries))
    check("save_frame writes the frame next to the transcript", path is not None and Path(path).exists() and "move" in path, str(path))
    check("an event is recorded too", any(x["kind"] == "event" and x["event"] == "follow" for x in hlog.tail(50)))
    check("describe makes a readable line", "tool move" in hlog.describe(e) and "FAILED" in hlog.describe(e), hlog.describe(e))
    rig.close()


# ----------------------------------------------------------------------------- learning
async def learn_tests() -> None:
    print("Hackathon E: learning routines on the fly")
    fresh_memory()
    rig = Rig()
    res = await call(rig, "do_routine", name="happy dance")
    check("an unknown routine is refused, and the known ones are listed", not res.ok and "unknown routine" in res.message, res.message)
    res = await call(rig, "learn_routine", name="Happy Dance!")
    check("learn_routine starts recording", res.ok and "learning 'happy_dance'" in res.message, res.message)
    res = await call(rig, "learn_routine", name="other")
    check("... a second one at the same time is refused", not res.ok and "already learning" in res.message, res.message)
    await call(rig, "play_trick", name="bow")
    await call(rig, "turn", direction="left", angle=180)
    await call(rig, "move", direction="forward", speed=9.0)  # refused: too fast
    await call(rig, "recall")  # not a motion
    await call(rig, "show_text", text="HI")
    check("accepted motions are recorded, refused ones and non-motions are not",
          [s["tool"] for s in learn.LEARNING["steps"]] == ["play_trick", "turn"], str(learn.LEARNING))
    res = await call(rig, "finish_learning")
    check("finish_learning saves it and says the steps", res.ok and "learned 'happy_dance'" in res.message and "then" in res.message, res.message)
    check("... in memory", "happy_dance" in memory.load()["routines"])
    check("... and the prompt section names it", "happy_dance" in memory.memory_prompt())
    n_switch = len(rig.robot.switches)
    res = await call(rig, "do_routine", name="happy dance")
    check("do_routine replays both steps through their tools", res.ok and "2 steps queued" in res.message, res.message)
    check("... the trick's controller took the legs again", any("trick_bow" in on for _, _, on in rig.robot.switches[n_switch:]))
    check("... and it turned 180 degrees left", len([m for m in turns(rig) if m.angular.z > 0]) >= 3 * 20)
    res = await call(rig, "learn_routine", name="nothing")
    res = await call(rig, "finish_learning")
    check("finishing with no steps is refused", not res.ok and "nothing recorded" in res.message, res.message)
    await call(rig, "learn_routine", name="scrap")
    await call(rig, "turn", direction="right")
    res = await call(rig, "finish_learning", save=False)
    check("finish_learning(save=false) throws it away", res.ok and "threw away" in res.message and "scrap" not in memory.load()["routines"], res.message)
    res = await call(rig, "list_routines")
    check("list_routines names it with its steps", res.ok and "happy_dance" in res.message and "play_trick" in res.message, res.message)
    see(("person", box(0.5, 0.6, 0.8)))  # someone right in front: the routine's walk would be refused
    await call(rig, "learn_routine", name="walkabout")
    vision.VISION.update(time=0.0, detections=[])
    await call(rig, "move", direction="forward", duration=1.0)
    await call(rig, "play_trick", name="bow")
    await call(rig, "finish_learning")
    see(("person", box(0.5, 0.6, 0.8)))
    res = await call(rig, "do_routine", name="walkabout")
    check("a replayed step that is refused now (someone in the way) stops the routine and says where",
          not res.ok and "stopped at step 1" in res.message and "blocked" in res.message, res.message)
    vision.VISION.update(time=0.0, detections=[])
    res = await call(rig, "forget_routine", name="walkabout")
    check("forget_routine", res.ok and "walkabout" not in memory.load()["routines"], res.message)
    check("the recorder is a registry hook", learn.record in tools.CALL_HOOKS)
    rig.close()


# ----------------------------------------------------------------------------- what Gemini reads
async def declaration_tests() -> None:
    print("Hackathon: what Gemini reads")
    rig = Rig()
    new_tools = ["remember", "recall", "forget", "switch_profile", "show_text", "sound_alerts", "set_reminder",
                 "list_reminders", "cancel_reminder", "found_object",
                 "learn_routine", "finish_learning", "do_routine", "list_routines", "forget_routine",
                 "go_to", "look", "what_is_around", "find"]
    tt.described(rig, new_tools, "hackathon")
    decls = {d["name"]: d for d in tools.declarations(rig.ts)}
    check("the hackathon tools come after the lab's", list(decls).index("remember") > list(decls).index("interactive_action"))
    for heading in ("Memory", "Reminders", "Hearing and the screen", "Finding things and not bumping into them", "Learning new tricks",
                    "Seeing with your own eyes (depth and Gemma)"):
        text = tt.prompt_sections(heading)
        check(f"prompt.md has a '{heading}' section with 30+ words", len(text.split()) >= 30, f"{len(text.split())} words")
    check("the prompt explains bracketed messages come from the robot, not the person",
          "[reminder]" in tt.system_prompt().lower() and "not from the person" in tt.system_prompt())
    rig.close()


GROUPS = {
    "memory": memory_tests, "screen": screen_tests, "sound": sound_tests,
    "vision": vision_tests, "eyes": eyes_tests, "explore": explore_tests, "guard": guard_tests, "learn": learn_tests,
    "declarations": declaration_tests,
}


def main(groups: list[str] | None = None) -> None:
    for name in groups or GROUPS:
        if name not in GROUPS:
            sys.exit(f"unknown hackathon test group {name!r}; choose from {list(GROUPS)}")
        asyncio.run(GROUPS[name]())
        print()
    if tt.FAIL:
        print(f"{tt.FAIL} check(s) failed, {tt.PASS} passed.")
        raise SystemExit(1)
    print(f"All {tt.PASS} hackathon checks passed.")


if __name__ == "__main__":
    main(sys.argv[1:])
