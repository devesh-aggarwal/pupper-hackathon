"""Tricks: everything under <repo>/tricks/<name>/trick.yaml.

Every trick is played by trick policies (a playback trick by one, an interactive
trick by one or more), so playing one is a controller switch into trick mode:

  type: policy   a trick policy: a tracking policy trained to follow a reference
                 motion (``policy``: the JSON from pupper-mjlab's
                 export_trick.py, which carries its own reference motion).
                 The launch file loads one inactive controller per policy
                 trick; playing the trick switches to it and back. The 13
                 tricks recorded in the Pupper monorepo are policies too
                 (tricks/monorepo_tricks.py turns a recording into a motion).
  type: reference  a joint-offset reference (``reference``: JSON with
                 playback_s, active_s, cross_fade_s and an n x 12 table) played
                 once through the locomotion policy, the way the joystick's game
                 mode plays the jump. Only references the locomotion policy was
                 trained on work (the default policy knows its jump).

trick.yaml fields: description (what Gemini reads), type, policy or reference,
start_pose / end_pose (standing, sitting or lying), and for policies an optional
max_body_angle_deg (the tilt e-stop while the trick plays; default 60).
Optional replaced_by: another trick's name. While that trick is installed, this
one is left out (no controller, not offered to Gemini): sit and
stand_sit_shake_sit_stand step aside once your sit_stay_shake is in.

Two kinds of policy trick. A playback trick plays one reference motion from
start to finish; its tracking policy keeps Pupper balanced while it plays. An
interactive trick is one Pupper stays in and that takes requests. Its policy
was trained on a clip library (pupper-mjlab's motion_design.concat_clips), and
its JSON carries the clip table. The controller
plays the first clip and follows each clip's ``next`` until it reaches a looping
clip (the dog stays) or a clip with nowhere to go (it holds the last frame).
Two more trick.yaml fields turn clips into things Gemini can ask for:

  actions:    clip name -> description. Each becomes an interactive action that plays
              that clip while the trick is running (the interactive_action tool).
  exit_clip:  a clip that brings the robot back to standing. It plays by itself
              before any other motion, and after it the locomotion policy takes over.

An interactive trick can also walk in its own stance (the spider). One more
field, a block:

  walk:
    policy: walk_policy.json   # a walking policy trained in that stance
    speed: 0.1                 # default m/s for move and arc in this trick
    max_forward: 0.15          # m/s, forward and backward
    max_sideways: 0.1          # m/s
    turn: 30                   # deg/s: turn's speed in this trick, and its max
    run: false                 # can it run?
    note: Creeps slowly.       # anything else Gemini should know

The launch loads the policy as a second controller, walk_<name>. Playing the
trick plays its opening clips, then hands the legs to the walking policy; move,
turn and arc drive it, at these speeds (robot.speeds()), and a faster move is
refused with these numbers. The speeds can't be more than the policy was trained
for (its command_clip). Gemini reads them in the trick's description. Leaving
the trick switches back to the trick's own policy, started at its exit_clip (the
controller's start_clip parameter).

This module has no ROS imports, so the launch file can use it too.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TRICK_DIR = Path(__file__).resolve().parent.parent / "tricks"
# Trick controllers hold the default pose briefly, then fade the policy in; the
# trick's own clock starts after the fade (see pupper_gait_deploy trick_motion.hpp).
TRICK_INIT_S = 0.5
TRICK_FADE_S = 1.0

# The locomotion controller Gemini drives: the course default policy with the
# running cap lifted and reference tricks enabled. The joystick's X controller
# (neural_controller_default) keeps its walk-only caps.
WALK_CONTROLLER = "gemini_walk"
WALK_SPEED_CAP = 1.5  # m/s, forward, backward and (clipped by the policy to 0.5) sideways
REFERENCE_SETTLE_S = 1.2  # after a reference trick's playback, time to land and settle
# A walking policy (gemini_walk, or a trick's walk_policy) holds its default pose for
# init_duration, then fades the policy in over fade_in_duration (2 s + 2 s).
WALK_READY_S = 4.2


@dataclass
class WalkConfig:
    """How a walking trick walks: its default and top speeds, in trick.yaml's walk block."""

    speed: float  # default m/s for move and arc
    max_forward: float  # m/s, forward and backward
    max_sideways: float  # m/s
    turn: float  # deg/s: turn's speed, and the top turning speed
    run: bool = False
    note: str = ""

    def describe(self) -> str:
        """What Gemini reads about walking in this trick."""
        run = "It can run." if self.run else "It cannot run."
        return (f"While in it, move, turn and arc walk in its stance: at most {self.max_forward:g} m/s forward or"
                f" backward and {self.max_sideways:g} m/s sideways (default {self.speed:g} m/s); it turns at"
                f" {self.turn:g} deg/s. {run} {self.note}").strip()


@dataclass
class Trick:
    name: str
    description: str
    kind: str  # "policy" or "reference"
    path: Path  # policy JSON or reference JSON
    duration: float  # seconds of motion
    start_pose: str = "standing"
    end_pose: str = "standing"
    max_body_angle: float = math.radians(60.0)
    clips: list[dict] = field(default_factory=list)  # the policy's clip table
    fps: float = 50.0
    actions: dict[str, str] = field(default_factory=dict)  # clip -> description
    exit_clip: str | None = None
    walk_path: Path | None = None  # a walking interactive trick's walking policy
    walk: WalkConfig | None = None  # ... and its speeds (trick.yaml's walk block)

    @property
    def interactive(self) -> bool:
        """An interactive trick: Pupper stays in it and takes requests (its clips)."""
        return bool(self.clips)

    @property
    def walks(self) -> bool:
        """An interactive trick that walks in its own stance (move, turn, arc)."""
        return self.walk_path is not None

    @property
    def walk_controller(self) -> str:
        """ros2_control controller name for a walking trick's walking policy."""
        return "walk_" + re.sub(r"[^a-z0-9_]", "_", self.name.lower())

    def clip(self, name: str) -> dict:
        return next(c for c in self.clips if c["name"] == name)

    def clip_length(self, name: str) -> float:
        """Seconds the controller spends in one pass of a clip (see ClipPlayer)."""
        c = self.clip(name)
        return (c["end"] - c["start"] - 1) / self.fps

    def clip_time_estimate(self, name: str) -> float:
        """About how long a requested clip takes: the looping clip that is playing
        finishes its pass first, then the clip plays."""
        loops = [c for c in self.clips if c.get("loop")]
        return self.clip_length(name) + max((self.clip_length(c["name"]) for c in loops), default=0.0)

    def intro_duration(self) -> float:
        """From the first clip, following next, until a looping clip or a hold."""
        total, name, seen = 0.0, self.clips[0]["name"], set()
        while name and name not in seen:  # "next" names the following clip
            seen.add(name)
            c = self.clip(name)
            if c.get("loop"):
                break
            total += self.clip_length(name)
            name = c.get("next")
        return total

    @property
    def controller(self) -> str:
        """ros2_control controller name for a policy trick."""
        return "trick_" + re.sub(r"[^a-z0-9_]", "_", self.name.lower())

    @property
    def play_time(self) -> float:
        """How long the trick's own policy plays, including the controller's blend-in."""
        if self.kind == "reference":
            return self.duration + REFERENCE_SETTLE_S
        if self.clips:  # an interactive trick is "played" once its opening clips are done
            return TRICK_INIT_S + TRICK_FADE_S + self.intro_duration()
        return TRICK_INIT_S + TRICK_FADE_S + self.duration

    @property
    def total_time(self) -> float:
        """How long playing the trick takes: its policy, then (a walking trick) the
        walking policy taking over."""
        return self.play_time + (WALK_READY_S if self.walks else 0.0)


def load_tricks(trick_dir: Path = TRICK_DIR) -> dict[str, Trick]:
    tricks = {}
    replaced_by: dict[str, str] = {}  # trick -> the trick that replaces it
    for spec_path in sorted(trick_dir.glob("*/trick.yaml")):
        spec = yaml.safe_load(spec_path.read_text())
        name = spec_path.parent.name
        kind = spec.get("type")
        clips, fps = [], 50.0
        if kind == "policy":
            path = spec_path.parent / spec["policy"]
            info = json.loads(path.read_text())["trick"]
            duration = float(info["duration_s"])
            clips, fps = list(info.get("clips", [])), float(info.get("fps", 50.0))
        elif kind == "reference":
            path = spec_path.parent / spec["reference"]
            duration = float(json.loads(path.read_text())["playback_s"])
        else:
            raise ValueError(f"{spec_path}: type must be policy or reference, not {kind!r}")
        actions = {str(k): str(v).strip() for k, v in (spec.get("actions") or {}).items()}
        exit_clip = spec.get("exit_clip")
        clip_names = {c["name"] for c in clips}
        for clip in [*actions, *([exit_clip] if exit_clip else [])]:
            if clip not in clip_names:
                raise ValueError(f"{spec_path}: clip {clip!r} is not in the policy's clips {sorted(clip_names)}")
        walk_path, walk = None, None
        if spec.get("walk"):
            w = spec["walk"]
            if not clips or not exit_clip:
                raise ValueError(f"{spec_path}: walk needs an interactive trick (clips and an exit_clip)")
            walk_path = spec_path.parent / w["policy"]
            walk = WalkConfig(speed=float(w["speed"]), max_forward=float(w["max_forward"]),
                              max_sideways=float(w["max_sideways"]), turn=float(w["turn"]),
                              run=bool(w.get("run", False)), note=str(w.get("note", "")).strip())
            clip_cmd = json.loads(walk_path.read_text())["command_clip"]  # what the policy was trained on
            trained = (max(map(abs, clip_cmd["vx"])), max(map(abs, clip_cmd["vy"])),
                       math.degrees(max(map(abs, clip_cmd["wz"]))))
            for what, asked, top in zip(("max_forward", "max_sideways", "turn"),
                                        (walk.max_forward, walk.max_sideways, walk.turn), trained):
                if asked > top + 1e-6:
                    raise ValueError(f"{spec_path}: walk {what} {asked:g} is more than its policy was trained for ({top:g})")
            if walk.speed > walk.max_forward:
                raise ValueError(f"{spec_path}: walk speed {walk.speed:g} is more than max_forward {walk.max_forward:g}")
        if spec.get("replaced_by"):
            replaced_by[name] = str(spec["replaced_by"])
        description = str(spec.get("description", name)).strip()
        if walk:  # what Gemini reads about walking in it comes from the numbers
            description = f"{description} {walk.describe()}"
        tricks[name] = Trick(
            name=name,
            description=description,
            kind=kind,
            path=path,
            duration=duration,
            start_pose=spec.get("start_pose", "standing"),
            end_pose=spec.get("end_pose", "standing"),
            max_body_angle=math.radians(float(spec.get("max_body_angle_deg", 60.0))),
            clips=clips,
            fps=fps,
            actions=actions,
            exit_clip=exit_clip,
            walk_path=walk_path,
            walk=walk,
        )
    return {name: t for name, t in tricks.items() if replaced_by.get(name) not in tricks}


def controller_params(trick: Trick, base: dict) -> dict:
    """Parameters for a trick's controller: the course default controller's, with the
  trick's policy, a short blend-in, its tilt limit, and no joystick game features."""
    params = dict(base)
    params.pop("chord_partner", None)
    params.update(
        model_path=str(trick.path),
        init_duration=TRICK_INIT_S,
        fade_in_duration=TRICK_FADE_S,
        max_body_angle=trick.max_body_angle,
        game_buttons=False,
        max_vx_forward=0.0,
        max_vx_backward=0.0,
    )
    return params


def walk_policy_params(trick: Trick, base: dict) -> dict:
    """Parameters for a walking trick's walking policy: the course default controller's,
  with that policy (its JSON carries its stance, gains and command ranges) and no
  joystick game features."""
    params = dict(base)
    params.pop("chord_partner", None)
    params.update(model_path=str(trick.walk_path), game_buttons=False, reference_tricks=False)
    return params


def reference_message_data(trick: Trick) -> list[float]:
    """The ~/reference_trick message: playback_s, active_s, cross_fade_s, then the table."""
    ref = json.loads(trick.path.read_text())
    data = [float(ref["playback_s"]), float(ref["active_s"]), float(ref["cross_fade_s"])]
    for row in ref["table"]:
        if len(row) != 12:
            raise ValueError(f"{trick.path}: every table row needs 12 joint offsets")
        data.extend(float(v) for v in row)
    return data


def walk_controller_params(base: dict) -> dict:
    """The gemini_walk controller: the course default controller, able to run and to
  play reference tricks, with no joystick game features."""
    params = dict(base)
    params.pop("chord_partner", None)
    params.update(
        game_buttons=False,
        speed_cap=WALK_SPEED_CAP,
        max_vx_forward=WALK_SPEED_CAP,
        max_vx_backward=WALK_SPEED_CAP,
        reference_tricks=True,
    )
    return params
