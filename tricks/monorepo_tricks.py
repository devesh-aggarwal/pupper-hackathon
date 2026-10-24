"""The Pupper monorepo's 13 recorded tricks, as tracking-policy motions.

The monorepo plays its tricks open loop (animation_controller_py): it blends from
the current pose to the first row over init_duration, then plays one CSV row per
frame at frame_rate, ignoring the recording's own timestamps. To turn a recording
into a trick policy we need the whole-body motion, not just joint angles, so this
replays each recording with physics the same way the robot does -- from the
default stand, the player's blend-in, the player's frame rate and gains -- and
keeps what the simulated robot did (root pose and joint angles) as the
reference, resampled to the tracking task's 50 Hz. The policy trick therefore
plays at the same speed as the recording did on the robot.

Run with pupper-mjlab's environment:

    cd ~/projects/cs123/pupper-mjlab
    uv run python ~/projects/cs123/gemini_pupper_solutions/tricks/monorepo_tricks.py \\
        --animations ~/projects/pupperv3-monorepo/ros2_ws/src/animation_controller_py/launch/animations

It writes tricks/<name>/motion.npz for every trick (train each with
Mjlab-Tracking-Flat-Pupper-v3) and prints how far the replay tilted.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from pathlib import Path

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  DEFAULT_POSE,
  JOINT_NAMES,
  PUPPER_MUJOCO_OPTIONS,
  get_pupper_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.tasks.tracking.config.pupper import motion_design as md
from mjlab.tasks.tracking.config.pupper.env_cfgs import get_spec_with_hulls
from mjlab.utils.spec_config import CollisionCfg

HERE = Path(__file__).resolve().parent

# Trick name (Gemini's id and the folder under tricks/) -> monorepo recording stem.
RECORDINGS = {
  "twerk": "twerk_recording_2025-09-04_16-14-51_0",
  "lie_sit_lie": "lie_sit_lie_recording_2025-09-03_12-44-08_0",
  "stand_sit_shake_sit_stand": "stand_sit_shake_sit_stand_recording_2025-09-03_12-47-18_0",
  "stand_sit_stand": "stand_sit_stand_recording_2025-09-03_12-46-36_0",
  "upward_dog": "upward_dog_recording_2025-10-22_17-17-07",
  "lie_downward_dog": "lie_downward_dog_recording_2025-09-04_16-08-00_0",
  "stand_downward_dog": "stand_downward_dog_recording_2025-09-04_16-09-51_0",
  "superman": "superman_recording_2025-10-22_17-47-41",
  "pee": "pee2_recording_2025-10-22_17-41-45",
  "push_up": "push_up_recording_2025-09-04_16-11-34_0",
  "sneeze": "sneeze_recording_2025-09-04_16-13-54_0",
  "spider": "spider_recording_2025-09-04_16-12-38_0",
  "swim": "swim_recording_2025-09-04_16-10-45_0",
}

# The robot's player as Gemini Pupper runs it (robot/launch/gemini_pupper.yaml).
PLAYER_RATE = 40.0  # rows per second
PLAYER_BLEND_S = 1.0  # init_duration: current pose -> first row
PLAYER_KP, PLAYER_KD = 7.5, 0.25
SETTLE_S = 0.5  # stand still in sim first, so the motion starts from a settled stand
HOLD_S = 1.0  # keep the last frame a little, so the policy learns to hold the end pose
FPS = 50.0
# The torso and leg hulls touch the ground (the tricks sit and lie down). The robot's
# collision config disables every geom it doesn't name, so enable the hulls again.
HULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_hull$",),
  contype=0,
  conaffinity=1,
  condim=3,
  friction=(0.8, 0.02, 0.01),
  disable_other_geoms=False,
)


def read_recording(path: Path) -> np.ndarray:
  """Rows of joint angles in JOINT_NAMES order (the CSV's columns are in any order)."""
  with open(path) as f:
    rows = list(csv.DictReader(f))
  return np.array([[float(r[j]) for j in JOINT_NAMES] for r in rows])


def player_targets(rows: np.ndarray, t: float) -> np.ndarray:
  """Joint targets the robot's player sends at time t after the trick starts."""
  default = np.array(DEFAULT_POSE)
  if t < PLAYER_BLEND_S:
    a = t / PLAYER_BLEND_S
    return default * (1 - a) + rows[0] * a
  x = min((t - PLAYER_BLEND_S) * PLAYER_RATE, len(rows) - 1)
  i = min(int(x), len(rows) - 2)
  a = x - i
  return rows[i] * (1 - a) + rows[i + 1] * a


def build_model() -> tuple[mujoco.MjModel, np.ndarray, np.ndarray]:
  """The tracking task's Pupper (hull collisions) on a floor, with the player's gains."""
  cfg = get_pupper_robot_cfg()
  cfg.spec_fn = get_spec_with_hulls
  cfg.collisions = (*cfg.collisions, HULL_COLLISION)
  cfg.articulation = dataclasses.replace(
    cfg.articulation,
    actuators=tuple(
      dataclasses.replace(a, stiffness=PLAYER_KP, damping=PLAYER_KD) for a in cfg.articulation.actuators
    ),
  )
  spec = Entity(cfg).spec
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(5, 5, 0.1), friction=(0.8, 0.02, 0.01),
    contype=1, conaffinity=1,
  )
  o = PUPPER_MUJOCO_OPTIONS
  spec.option.timestep = o["timestep"]
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
  spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  spec.option.impratio = o["impratio"]
  spec.option.iterations = o["iterations"]
  spec.option.ls_iterations = o["ls_iterations"]
  spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
  m = spec.compile()
  qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
  act = np.array([
    next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n))
    for n in JOINT_NAMES
  ])
  return m, qadr, act


def replay(rows: np.ndarray, m, qadr, act) -> md.Motion:
  """Play the recording like the robot does and record the robot at 50 Hz."""
  d = mujoco.MjData(m)
  d.qpos[:3] = (0, 0, md.STAND_HEIGHT + 0.005)
  d.qpos[3:7] = (1, 0, 0, 0)
  d.qpos[qadr] = DEFAULT_POSE
  default = np.array(DEFAULT_POSE)
  steps_per_frame = round(1 / FPS / m.opt.timestep)
  for _ in range(round(SETTLE_S / m.opt.timestep)):
    d.ctrl[act] = default
    mujoco.mj_step(m, d)
  length = PLAYER_BLEND_S + (len(rows) - 1) / PLAYER_RATE + HOLD_S
  root_pos, root_quat, joints = [], [], []
  for f in range(int(length * FPS) + 1):
    root_pos.append(d.qpos[:3].copy())
    root_quat.append(d.qpos[3:7].copy())
    joints.append(d.qpos[qadr].copy())
    for s in range(steps_per_frame):
      d.ctrl[act] = player_targets(rows, (f * steps_per_frame + s) * m.opt.timestep)
      mujoco.mj_step(m, d)
  root_pos = np.array(root_pos)
  root_pos[:, :2] -= root_pos[0, :2]  # start at the origin
  return md.Motion(fps=FPS, root_pos=root_pos, root_quat=np.array(root_quat), joint_pos=np.array(joints))


def max_tilt_deg(motion: md.Motion) -> float:
  """Largest angle between the body's up axis and the world's."""
  w, x, y, z = motion.root_quat.T
  up_z = 1 - 2 * (x * x + y * y)
  return float(np.degrees(np.arccos(np.clip(up_z, -1, 1))).max())


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--animations", type=Path, required=True, help="animation_controller_py/launch/animations")
  parser.add_argument("names", nargs="*", help="tricks to convert (default: all 13)")
  args = parser.parse_args()
  m, qadr, act = build_model()
  for name in args.names or RECORDINGS:
    rows = read_recording(args.animations / f"{RECORDINGS[name]}.csv")
    motion = replay(rows, m, qadr, act)
    out = HERE / name
    out.mkdir(exist_ok=True)
    motion.save_npz(out / "motion.npz")
    checks = md.check_motion(motion)
    print(f"{name:27} {checks['duration_s']:5.1f} s  max tilt {max_tilt_deg(motion):5.1f} deg  "
          f"final height {motion.root_pos[-1, 2]:.3f} m  lowest body {checks['lowest_body_point_m']:+.3f} m  "
          f"max joint speed {checks['max_joint_speed_rad_s']:.1f} rad/s")


if __name__ == "__main__":
  main()
