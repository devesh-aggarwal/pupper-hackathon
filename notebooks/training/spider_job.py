# Spider (CS123_Pupper_Spider.ipynb): cut/hold/reverse, the clips, and the spider-walk reward
# weights, as one script. Run inside a logged-in Colab runtime with pupper-mjlab installed:
#   python spider_job.py clips   # build the clip library, replay it, upload, train the clip policy
#   python spider_job.py walk    # train the spider walking policy
import os, subprocess, sys
sys.path.insert(0, "/content/pupper-mjlab/src")
os.environ["MUJOCO_GL"] = "egl"
os.chdir("/content/pupper-mjlab")
mode = sys.argv[1]
from pathlib import Path
import numpy as np
from mjlab.tasks.tracking.config.pupper import motion_design as md
from mjlab.tasks.tracking.config.pupper.tricks import spider
import mjlab.envs
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
import mjlab.tasks
import wandb
OUT = Path("/content/tricks"); OUT.mkdir(exist_ok=True)

def cut(motion: md.Motion, start_s: float, end_s: float) -> md.Motion:
  # Seconds -> frame numbers; +1 keeps the end frame, so neighbours share it.
  i0 = round(start_s * motion.fps)
  i1 = round(end_s * motion.fps)
  return md.Motion(
    fps=motion.fps,
    root_pos=motion.root_pos[i0 : i1 + 1].copy(),
    root_quat=motion.root_quat[i0 : i1 + 1].copy(),
    joint_pos=motion.joint_pos[i0 : i1 + 1].copy(),
  )


def hold(motion: md.Motion, at_s: float, seconds: float) -> md.Motion:
  # Repeat one frame: seconds * fps + 1 frames, so it starts and ends on that pose.
  i = round(at_s * motion.fps)
  n = round(seconds * motion.fps) + 1
  return md.Motion(
    fps=motion.fps,
    root_pos=np.repeat(motion.root_pos[i : i + 1], n, axis=0),
    root_quat=np.repeat(motion.root_quat[i : i + 1], n, axis=0),
    joint_pos=np.repeat(motion.joint_pos[i : i + 1], n, axis=0),
  )


def reverse(motion: md.Motion) -> md.Motion:
  return md.Motion(
    fps=motion.fps,
    root_pos=motion.root_pos[::-1].copy(),
    root_quat=motion.root_quat[::-1].copy(),
    joint_pos=motion.joint_pos[::-1].copy(),
  )


spider.cut, spider.hold, spider.reverse = cut, hold, reverse
assert spider.check_helpers()

def clips() -> list[md.Clip]:
  rec = spider.recorded()
  down = spider.settle_into_stance(cut(rec, 0.0, spider.WIGGLE_UNTIL_S))
  idle = hold(down, down.duration, 1.0)
  # spider_up: the stance back to standing, wiggling on the way = spider_down backwards.
  up = reverse(down)
  return [
    md.Clip("spider_down", down, next="spider_idle"),
    md.Clip("spider_idle", idle, loop=True, next="spider_idle"),
    md.Clip("spider_up", up, next=None),
  ]


ORDER = ["spider_down", "spider_idle", "spider_idle", "spider_up"]

if mode == "clips":
  library = md.concat_clips(clips(), order=ORDER)
  for k, v in library.report.items(): print(f"{k:36} {v}")
  for c in library.clips: print(c)
  CLIP_FILE = library.save_npz(OUT / "spider_walk.npz")
  subprocess.run([sys.executable, "-m", "mjlab.tasks.tracking.config.pupper.play_motion", str(CLIP_FILE)])
  subprocess.run([sys.executable, "-m", "mjlab.tasks.tracking.config.pupper.upload_motion", str(CLIP_FILE), "--name", "spider_walk"], check=True)
  ENTITY = wandb.Api().default_entity
  TRACK_TASK = "Mjlab-Tracking-Flat-Pupper-v3"
  env_cfg = load_env_cfg(TRACK_TASK)
  env_cfg.scene.num_envs = 4096
  agent_cfg = load_rl_cfg(TRACK_TASK)
  agent_cfg.max_iterations = 3000
  agent_cfg.logger = "wandb"
  agent_cfg.run_name = "spider_walk_clips"; agent_cfg.seed = 7  # the default seed hit a NaN at ~iter 230
  launch_training(TRACK_TASK, TrainConfig(env=env_cfg, agent=agent_cfg, registry_name=f"{ENTITY}/mjlab/spider_walk"))
else:
  TASK = "Mjlab-SpiderWalk-Flat-Pupper-v3"
  REWARD_WEIGHTS = {
    # objectives
    "track_linear_velocity": 1.5,  # follow commanded vx/vy
    "track_yaw_velocity": 0.8,  # follow commanded turn rate
    "upright": 0.5,  # stay upright
    "air_time": 0.0,  # off: gait_tracking already sets the stepping rhythm
    "base_height": 2.0,  # hold the body at 4.2 cm (raised: the motors alone sag)
    "gait_tracking": 1.0,  # match the spider reference gait
    "track_angular_velocity": 0.3,  # roll/pitch-rate stabilizer
    # penalties (negative weights)
    "termination": -50.0,  # falling over
    "orientation_l2": -2.0,  # body tilt
    "lin_vel_z_l2": -1.0,  # vertical bouncing
    "ang_vel_xy_l2": -0.05,  # roll/pitch thrash
    "foot_slip": -0.1,  # planted feet sliding
    "action_rate_l2": -0.01,  # jerky actions
    "joint_torques_l2": -0.002,  # effort
    "joint_acc_l2": -2.5e-7,  # joint acceleration
    "stand_still_pose": -0.1,  # fidgeting at zero command (default pose = spider stance)
    "stand_still_joint_velocity": -0.01,  # creeping at zero command
    "knee_ground_contact": -1.0,  # knees on the floor: the sign of sagging
    "abduction_angle": 0.0,  # off: the spider splays its legs on purpose
    "self_collision_l": -1.0,  # left legs colliding
    "self_collision_r": -1.0,  # right legs colliding
  }
  ITERATIONS = 4000
  env_cfg = load_env_cfg(TASK)
  for name, weight in REWARD_WEIGHTS.items():
    env_cfg.rewards[name].weight = weight
  env_cfg.scene.num_envs = 4096
  agent_cfg = load_rl_cfg(TASK)
  agent_cfg.max_iterations = ITERATIONS
  agent_cfg.logger = "wandb"
  agent_cfg.run_name = "spider_walk"
  launch_training(TASK, TrainConfig(env=env_cfg, agent=agent_cfg))
