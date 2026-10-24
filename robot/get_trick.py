#!/usr/bin/env python3
"""Fetch a trick from Weights & Biases into tricks/<name>/.

Pupper trick training (pupper-mjlab's Mjlab-Tracking-Flat-Pupper-v3) uploads the
trick policy to its W&B run as policy.json; the training log prints the exact
command to run here. The Pupper is already logged into W&B for policy deploys.

    python3 robot/get_trick.py <entity>/mjlab/<run-id> --name sit
    python3 robot/get_trick.py mjlab/<run-id> --name sit          # your default entity

A trick that walks in its own stance (the spider) has two policies: its clips
(a trick run) and its walking policy (a Mjlab-SpiderWalk run). Fetch both at once:

    python3 robot/get_trick.py <clip-trick-run> --walk <spider-walk-run> --name spider_walk

The trick lands as tricks/<name>/policy.json. Every trick on the robot is a
policy: check a design in simulation (pupper-mjlab's play_motion) before
training it, not by replaying it open loop on the robot.

If tricks/<name>/trick.yaml does not exist yet a template is written; edit its
description (it is what Gemini reads). Restart Gemini Pupper afterwards
(robot/stop.sh, then robot/start.sh): the launch file loads one controller per
policy trick at startup.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TRICK_DIR = Path(__file__).resolve().parent.parent / "tricks"
VALIDATOR = Path.home() / "pupper_gait_deploy" / "download_policy.py"


def resolve_run_path(run: str, project: str, entity: str | None) -> str:
  """Accept entity/project/run-id, project/run-id or a bare run id."""
  import wandb

  parts = run.split("/")
  if len(parts) == 3:
    return run
  if len(parts) == 2:
    project, run = parts
  entity = entity or wandb.Api().default_entity
  if entity is None:
    sys.exit("Could not determine the W&B entity; pass a full entity/project/run-id path.")
  return f"{entity}/{project}/{run}"


def fetch_policy(run_path: str, dest: Path) -> dict:
  import wandb

  run = wandb.Api().run(run_path)
  if "policy.json" not in [f.name for f in run.files()]:
    sys.exit(
      f"{run_path} has no policy.json yet. Trick runs upload it at every checkpoint "
      "save and on Ctrl+C; wait for the first checkpoint, or check the run id."
    )
  with tempfile.TemporaryDirectory() as tmp:
    path = Path(run.file("policy.json").download(root=tmp, replace=True).name)
    policy = json.loads(path.read_text())
    if "trick" not in policy or "motion" not in policy:
      sys.exit(f"{run_path}'s policy.json is not a trick policy (no trick/motion blocks).")
    if VALIDATOR.exists():  # the controller's own checks: layout, shapes, clips
      check = subprocess.run([sys.executable, str(VALIDATOR), "--validate-only", str(path)])
      if check.returncode != 0:
        sys.exit("validation failed; not installing")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest / "policy.json")
  print(f"Run {run.name} ({run_path}) -> {dest / 'policy.json'}")
  return policy


def fetch_walk_policy(run_path: str, dest: Path) -> None:
  """A walking trick's walking policy (a velocity policy, not a trick) -> walk_policy.json."""
  import wandb

  run = wandb.Api().run(run_path)
  if "policy.json" not in [f.name for f in run.files()]:
    sys.exit(f"{run_path} has no policy.json yet; wait for the run to save a checkpoint.")
  with tempfile.TemporaryDirectory() as tmp:
    path = Path(run.file("policy.json").download(root=tmp, replace=True).name)
    if "trick" in json.loads(path.read_text()):
      sys.exit(f"{run_path} is a trick run; --walk takes the walking policy's run (Mjlab-SpiderWalk).")
    if VALIDATOR.exists():
      check = subprocess.run([sys.executable, str(VALIDATOR), "--validate-only", str(path)])
      if check.returncode != 0:
        sys.exit("validation failed; not installing")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest / "walk_policy.json")
  print(f"Run {run.name} ({run_path}) -> {dest / 'walk_policy.json'}")


def write_template(dest: Path, name: str, policy: dict, walks: bool = False) -> None:
  spec = dest / "trick.yaml"
  if spec.exists():
    print(f"Kept the existing {spec}")
    return
  lines = [
    f"# {name}: fetched with get_trick.py. Edit the description -- it is what Gemini reads",
    "# when deciding whether to play this trick.",
    f"description: TODO describe what the robot does in {name}.",
  ]
  lines += ["type: policy", "policy: policy.json", "start_pose: standing", "end_pose: standing"]
  lines.append("max_body_angle_deg: 60   # the tilt e-stop while the trick plays")
  clips = policy.get("trick", {}).get("clips", [])
  if clips:
    lines += [
      "# This policy has clips. Name the ones Gemini may ask for while the trick runs,",
      "# and the clip that returns to standing (played before any other motion):",
      "# actions:",
    ]
    lines += [f"#   {c['name']}: TODO what this clip does" for c in clips]
    lines += ["# exit_clip: <clip>", "# end_pose: <the pose the looping clip holds>"]
  if walks:
    cmd = json.loads((dest / "walk_policy.json").read_text())["command_clip"]
    lines += [
      "walk:                      # it walks in its stance: move, turn and arc drive this policy",
      "  policy: walk_policy.json",
      "  speed: TODO              # default m/s for move and arc",
      f"  max_forward: TODO        # m/s; at most {max(map(abs, cmd['vx'])):g} (what it was trained for)",
      f"  max_sideways: TODO       # m/s; at most {max(map(abs, cmd['vy'])):g}",
      f"  turn: TODO               # deg/s; at most {math.degrees(max(map(abs, cmd['wz']))):.0f}",
      "  run: false",
      "  note: TODO anything else Gemini should know about walking like this",
    ]
  spec.write_text("\n".join(lines) + "\n")
  print(f"Wrote a template {spec}; edit its description.")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("run", help="W&B training run: entity/project/run-id, project/run-id or run-id")
  parser.add_argument("--name", required=True, help="trick name: the folder under tricks/ and Gemini's id")
  parser.add_argument("--project", default="mjlab")
  parser.add_argument("--entity", default=None)
  parser.add_argument("--walk", default=None, help="a walking policy's run (Mjlab-SpiderWalk), for a trick that walks")
  args = parser.parse_args()

  dest = TRICK_DIR / args.name
  policy = fetch_policy(resolve_run_path(args.run, args.project, args.entity), dest)
  trick = policy["trick"]
  print(f"Trick '{trick['name']}': {trick['frames']} frames, {trick['duration_s']:.1f} s")
  for c in trick.get("clips", []):
    print(f"  clip {c['name']:14} {(c['end'] - c['start']) / trick['fps']:.1f} s"
          f"{' loops' if c.get('loop') else ''}{' -> ' + str(c['next']) if c.get('next') is not None else ''}")
  if args.walk:
    fetch_walk_policy(resolve_run_path(args.walk, args.project, args.entity), dest)
  write_template(dest, args.name, policy, walks=bool(args.walk))
  print("Restart Gemini Pupper (robot/stop.sh, then robot/start.sh) to load it.")


if __name__ == "__main__":
  main()
