"""Load Gemini's controllers (gemini_walk and one per policy trick) inactive.

Run by launch/gemini_pupper.launch.py once the course's own controllers are up:

    python3 robot/spawn_controllers.py <param-file> <controller> [<controller> ...]

One spawner loads every controller in turn, and a spawner gives up on the rest of
its list at the first controller that fails to load or configure (for example a
policy JSON that took longer to parse than a service call waited). So afterwards
every controller is checked, and any that is not loaded and configured gets its
own spawner, up to RETRIES times. The result goes to the log and to READY_FILE,
which start.sh waits for.

The tool server also calls spawn() for a trick whose controller is still missing
when Gemini asks for it.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

PARAM_FILE = Path("/tmp/gemini-pupper/trick_controllers.yaml")  # written by the launch file
READY_FILE = Path("/tmp/gemini-pupper/controllers_ready")
RETRIES = 2
# How long one service call to the controller manager may take. The spawner retries a call three
# times, and this script spawns whatever is still missing afterwards, so a lost reply (it happens
# on a fresh stack: "failed to send response ... (timeout)") costs this long, not a minute. It must
# still cover a slow load: the biggest policy file takes a few seconds to parse.
SERVICE_CALL_TIMEOUT_S = 20
LOADED = {"inactive", "active"}  # loaded and configured: ready to switch to


def controller_states() -> dict[str, str]:
  try:
    out = subprocess.run(
      ["ros2", "control", "list_controllers"], capture_output=True, text=True, timeout=20
    ).stdout
  except subprocess.TimeoutExpired:
    return {}
  # "<name>  <type>  <state>", the state possibly in ANSI colors
  states = {}
  for line in re.sub(r"\x1b\[[0-9;]*m", "", out).splitlines():
    words = line.split()
    if len(words) >= 3:
      states[words[0]] = words[2]
  return states


def spawn(names: list[str], param_file: str | Path = PARAM_FILE) -> bool:
  """Load and configure controllers inactive; True if the spawner reported success."""
  cmd = [
    "ros2", "run", "controller_manager", "spawner", *names,
    "--controller-manager", "/controller_manager",
    "--controller-manager-timeout", "60", "--service-call-timeout", str(SERVICE_CALL_TIMEOUT_S),
    "--inactive", "--param-file", str(param_file),
  ]
  return subprocess.run(cmd).returncode == 0


def wait_for_course_controllers(timeout: float = 90.0) -> None:
  """Our large policy files block the controller manager while they load, which made
  the course's spawners time out when they all started together, so go after them."""
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    states = controller_states()
    if states.get("imu_sensor_broadcaster") == "active" and "neural_controller_default" in states:
      return
    time.sleep(1.0)
  print("spawn_controllers: the course controllers did not come up; loading ours anyway", flush=True)


def main() -> int:
  param_file, names = sys.argv[1], sys.argv[2:]
  READY_FILE.unlink(missing_ok=True)
  wait_for_course_controllers()
  spawn(names, param_file)
  for attempt in range(RETRIES):
    states = controller_states()
    missing = [n for n in names if states.get(n) not in LOADED]
    if not missing:
      break
    print(f"spawn_controllers: not ready after attempt {attempt + 1}: "
          + ", ".join(f"{n} ({states.get(n, 'not loaded')})" for n in missing), flush=True)
    for n in missing:
      spawn([n], param_file)
  states = controller_states()
  failed = [n for n in names if states.get(n) not in LOADED]
  summary = f"{len(names) - len(failed)}/{len(names)} controllers ready"
  if failed:
    summary += "; FAILED: " + ", ".join(f"{n} ({states.get(n, 'not loaded')})" for n in failed)
  print("spawn_controllers: " + summary, flush=True)
  READY_FILE.parent.mkdir(parents=True, exist_ok=True)
  READY_FILE.write_text(summary + "\n")
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
