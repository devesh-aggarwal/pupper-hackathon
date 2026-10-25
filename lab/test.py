#!/usr/bin/env python3
"""Run the desk tests (a fake robot: no Pupper, no ROS, no Gemini needed).

    python3 lab/test.py            # everything: the lab's checks, then the hackathon's
    python3 lab/test.py todo3      # one TODO (todo1 ... todo11)
    python3 lab/test.py part2      # one part of the lab (part1 ... part7; part6, the spider, is the notebook)
    python3 lab/test.py robot      # the provided robot (should always pass)
    python3 lab/test.py hackathon  # memory, screen, hearing, reminders, vision, learning
    python3 lab/test.py hackathon vision     # one hackathon group (memory, screen, sound, reminders, vision, eyes, learn, declarations)
    python3 lab/test.py brain      # the local Gemma brain (robot/gemma_brain.py), with a fake Ollama
    python3 lab/test.py brain thor # Gemma on the Thor (tools/thor_relay.py), with a fake vLLM

The lab's tests are in robot/test_tools.py, the hackathon's in robot/test_hackathon.py,
the local brain's in robot/test_gemma_brain.py.
Each file runs in its own interpreter (they each install a fake ROS), with the memory
in a temporary file, never your ~/.config.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROBOT = Path(__file__).resolve().parent.parent / "robot"


def run(script: str, *args: str) -> int:
    env = dict(os.environ)
    env.setdefault("PUPPER_MEMORY_FILE", str(Path(tempfile.mkdtemp(prefix="pupper_memory_test_")) / "memory.json"))
    return subprocess.call([sys.executable, str(ROBOT / script), *args], env=env)


args = sys.argv[1:]
if args[:1] == ["hackathon"]:
    sys.exit(run("test_hackathon.py", *args[1:]))
if args[:2] == ["brain", "thor"]:
    sys.exit(run("test_thor_relay.py", *args[2:]))
if args[:1] == ["brain"]:
    sys.exit(run("test_gemma_brain.py", *args[1:]))
if args:
    sys.exit(run("test_tools.py", *args))
lab = run("test_tools.py")
print("=" * 60)
hack = run("test_hackathon.py")
print("=" * 60)
brain = run("test_gemma_brain.py")
print("=" * 60)
thor = run("test_thor_relay.py")
sys.exit(lab or hack or brain or thor)
