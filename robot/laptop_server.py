#!/usr/bin/env python3
"""The robot server on a laptop: everything real except the robot (nothing moves).

    ./run_local.sh                    (builds the web app if needed, then runs this)
    python3 robot/laptop_server.py    (if web/dist is already built)

The real server.py, your lab/ tools and prompt, the reminders and the web app at
http://localhost:8000, talking to the real Gemini Live from your browser (your
laptop's mic, speaker and camera). Only ROS is swapped for the fake robot from
test_tools.py, so every motion is logged here instead of moving the legs:
"[fake robot] legs: gemini_walk took over", "[fake robot] walking vx=0.4 ...".

Reminders are the real ones, saved to $PUPPER_REMINDERS_FILE (default here:
~/.config/gemini-pupper/reminders-laptop.json, so they never mix with the Pupper's).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("PUPPER_REMINDERS_FILE",
                      str(Path.home() / ".config" / "gemini-pupper" / "reminders-laptop.json"))

import test_tools  # noqa: E402  (installs the fake ROS before robot_api is imported, and loads lab/)
import robot_api  # noqa: E402
from tricks import load_tricks  # noqa: E402

log = logging.getLogger("fake_robot")


class LoggingFakeRobot(test_tools.FakeRobot):
    """The tests' fake robot, in real time, saying what the real one would do."""

    def __init__(self) -> None:
        super().__init__(test_tools.VirtualClock())
        self.last_speed = (0.0, 0.0, 0.0)

    def handle(self, service: str, req):
        res = super().handle(service, req)
        if service.endswith("switch_controller") and (req.activate_controllers or req.deactivate_controllers):
            on = ", ".join(req.activate_controllers)
            log.info("[fake robot] legs: %s", f"{on} took over" if on else "relaxed (lying down)")
        elif service in ("activate_person_following", "deactivate_person_following"):
            log.info("[fake robot] person follower %s", "on" if service.startswith("activate") else "off")
        return res

    def publish(self, topic: str, msg) -> None:
        super().publish(topic, msg)
        if topic == "/llm_cmd_vel":
            speed = (round(msg.linear.x, 2), round(msg.linear.y, 2), round(msg.angular.z * 57.2958))
            if speed != self.last_speed:  # it is sent 20 times a second: log changes only
                self.last_speed = speed
                log.info("[fake robot] %s", "standing still" if speed == (0, 0, 0) else
                         "walking vx=%g m/s vy=%g m/s wz=%g deg/s" % speed)
        elif topic.endswith("/clip"):
            log.info("[fake robot] %s plays clip %s", topic.split("/")[1], msg.data)


def main() -> None:
    robot = test_tools.ROBOT = LoggingFakeRobot()
    tricks = load_tricks().values()
    names = (robot_api.LEG_CONTROLLERS | {t.controller for t in tricks if t.kind == "policy"}
             | {t.walk_controller for t in tricks if t.walks})
    robot.controllers = {n: "inactive" for n in names}  # loaded, none active: motors idle, like after start.sh

    import server  # the real one
    import uvicorn

    if not server.WEB_DIST.is_dir():
        sys.exit("web/dist is missing: build the web app first (./run_local.sh does it)")
    port = int(os.environ.get("PORT", "8000"))
    print(f"\nGemini Pupper on your laptop, with a fake robot (nothing moves).\n"
          f"  Open http://localhost:{port} , Settings -> paste your Gemini API key -> WAKE PUPPER\n"
          f"  Reminders file: {os.environ['PUPPER_REMINDERS_FILE']}\n"
          f"  Ctrl+C to stop.\n", flush=True)
    # No request log: the web app asks for reminders every 5 s and for status often; the tool
    # calls, the fake robot's moves and the reminders are logged on their own.
    uvicorn.run(server.app, host="127.0.0.1", port=port, access_log=False)


if __name__ == "__main__":
    main()
