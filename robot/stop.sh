#!/bin/bash
# Stop Gemini Pupper: relax the motors, stop the app, server and robot stack,
# and bring back the robot's normal screen app.
curl -s -m 3 -X POST localhost:8000/stop/immediate >/dev/null
curl -s -m 3 -X POST localhost:8000/walking/deactivate >/dev/null && sleep 2
free_port() {  # stop whatever is serving port 8000 (an older robot server)
  local pids; pids=$(ss -ltnpH "sport = :8000" 2>/dev/null | grep -oP "pid=\\K[0-9]+" | sort -u)
  [ -n "$pids" ] && kill $pids 2>/dev/null && sleep 1
  return 0
}
pkill -f "[r]obot/kiosk.py"; pkill -f "[g]emini-pupper-kiosk"; pkill -f "[r]obot/server.py"; free_port
pkill -f "[v]ision/eyes.py"  # the eyes service on the Pi
pkill -INT -f "[g]emini_pupper.launch.py"
for i in $(seq 1 15); do pgrep -f "[r]os2_control_node" >/dev/null || break; sleep 1; done
pkill -9 -f "[g]emini_pupper.launch.py"; pkill -9 -f "[/]opt/ros/jazzy/lib/"; pkill -9 -f "[/]home/pi/pupperv3-monorepo/ros2_ws/install/"
# The person follower runs from the follow_me_pupper repo, outside the paths above.
# SIGTERM first so it releases the Hailo; a respawned copy can outlive the launch.
pkill -f "[f]ollow_service.py" && sleep 2; pkill -9 -f "[f]ollow_service.py"
sudo -n systemctl start pupper-gui 2>/dev/null
echo "stopped"
