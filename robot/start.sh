#!/bin/bash
# Start Gemini Pupper on the robot.
#   robot/start.sh            robot stack + robot server (use from a laptop through an SSH tunnel)
#   robot/start.sh --kiosk    alias for --face on this local-brain branch
#   robot/start.sh --face     the app full-screen as the local brain's face: no Gemini,
#                             no key; robot/gemma_brain.py listens and talks (run_pupper.sh does this)
# Stop everything with robot/stop.sh.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
LOG=/tmp/gemini-pupper; mkdir -p $LOG
KEY_FILE=${GEMINI_KEY_FILE:-$HOME/.config/gemini-pupper/api_key}
KIOSK=0
case "${1:-}" in --kiosk|--face) KIOSK=1 ;; esac
free_port() {  # stop whatever is serving port 8000 (an older robot server)
  local pids; pids=$(ss -ltnpH "sport = :8000" 2>/dev/null | grep -oP "pid=\\K[0-9]+" | sort -u)
  [ -n "$pids" ] && kill $pids 2>/dev/null && sleep 1
  return 0
}
ROS="source /opt/ros/jazzy/setup.bash; source $HOME/pupperv3-monorepo/ros2_ws/install/local_setup.bash; export ROS_LOCALHOST_ONLY=1"

[ -f "$REPO/web/dist/index.html" ] || { echo "Missing $REPO/web/dist: run robot/setup.sh first (it builds the web app)"; exit 1; }

# A stack only counts as running if its controllers answer. One that a stop is still taking down
# (an earlier run's Ctrl+C, or its ssh dropping, while this one starts) still has a ros2_control_node
# for a few seconds: taking that for "already running" left a robot server with no controllers.
stack_answers() { timeout 20 bash -c "$ROS; ros2 control list_controllers" 2>/dev/null | grep -q gemini_walk; }
stop_stack() {  # what robot/stop.sh does to the ROS launch
  pkill -INT -f "[g]emini_pupper.launch.py"
  for i in $(seq 1 15); do pgrep -f "[r]os2_control_node" >/dev/null || break; sleep 1; done
  pkill -9 -f "[g]emini_pupper.launch.py"; pkill -9 -f "[/]opt/ros/jazzy/lib/"; pkill -9 -f "[/]home/pi/pupperv3-monorepo/ros2_ws/install/"
  sleep 1
}
if pgrep -f "[r]os2_control_node" >/dev/null && stack_answers; then
  echo "[1/3] robot stack already running"
else
  if pgrep -f "[r]os2_control_node|[g]emini_pupper.launch.py" >/dev/null; then
    echo "      a robot stack is there but its controllers don't answer (half stopped?): restarting it"
    stop_stack
  fi
  echo "[1/3] starting robot stack (about a minute; log $LOG/robot.log)"
  rm -f $LOG/controllers_ready
  setsid nohup bash -c "$ROS; exec ros2 launch $HERE/launch/gemini_pupper.launch.py" > $LOG/robot.log 2>&1 < /dev/null &
  # spawn_controllers.py writes this once gemini_walk and every trick controller is loaded
  for i in $(seq 1 240); do [ -f $LOG/controllers_ready ] && break; sleep 1; done
  if [ -f $LOG/controllers_ready ]; then echo "      $(cat $LOG/controllers_ready)"
  else echo "      controllers still loading after 4 min; see $LOG/robot.log"; fi
fi

$HERE/camera_setup.sh  # the fisheye: Lab 5's fixed white balance, and 10 frames a second (robot/camera_setup.sh)

echo "[2/3] starting robot server on localhost:8000 (log $LOG/server.log)"
pkill -f "[r]obot/server.py" 2>/dev/null; free_port
setsid nohup bash -c "$ROS; cd $HERE && exec python3 -u $HERE/server.py" > $LOG/server.log 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -s -o /dev/null localhost:8000/health && break; sleep 1; done
curl -s -o /dev/null localhost:8000/health || { echo "robot server did not start; see $LOG/server.log"; exit 1; }

# The eyes service on the Pi itself: the depth sense and the COCO detector, as a slow fallback (the
# Pi 5 runs hot and throttled with the stack up: a depth frame takes about a second, so this is
# 0.5-1 Hz). Two threads, not four: with all four cores the depth model competed with the 520 Hz
# motor loop and the robot server, and the Pi reached 84 C. 224 px stays: it is the smallest input
# that still finds a backpack on the floor (196 does not; README, the eyes service). The demo runs
# vision/eyes.py on a laptop (5 Hz) or the Jetson, which posts over the Pi's. Without any eyes the obstacle rule, go_to, explore and the follow guard are blind.
# Log: $LOG/eyes.log. PUPPER_EYES=off skips this.
pkill -f "[v]ision/eyes.py" 2>/dev/null
# Gemma uses local Ollama by default. PUPPER_EYES_GEMMA=off is depth + detector only;
# api is an explicit legacy opt-in, never part of the normal local-brain launch.
GEMMA="--no-gemma"
case "${PUPPER_EYES_GEMMA:-ollama}" in
  api) [ -s "$KEY_FILE" ] && GEMMA="--gemma-api --gemma-api-key-file $KEY_FILE" ;;
  # every 2 minutes, not 15 s: each description holds the Mac's Gemma for ~3.5 s, and a spoken
  # request waits behind it
  ollama) GEMMA="--ollama ${PUPPER_OLLAMA:-http://localhost:11435} --gemma gemma4:26b --describe-every 120" ;;
esac
if [ "${PUPPER_EYES:-on}" = "on" ] && python3 -c "import onnxruntime, numpy, PIL" 2>/dev/null; then
  setsid nohup nice -n 10 python3 -u $REPO/vision/eyes.py $GEMMA --hz 1 --depth-size 224 --threads 2 --fetch-depth-model --fetch-detector-model \
    > $LOG/eyes.log 2>&1 < /dev/null &
  for i in $(seq 1 20); do curl -s -o /dev/null localhost:8100/health && break; sleep 1; done
  curl -s -o /dev/null localhost:8100/health && echo "      eyes: depth + detector on the Pi (slow fallback: run vision/eyes.py on a laptop for the demo), Gemma $(case "$GEMMA" in --no-gemma) echo off ;; --gemma-api*) echo "through the API" ;; *) echo "at ${PUPPER_OLLAMA:-localhost}" ;; esac) (log $LOG/eyes.log)" \
    || echo "      eyes: not up yet; see $LOG/eyes.log"
else
  echo "      eyes: python3 -m pip install -r $REPO/vision/requirements.txt mediapipe  (no depth sense until then)"
fi

if [ $KIOSK = 0 ]; then
  echo "[3/3] ready. From a laptop:  ssh -N -L 8000:localhost:8000 pi@<pupper>  then open http://localhost:8000"
  exit 0
fi
echo "[3/3] opening the app on the robot's screen"
export XDG_RUNTIME_DIR=/run/user/$(id -u) WAYLAND_DISPLAY=wayland-0
sudo -n systemctl stop pupper-gui 2>/dev/null
pkill -f "[g]emini-pupper-kiosk" 2>/dev/null; pkill -f "[r]obot/kiosk.py" 2>/dev/null; sleep 1
# Preserve the volume selected by run_pupper.sh or the person at the robot.
setsid nohup chromium --ozone-platform=wayland --disable-features=WebRtcAllowInputVolumeAdjustment --kiosk --noerrdialogs --disable-infobars --no-first-run \
  --password-store=basic --use-fake-ui-for-media-stream --autoplay-policy=no-user-gesture-required \
  --remote-debugging-port=9222 --user-data-dir=$HOME/.cache/gemini-pupper-kiosk "http://localhost:8000/?brain=local" \
  > $LOG/chromium.log 2>&1 < /dev/null &
sleep 5  # the local face connects through the brain heartbeat; no key or WAKE helper
echo "ready. Stop with robot/stop.sh"
