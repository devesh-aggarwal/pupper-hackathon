#!/bin/bash
# Start Gemini Pupper on the robot.
#   robot/start.sh            robot stack + robot server (use from a laptop through an SSH tunnel)
#   robot/start.sh --kiosk    also show the app full-screen on the robot, using its mic and speaker
#   robot/start.sh --face     the app full-screen as the local brain's face (?brain=local): no Gemini,
#                             no key; robot/gemma_brain.py listens and talks (run_pupper.sh does this)
# Stop everything with robot/stop.sh.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
LOG=/tmp/gemini-pupper; mkdir -p $LOG
KEY_FILE=${GEMINI_KEY_FILE:-$HOME/.config/gemini-pupper/api_key}
KIOSK=0; FACE=0
case "${1:-}" in --kiosk) KIOSK=1 ;; --face) KIOSK=1 FACE=1 ;; esac
free_port() {  # stop whatever is serving port 8000 (an older robot server)
  local pids; pids=$(ss -ltnpH "sport = :8000" 2>/dev/null | grep -oP "pid=\\K[0-9]+" | sort -u)
  [ -n "$pids" ] && kill $pids 2>/dev/null && sleep 1
  return 0
}
ROS="source /opt/ros/jazzy/setup.bash; source $HOME/pupperv3-monorepo/ros2_ws/install/local_setup.bash; export ROS_LOCALHOST_ONLY=1"

[ -f "$REPO/web/dist/index.html" ] || { echo "Missing $REPO/web/dist: run robot/setup.sh first (it builds the web app)"; exit 1; }
if [ $KIOSK = 1 ] && [ $FACE = 0 ] && [ ! -s "$KEY_FILE" ]; then
  echo "Missing $KEY_FILE. Run robot/setup.sh, or create it with:  mkdir -p $(dirname $KEY_FILE) && (umask 077; cat > $KEY_FILE)"; exit 1
fi

if pgrep -f "[r]os2_control_node" >/dev/null; then
  echo "[1/3] robot stack already running"
else
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
# 0.5-1 Hz). The demo runs vision/eyes.py on a laptop (5 Hz) or the Jetson, which posts over the
# Pi's. Without any eyes the obstacle rule, go_to, explore and the follow guard are blind.
# Log: $LOG/eyes.log. PUPPER_EYES=off skips this.
pkill -f "[v]ision/eyes.py" 2>/dev/null
# Gemma: through Google's API with the Gemini key while there is no Jetson; PUPPER_EYES_GEMMA=ollama
# asks the local brain's Ollama instead ($PUPPER_OLLAMA, $PUPPER_GEMMA: the Mac or the Jetson), and
# PUPPER_EYES_GEMMA=off is depth + detector only.
GEMMA="--no-gemma"
case "${PUPPER_EYES_GEMMA:-api}" in
  api) [ -s "$KEY_FILE" ] && GEMMA="--gemma-api --gemma-api-key-file $KEY_FILE" ;;
  ollama) GEMMA="--ollama ${PUPPER_OLLAMA:-http://localhost:11434} --gemma ${PUPPER_GEMMA:-gemma4:26b}" ;;
esac
if [ "${PUPPER_EYES:-on}" = "on" ] && python3 -c "import onnxruntime, numpy, PIL" 2>/dev/null; then
  setsid nohup nice -n 10 python3 -u $REPO/vision/eyes.py $GEMMA --hz 1 --depth-size 224 --fetch-depth-model --fetch-detector-model \
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
amixer -q sset Master 100% 2>/dev/null; wpctl set-volume @DEFAULT_SINK@ 1.0 2>/dev/null
setsid nohup chromium --ozone-platform=wayland --disable-features=WebRtcAllowInputVolumeAdjustment --kiosk --noerrdialogs --disable-infobars --no-first-run \
  --password-store=basic --use-fake-ui-for-media-stream --autoplay-policy=no-user-gesture-required \
  --remote-debugging-port=9222 --user-data-dir=$HOME/.cache/gemini-pupper-kiosk "http://localhost:8000/$([ $FACE = 1 ] && echo '?brain=local')" \
  > $LOG/chromium.log 2>&1 < /dev/null &
if [ $FACE = 1 ]; then  # the face only: nothing presses WAKE PUPPER, so Gemini never connects
  sleep 5
else
  setsid nohup python3 -u $HERE/kiosk.py "$KEY_FILE" > $LOG/kiosk.log 2>&1 < /dev/null &
  sleep 12; tail -2 $LOG/kiosk.log
fi
echo "ready. Stop with robot/stop.sh"
