#!/bin/bash
# Pupper's fisheye camera, set up on the running camera node (start.sh runs this once the stack is up):
#   - fixed white balance: the lens's UV layer leaves auto white balance out of range, so frames come
#     out pink; Lab 5 (follow_me_pupper/robot/camera_params.yaml) fixed it with these gains, tuned
#     under ceiling lights. Gemini, the browser's detector and the eyes service all get the same frames.
#   - frame rate: nothing reads more than a few frames a second (the browser 1, the eyes at most 5),
#     and encoding 1400 x 1050 JPEGs at the sensor's full rate cost the Pi half a core, so the camera
#     is asked for CAMERA_FPS (10) frames a second instead.
#   robot/camera_setup.sh            # gains red 1.4, blue 3.3; 10 fps
#   robot/camera_setup.sh 1.6 3.0    # other gains (raise red to warm, raise blue to cool)
#   robot/camera_setup.sh auto       # back to auto white balance (the frame rate is still set)
#   CAMERA_FPS=30 robot/camera_setup.sh   # another frame rate
set -o pipefail  # not -u: the ROS setup script reads unset variables
source /opt/ros/jazzy/setup.bash; source $HOME/pupperv3-monorepo/ros2_ws/install/local_setup.bash; export ROS_LOCALHOST_ONLY=1
# Each call is given 20 s: a ros2 param call can hang on discovery, and start.sh must not wait on it.
setp() { timeout 20 ros2 param set --spin-time 5 /camera "$@" >/dev/null 2>&1; }
FPS=${CAMERA_FPS:-10}
US=$((1000000 / FPS))
setp FrameDurationLimits "[$US, $US]" && echo "camera: $FPS frames a second" || echo "camera: could not set the frame rate (is the robot stack up?)"
if [ "${1:-}" = "auto" ]; then
  setp AwbEnable true && echo "camera: auto white balance" || echo "camera: could not set white balance"
  exit 0
fi
RED=${1:-1.4}; BLUE=${2:-3.3}
setp AwbEnable false && setp ColourGains "[$RED, $BLUE]" && \
echo "camera: fixed white balance, gains red $RED blue $BLUE" || echo "camera: could not set white balance (is the robot stack up?)"
