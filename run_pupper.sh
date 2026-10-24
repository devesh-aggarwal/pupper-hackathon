#!/bin/bash
# One command to run Pupper on the robot:   ./run_pupper.sh [volume 0-100] [--laptop]
#
# Does every setup step for you: puts Node on the PATH, rebuilds the
# web app if its sources are newer than the build, installs missing Python packages,
# then starts the robot stack, the robot server and (unless --laptop) the app full-screen
# on Pupper's own screen with its mic and speaker. Then it shows every tool call, event
# (reminders, sound alerts, searches) and outcome live. Ctrl+C stops everything (robot/stop.sh).
#
# --laptop: no kiosk. Open the app from your laptop instead:
#     ssh -N -L 8000:localhost:8000 pi@pupper5.local   then http://localhost:8000
#
# The brain is Gemma on the Mac (no cloud): ./run_pupper.sh on the Mac starts it, with a link that
# makes it localhost:11435 (Gemma) and :8178 (whisper) here, then runs this script here. While that
# link is up, ./run_pupper.sh here works on its own too. The kiosk is the brain's face, and
# robot/gemma_brain.py listens and talks through Pupper's mic and speaker.
#
# Motors stay idle until someone asks Pupper to move. Keep it on the floor, hand near the
# e-stop (the right stick on the gamepad relaxes the motors).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
[ "$(uname)" = Darwin ] && exec "$HERE/tools/brain_on_mac.sh" "$@"  # on the Mac: the local brain
VOLUME=100; KIOSK=1  # Pupper's voice (Piper) needs it: at 30% it was hard to hear across a room
for arg in "$@"; do
  case "$arg" in
    --laptop|--no-kiosk) KIOSK=0 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *[!0-9]*) echo "unknown argument: $arg (volume 0-100, or --laptop)"; exit 2 ;;
    *) VOLUME=$arg ;;
  esac
done
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export PATH=$HOME/.local/node/bin:$PATH   # robot/setup.sh installs Node there
export PUPPER_MEMORY_FILE=${PUPPER_MEMORY_FILE:-$HOME/.config/gemini-pupper/memory.json}
export PUPPER_OLLAMA=${PUPPER_OLLAMA:-http://localhost:11435}    # the Mac's Gemma, through its link
export PUPPER_WHISPER=${PUPPER_WHISPER:-http://localhost:8178}   # the Mac's whisper, the same way

stop_all() {
  echo; echo "stopping..."
  "$HERE/robot/stop.sh"
  exit 0
}
trap stop_all INT TERM HUP  # HUP: the Mac's ssh went away (Ctrl+C there, or the WiFi dropped)

# --- 0. the brain: can the Pupper reach Gemma and whisper on the Mac?
if ! curl -sf -m 3 "$PUPPER_OLLAMA/api/version" >/dev/null; then
  echo "Pupper's brain isn't reachable: on the Mac, run ./run_pupper.sh (it starts the brain and links it here)."
  exit 1
fi
python3 robot/gemma_brain.py --check --voice || { echo "nothing started: the Pupper can't use the brain on the Mac (above)"; exit 1; }
[ -f "${PUPPER_PIPER_VOICE:-$HOME/.local/share/piper/en_US-lessac-medium.onnx}" ] && [ -f "$HOME/.cache/pupper/silero_vad.onnx" ] \
  || "$HERE/robot/setup_voice.sh"

# --- 1. the web app build (web/dist), rebuilt when its sources changed. By content, not by date:
# tools/pup sync copies the Mac's dates, which can be older than the last build here.
web_sources() { find web -type f -not -path 'web/node_modules/*' -not -path 'web/dist/*' | sort | xargs cat | sha1sum | cut -d' ' -f1; }
need_build=0
if [ ! -f web/dist/index.html ] || [ "$(cat web/dist/.sources 2>/dev/null)" != "$(web_sources)" ]; then need_build=1; fi
if [ $need_build = 1 ]; then
  if ! command -v npm >/dev/null; then
    echo "web app not built and no npm: running robot/setup.sh first"; "$HERE/robot/setup.sh" || exit 1
  else
    echo "building the web app (sources changed)..."
    (cd web && npm ci --no-audit --no-fund --loglevel=error && npm run build --silent 2>&1 | grep -vE "chunks are larger|dynamic import|manualChunks|chunkSizeWarningLimit|^$") || { echo "web build failed"; exit 1; }
  fi
  [ -f web/dist/index.html ] && web_sources > web/dist/.sources
fi

# --- 2. Python packages for the robot server
if ! python3 -c "import fastapi, uvicorn, psutil, pydantic, websockets, yaml" 2>/dev/null; then
  echo "installing the robot server's Python packages..."
  python3 -m pip install --user --break-system-packages -q fastapi uvicorn psutil pydantic websockets pyyaml || exit 1
fi

if ! python3 -c "import onnxruntime, mediapipe" 2>/dev/null; then
  echo "installing the eyes service's Python packages (depth + detector)..."
  python3 -m pip install --user --break-system-packages -q -r "$HERE/vision/requirements.txt" mediapipe || echo "WARNING: eyes packages failed; no depth sense"
fi

# --- 3. sound
if arecord -l 2>/dev/null | grep -q card; then
  echo "mic: $(arecord -l | grep card | head -1 | sed -E 's/.*\[([^]]+)\].*\[.*/\1/')"
else
  echo "WARNING: no microphone found (plug in the webcam or a USB mic); Pupper will not hear you"
fi
wpctl set-volume @DEFAULT_AUDIO_SINK@ "$(awk "BEGIN{print $VOLUME/100}")" 2>/dev/null && echo "volume: $VOLUME%"

# --- 4. memory: who Pupper is talking with (python3 lab/hackathon_memory.py show / reset)
echo "memory: $(python3 lab/hackathon_memory.py summary 2>/dev/null || echo 'unavailable')"

# --- 5. start: the kiosk is the brain's face; the eyes ask the Mac's Gemma
PUPPER_BRAIN=gemma PUPPER_EYES_GEMMA=ollama "$HERE/robot/start.sh" $([ $KIOSK = 1 ] && echo --face) \
  || { echo "start failed; see /tmp/gemini-pupper/*.log"; "$HERE/robot/stop.sh"; exit 1; }
echo "state: $(curl -s localhost:8000/system/mode)"
cat <<'EOF'

Talk to Pupper from about an arm's length. Try, in order:
  "Hi, I'm Devesh."  ->  "Remember that I like tennis balls."  ->  "What do you know about me?"
  "Stand up."  ->  "Sit." (wait 10 s)  ->  "Shake your left paw."  ->  "Stand up."
  "Wake me up in one minute."  (then wait: the screen and Pupper both remind you)
  "Where is my cup?"  (Pupper turns in steps and looks)   ->   clap once (the screen flashes LOUD SOUND)
  "Learn a happy dance: a push-up, then spin around."  ->  "Do the happy dance."
Say "stop" (or use the gamepad's right stick) to stop right away. Ctrl+C here shuts everything down.

EOF
echo "--- the brain: Gemma on the Mac (what it hears, says and does) ---"
echo "session log: ~/pupper-logs/$(date +%F)/transcript.jsonl (+ frames/); read it with: python3 lab/hackathon_log.py 80"
python3 -u robot/gemma_brain.py --robot --voice
stop_all
