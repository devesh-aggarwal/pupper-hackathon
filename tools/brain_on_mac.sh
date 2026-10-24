#!/usr/bin/env bash
# Pupper's brain on this Mac, no cloud:   ./run_pupper.sh [volume 0-100]   (on the Mac; it runs this)
#
#   1. finds the Pupper over ssh (the address saved in ~/.config/pupper/ssh; asked for once when it
#      changes, e.g. a new ngrok address) and updates its code from the repo (git pull)
#   2. starts the brain's servers on this Mac, if they aren't up: Ollama with Gemma (port 11435) and
#      whisper.cpp (speech to text, port 8178), for this Mac only
#   3. starts the link: an ssh connection to the Pupper, kept up in the background, that makes
#      those two ports localhost on the Pupper too (the Pupper can't reach this Mac by itself)
#   4. starts the eyes on this Mac (vision/eyes.py: depth, detector and Gemma, 5 frames a second,
#      against the Pupper's camera; the Pupper's own eyes manage about 1 and stand down for these)
#   5. runs ./run_pupper.sh on the Pupper: the robot, its face, its mic and speaker, the brain
#
# Ctrl+C stops the robot. The servers and the link keep running, so ./run_pupper.sh on the Pupper
# itself works too, until the Mac sleeps or restarts.   tools/brain_on_mac.sh stop   ends them.
# Gemma is gemma4:26b once it is pulled (ollama pull gemma4:26b, about 17 GB, once); until then the
# best Gemma this Mac already has. The first run also downloads whisper's model (about 550 MB).
# While the link is up, http://localhost:8000 on this Mac is Pupper's app (its face and debug view).
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
PUP="$HERE/tools/pup"
WANT=gemma4:26b    # the best brain this Mac can run: a mixture of experts, 4B active per token
WHISPER_MODEL=${PUPPER_WHISPER_MODEL:-large-v3-turbo-q5_0}  # far better than base.en in a noisy room; fast on the GPU
OLLAMA_PORT=11435  # not 11434: the Ollama app may hold that
WHISPER_PORT=8178
EYES_PORT=8101     # the Mac's eyes (vision/eyes.py) answer /look here; the Pupper's own use 8100
WHISPER_ARGS="-l en -sns"  # English, and no tokens for non-speech (coughs, music, chatter)
CACHE=$HOME/.cache/pupper
LOG=$CACHE/logs
SAVED=$HOME/.config/pupper/ssh
mkdir -p "$LOG" "$(dirname "$SAVED")"
die() { echo "!! $*" >&2; exit 1; }
alive() { [ -f "$CACHE/$1.pid" ] && kill -0 "$(cat "$CACHE/$1.pid")" 2>/dev/null; }

if [ "${1:-}" = stop ]; then
  for name in eyes web link whisper ollama; do alive $name && kill "$(cat "$CACHE/$name.pid")" && echo "stopped $name"; rm -f "$CACHE/$name.pid"; done
  pkill -f "ssh .*-R $OLLAMA_PORT:localhost:$OLLAMA_PORT" 2>/dev/null
  exit 0
fi

# --- 1. the Pupper
reach() { "$PUP" sh true >/dev/null 2>&1; }
if ! reach; then
  [ -t 0 ] || die "can't reach the Pupper over ssh ($(cat "$SAVED" 2>/dev/null || echo pi@pupper5.local))"
  echo "Can't reach the Pupper at $(cat "$SAVED" 2>/dev/null || echo pi@pupper5.local)."
  read -rp "Paste the ssh command you use to reach it (e.g. ssh -p 18196 pi@4.tcp.us-cal-1.ngrok.io): " cmd
  port=$(echo "$cmd" | grep -oE -- '-p *[0-9]+' | grep -oE '[0-9]+'); host=$(echo "$cmd" | grep -oE '[^ ]+@[^ ]+' | head -1)
  [ -n "$host" ] || die "no user@host in that"
  echo "$host ${port:-22}" > "$SAVED"
  reach || die "still can't reach $host (port ${port:-22}). Once per laptop: ./tools/pup setup"
fi
echo "Pupper: $(cat "$SAVED" 2>/dev/null || echo pi@pupper5.local)"
# The code comes from the repo: push from here, and the Pupper pulls it.
"$PUP" sh "cd ~/pupper-hackathon && git pull -q --ff-only" \
  || echo "!! git pull on the Pupper failed (local changes there?); it runs the code it has"

# --- 2. the brain's servers (they outlive this script: the Pupper may start on its own later)
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT  # this Mac only: the link carries it to the Pupper
ollama_up() { curl -sf -m 2 "http://$OLLAMA_HOST/api/version" >/dev/null; }
command -v ollama >/dev/null || die "no Ollama: brew install ollama (or ollama.com/download)"
if ! ollama_up; then
  # The context the brain asks for (so the eyes' requests share it and nothing reloads), kept
  # loaded, two at a time (the eyes don't push the brain's prompt out of the cache).
  OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=2 nohup ollama serve > "$LOG/ollama.log" 2>&1 &
  echo $! > "$CACHE/ollama.pid"
  for _ in $(seq 20); do ollama_up && break; sleep 1; done
  ollama_up || die "Ollama didn't start (log $LOG/ollama.log)"
  nohup caffeinate -i -w "$(cat "$CACHE/ollama.pid")" >/dev/null 2>&1 &  # no sleeping while it's the brain
fi
if ! ollama show "$WANT" >/dev/null 2>&1; then
  HAVE=$(cd "$HERE" && python3 -c 'import sys; sys.path.insert(0, "robot"); import gemma_brain as g; print(g.pick_model(sys.argv[1:]) or "")' \
         $(ollama list 2>/dev/null | awk 'NR > 1 {print $1}'))
  if [ -n "$HAVE" ]; then
    echo "Gemma: $WANT isn't pulled yet, so $HAVE for now (the better brain: ollama pull $WANT, about 17 GB)"
  else
    echo "pulling $WANT (once, about 17 GB)..."
    ollama pull "$WANT" || die "couldn't pull $WANT. Gemma 4 needs a recent Ollama: brew upgrade ollama, or update the app"
  fi
fi

whisper_up() { curl -sf -m 2 -o /dev/null "http://127.0.0.1:$WHISPER_PORT/"; }
if whisper_up && [ "$(cat "$CACHE/whisper.model" 2>/dev/null)" != "$WHISPER_MODEL $WHISPER_ARGS" ]; then  # older settings: restart
  alive whisper && kill "$(cat "$CACHE/whisper.pid")"
  pkill -f "whisper-server .*--port $WHISPER_PORT" 2>/dev/null; sleep 1
fi
if ! whisper_up; then
  if ! command -v whisper-server >/dev/null; then
    command -v brew >/dev/null || die "no whisper-server and no Homebrew (brew.sh): install one, then brew install whisper.cpp"
    echo "installing whisper.cpp (brew install whisper.cpp, once)..."
    HOMEBREW_NO_AUTO_UPDATE=1 brew install whisper.cpp >/dev/null || die "brew install whisper.cpp failed"
  fi
  WM=$CACHE/ggml-$WHISPER_MODEL.bin
  if [ ! -f "$WM" ]; then
    echo "downloading whisper's $WHISPER_MODEL model (once)..."
    # -C -: a download cut short (a network change) picks up where it stopped, on retry or next run
    curl -fL -C - --retry 5 --retry-delay 2 --progress-bar -o "$WM.part" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$WHISPER_MODEL.bin" \
      && mv "$WM.part" "$WM" || die "the whisper model download failed"
  fi
  nohup whisper-server -m "$WM" $WHISPER_ARGS --host 127.0.0.1 --port $WHISPER_PORT > "$LOG/whisper.log" 2>&1 &
  echo $! > "$CACHE/whisper.pid"; echo "$WHISPER_MODEL $WHISPER_ARGS" > "$CACHE/whisper.model"
  for _ in $(seq 30); do whisper_up && break; sleep 1; done
  whisper_up || die "whisper-server didn't start (log $LOG/whisper.log)"
fi
echo "brain: Ollama :$OLLAMA_PORT, whisper :$WHISPER_PORT ($WHISPER_MODEL) (logs in $LOG)"

# --- 3. the link: the two ports on the Pupper, through ssh, reconnecting whenever it drops
pupper_sees_brain() { "$PUP" sh "curl -sf -m 3 localhost:$OLLAMA_PORT/api/version >/dev/null && curl -sf -m 3 -o /dev/null localhost:$WHISPER_PORT/" >/dev/null 2>&1; }
PORTS="$OLLAMA_PORT $WHISPER_PORT $EYES_PORT"
if alive link && [ "$(cat "$CACHE/link.ports" 2>/dev/null)" != "$PORTS" ]; then  # an older link: other ports
  kill "$(cat "$CACHE/link.pid")"; pkill -f "ssh .*-R $OLLAMA_PORT:localhost:$OLLAMA_PORT" 2>/dev/null; sleep 1
fi
if ! alive link; then
  echo "$PORTS" > "$CACHE/link.ports"
  nohup bash -c "while true; do
      PUPPER_TUNNEL_PORTS='$PORTS' '$PUP' link
      sleep 3
    done" > "$LOG/link.log" 2>&1 &
  echo $! > "$CACHE/link.pid"
fi
for _ in $(seq 20); do pupper_sees_brain && break; sleep 1; done
pupper_sees_brain || die "the Pupper can't see the brain through the link (log $LOG/link.log)"
echo "link: up (the Pupper reaches the brain at localhost:$OLLAMA_PORT and :$WHISPER_PORT)"
if ! alive web; then  # Pupper's app on this Mac; if port 8000 is taken here, it just keeps retrying
  nohup bash -c "while true; do '$PUP' web; sleep 10; done" > "$LOG/web.log" 2>&1 &
  echo $! > "$CACHE/web.pid"
fi
# --- 4. the eyes, here: the robot server through the web forward, /look back through the link
VENV=$CACHE/venv
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import onnxruntime, numpy, PIL, mediapipe" 2>/dev/null; then
  echo "setting up the eyes' Python packages (once)..."
  python3 -m venv "$VENV" && "$VENV/bin/pip" install -q -r "$HERE/vision/requirements.txt" mediapipe \
    || die "the eyes' packages didn't install (log above)"
fi
if ! alive eyes; then
  nohup bash -c "cd '$HERE' && while true; do
      '$VENV/bin/python' -u vision/eyes.py --robot http://localhost:8000 --ollama http://127.0.0.1:$OLLAMA_PORT \
        --gemma '$(ollama show "$WANT" >/dev/null 2>&1 && echo "$WANT" || echo "${HAVE:-$WANT}")' --hz 5 --always \
        --port $EYES_PORT --advertise http://127.0.0.1:$EYES_PORT --source mac --describe-every 120 \
        --fetch-depth-model --fetch-detector-model
      sleep 5
    done" > "$LOG/eyes.log" 2>&1 &
  echo $! > "$CACHE/eyes.pid"
fi
echo "eyes: on this Mac at 5 Hz (log $LOG/eyes.log)"
echo "watch: http://localhost:8000 (Pupper's app), tail -f $LOG/ollama.log (each Gemma request),"
echo "       OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT ollama ps (the model on the GPU)"

# --- 5. the Pupper
echo
"$PUP" tty "cd ~/pupper-hackathon && ./run_pupper.sh $*"
"$PUP" sh "~/pupper-hackathon/robot/stop.sh" >/dev/null 2>&1 \
  || echo "!! couldn't reach the Pupper to stop it: use the e-stop, then ./tools/pup sh pupper-hackathon/robot/stop.sh"
echo "The brain and the link keep running: ./run_pupper.sh on the Pupper works too. To end them: tools/brain_on_mac.sh stop"
