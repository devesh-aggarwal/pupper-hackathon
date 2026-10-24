#!/usr/bin/env bash
# Pupper with its brain on this Mac, no cloud:   ./run_pupper.sh [volume 0-100]   (on the Mac; it runs this)
#
#   1. the Pupper reachable over ssh, and this checkout copied to it (tools/pup sync)
#   2. an Ollama of its own on this Mac (port 11435), with Gemma kept loaded; an Ollama app you
#      already run is left alone (they share the downloaded models)
#   3. whisper.cpp's server (speech to text, on the Mac's GPU, port 8178)
#   4. ./run_pupper.sh on the Pupper with the brain pointed here: the robot, Pupper's mic and
#      speaker, and robot/gemma_brain.py. Both servers reach the Pupper through the ssh
#      connection (tunnels), so the Mac and the Pupper need not be on the same network.
#
# Ctrl+C stops all of it, the robot first. Gemma is gemma4:26b once it is pulled
# (ollama pull gemma4:26b: about 17 GB, once); until then the best Gemma this Mac already has.
# The first run also downloads whisper's model (about 150 MB). Nothing goes to the cloud.
#   PUPPER_HOST=pi@<its IP> ./run_pupper.sh    when pupper5.local doesn't resolve
#   PUPPER_HOST=pi@4.tcp.us-cal-1.ngrok.io PUPPER_PORT=18196 ./run_pupper.sh   through an ngrok tunnel
#   PUPPER_GEMMA=gemma4:e4b ./run_pupper.sh    another model (pulled if missing)
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
PUP="$HERE/tools/pup"
WANT=gemma4:26b    # the best brain this Mac can run: a mixture of experts, 4B active per token
WHISPER_MODEL=${PUPPER_WHISPER_MODEL:-base.en}
OLLAMA_PORT=11435  # not 11434: the Ollama app may hold that, for this Mac only
WHISPER_PORT=8178  # not whisper.cpp's 8080, which a dev server on the Mac often has
CACHE=$HOME/.cache/pupper
LOG=$CACHE/logs
mkdir -p "$LOG"
die() { echo "!! $*" >&2; exit 1; }

PIDS=""         # the servers this script started (stopped at the end)
PI_STARTED=0
cleanup() {
  [ $PI_STARTED = 1 ] && { "$PUP" sh "~/pupper-hackathon/robot/stop.sh" >/dev/null 2>&1 \
    || echo "!! couldn't reach the Pupper to stop it: use the e-stop, then ./tools/pup sh pupper-hackathon/robot/stop.sh"; }
  [ -n "$PIDS" ] && kill $PIDS 2>/dev/null
  return 0
}
trap cleanup EXIT
trap 'echo; echo "stopping..."; exit 130' INT TERM
caffeinate -i -w $$ &  # the Mac must not sleep while it is Pupper's brain

# --- 1. the Pupper, first: nothing heavy starts if it can't be reached
"$PUP" sh true >/dev/null 2>&1 || die "can't reach ${PUPPER_HOST:-pi@pupper5.local} over ssh. Once per laptop: ./tools/pup setup.
   If pupper5.local doesn't resolve: PUPPER_HOST=pi@<its IP> ./run_pupper.sh, or through ngrok:
   PUPPER_HOST=pi@<ngrok host> PUPPER_PORT=<ngrok port> ./run_pupper.sh"
"$PUP" sync >/dev/null || die "copying this checkout to the Pupper failed (./tools/pup sync)"
echo "Pupper: ${PUPPER_HOST:-pi@pupper5.local}${PUPPER_PORT:+ port $PUPPER_PORT}, running this checkout (~/pupper-hackathon)"

# --- 2. Ollama of our own, on the network, with Gemma
ollama_up() { curl -sf -m 2 "http://127.0.0.1:$OLLAMA_PORT/api/version" >/dev/null; }
command -v ollama >/dev/null || die "no Ollama: brew install ollama (or ollama.com/download)"
if ! ollama_up; then
  # On the network; the context the brain asks for (so the eyes' requests share it and nothing
  # reloads); kept loaded; two at a time, so the eyes don't push the brain's prompt out of the cache.
  OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=2 \
    nohup ollama serve > "$LOG/ollama.log" 2>&1 &
  PIDS="$PIDS $!"
  for _ in $(seq 20); do ollama_up && break; sleep 1; done
  ollama_up || die "Ollama didn't come up on port $OLLAMA_PORT (log $LOG/ollama.log)"
fi
echo "Ollama: on port $OLLAMA_PORT (log $LOG/ollama.log)"
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT  # for the ollama commands below
MODEL=${PUPPER_GEMMA:-$WANT}
if ! ollama show "$MODEL" >/dev/null 2>&1; then
  # The default isn't pulled: the best Gemma already here (robot/gemma_brain.py's own choice), so a
  # slow network doesn't mean no brain; pull the default when the network allows.
  HAVE=$(cd "$HERE" && python3 -c 'import sys; sys.path.insert(0, "robot"); import gemma_brain as g; print(g.pick_model(sys.argv[1:]) or "")' \
         $(ollama list 2>/dev/null | awk 'NR > 1 {print $1}'))
  if [ -z "${PUPPER_GEMMA:-}" ] && [ -n "$HAVE" ]; then
    echo "Gemma: $WANT isn't pulled yet, so $HAVE for now (the better brain: ollama pull $WANT, about 17 GB)"
    MODEL=$HAVE
  else
    echo "pulling $MODEL (once)..."
    ollama pull "$MODEL" || die "couldn't pull $MODEL. Gemma 4 needs a recent Ollama: brew upgrade ollama, or update the app"
  fi
fi
echo "Gemma: $MODEL"

# --- 3. whisper.cpp's server, on the network
whisper_up() { curl -sf -m 2 -o /dev/null "http://127.0.0.1:$WHISPER_PORT/"; }
if ! whisper_up; then
  if ! command -v whisper-server >/dev/null; then
    command -v brew >/dev/null || die "no whisper-server and no Homebrew (brew.sh): install one, then brew install whisper.cpp"
    echo "installing whisper.cpp (brew install whisper.cpp, once)..."
    HOMEBREW_NO_AUTO_UPDATE=1 brew install whisper.cpp >/dev/null || die "brew install whisper.cpp failed"
    command -v whisper-server >/dev/null || die "Homebrew's whisper.cpp has no whisper-server: build it from github.com/ggml-org/whisper.cpp"
  fi
  WM=$CACHE/ggml-$WHISPER_MODEL.bin
  if [ ! -f "$WM" ]; then
    echo "downloading whisper's $WHISPER_MODEL model (once)..."
    # -C -: a download cut short (a network change) picks up where it stopped, on retry or next run
    curl -fL -C - --retry 5 --retry-delay 2 --progress-bar -o "$WM.part" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$WHISPER_MODEL.bin" \
      && mv "$WM.part" "$WM" || die "the whisper model download failed"
  fi
  nohup whisper-server -m "$WM" -l en --host 127.0.0.1 --port $WHISPER_PORT > "$LOG/whisper.log" 2>&1 &
  PIDS="$PIDS $!"
  for _ in $(seq 30); do whisper_up && break; sleep 1; done
  whisper_up || die "whisper-server didn't come up on port $WHISPER_PORT (log $LOG/whisper.log)"
fi
echo "whisper: on port $WHISPER_PORT ($WHISPER_MODEL, log $LOG/whisper.log)"

# --- 4. the Pupper: ./run_pupper.sh there, with the brain pointed here
echo
PI_STARTED=1
PUPPER_TUNNEL_PORTS="$OLLAMA_PORT $WHISPER_PORT" "$PUP" tty "cd ~/pupper-hackathon && PUPPER_OLLAMA=http://localhost:$OLLAMA_PORT PUPPER_WHISPER=http://localhost:$WHISPER_PORT PUPPER_GEMMA=$MODEL ./run_pupper.sh $*"
