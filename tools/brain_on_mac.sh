#!/usr/bin/env bash
# Pupper's brain on this Mac, no cloud:   ./run_pupper.sh [volume 0-100]   (on the Mac; it runs this)
#
#   1. finds the Pupper over ssh (the address saved in ~/.config/pupper/ssh; by default Tailscale,
#      pi@100.88.25.110) and prints the checkout it will run (updates are explicit: git pull there)
#   2. starts the brain's servers on this Mac, if they aren't up: Ollama with Gemma (port 11437),
#      behind tools/ollama_relay.py (port 11435, takes gzipped requests), and whisper.cpp (speech to
#      text, port 8178, takes Opus), for this Mac only
#   3. starts the link: an ssh connection to the Pupper, kept up in the background, that makes
#      ports 11435 and 8178 localhost on the Pupper too (the Pupper can't reach this Mac by itself)
#   4. runs ./run_pupper.sh on the Pupper: the robot, its face, its eyes, its mic and speaker, the brain
#
# The Pupper's WiFi can send very slowly (a public hotspot: ~11 KB/s up), so what crosses the link
# is kept small: the brain gzips its requests, speech goes as Opus, and the eyes stay on the Pupper
# (no camera frames cross), sending Gemma 512 px images.
#
# Ctrl+C stops the robot. The servers and the link keep running, so ./run_pupper.sh on the Pupper
# itself works too, until the Mac sleeps or restarts.   tools/brain_on_mac.sh stop   ends them.
# Gemma is gemma4:26b (pulled the first time if it isn't here: about 17 GB). The first run also downloads whisper's model (about 550 MB).
# With PUPPER_THOR set (./run_thor.sh sets it), Gemma is the Thor's instead: no Ollama here, and
# tools/thor_relay.py stands on port 11435 in the relay's place, so the Pupper's side is the same.
# While the link is up, http://localhost:8000 on this Mac is Pupper's app (its face and debug view).
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
PUP="$HERE/tools/pup"
WANT=gemma4:26b  # the only model: the brain and the eyes
WHISPER_MODEL=${PUPPER_WHISPER_MODEL:-large-v3-turbo-q5_0}  # far better than base.en in a noisy room; fast on the GPU
OLLAMA_PORT=11435      # the relay: what the Pupper reaches (not 11434: the Ollama app may hold that)
OLLAMA_INTERNAL=11437  # Ollama itself, behind the relay
WHISPER_PORT=8178
WHISPER_ARGS="-l en -sns --convert"  # English, no tokens for non-speech, and Opus in (converted with ffmpeg)
CACHE=$HOME/.cache/pupper
LOG=$CACHE/logs
SAVED=$HOME/.config/pupper/ssh
mkdir -p "$LOG" "$(dirname "$SAVED")"
die() { echo "!! $*" >&2; exit 1; }
alive() { [ -f "$CACHE/$1.pid" ] && kill -0 "$(cat "$CACHE/$1.pid")" 2>/dev/null; }

if [ "${1:-}" = stop ]; then
  for name in eyes web link whisper relay ollama; do alive $name && kill "$(cat "$CACHE/$name.pid")" && echo "stopped $name"; rm -f "$CACHE/$name.pid"; done
  pkill -f "ssh .*-R $OLLAMA_PORT:localhost:$OLLAMA_PORT" 2>/dev/null
  exit 0
fi

# --- 1. the Pupper
reach() { "$PUP" sh true >/dev/null 2>&1; }
if ! reach; then
  [ -t 0 ] || die "can't reach the Pupper over ssh ($(cat "$SAVED" 2>/dev/null || echo pi@100.88.25.110)). Is Tailscale up on both?"
  echo "Can't reach the Pupper at $(cat "$SAVED" 2>/dev/null || echo pi@100.88.25.110). Is Tailscale up on both?"
  read -rp "Or paste the ssh command you use to reach it (e.g. ssh pi@100.88.25.110): " cmd
  port=$(echo "$cmd" | grep -oE -- '-p *[0-9]+' | grep -oE '[0-9]+'); host=$(echo "$cmd" | grep -oE '[^ ]+@[^ ]+' | head -1)
  [ -n "$host" ] || die "no user@host in that"
  echo "$host ${port:-22}" > "$SAVED"
  reach || die "still can't reach $host (port ${port:-22}). Once per laptop: ./tools/pup setup"
fi
echo "Pupper: $(cat "$SAVED" 2>/dev/null || echo pi@100.88.25.110)"
# Do not silently update or change the teammate's chosen branch during startup.
"$PUP" sh 'cd ~/pupper-hackathon && git status --short --branch && git log -1 --oneline' \
  || die "couldn't inspect ~/pupper-hackathon on the Pupper"

# --- 2. the brain's servers (they outlive this script: the Pupper may start on its own later)
export OLLAMA_HOST=127.0.0.1:$OLLAMA_INTERNAL  # this Mac only: the relay and the link carry it to the Pupper
THOR=${PUPPER_THOR:-}
ollama_up() { curl -sf -m 2 "http://$OLLAMA_HOST/api/version" >/dev/null; }
# Which relay is on the port: the Thor's says so (and which Thor); Ollama's doesn't.
relay_is() { curl -sf -m 2 "http://127.0.0.1:$OLLAMA_PORT/relay" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("thor") or "ollama")' 2>/dev/null; }
relay_up() { [ "$(relay_is)" = "${THOR:-ollama}" ]; }
if [ -z "$(relay_is)" ] && curl -sf -m 2 "http://127.0.0.1:$OLLAMA_PORT/api/version" >/dev/null; then  # an older setup: Ollama itself here
  alive ollama && kill "$(cat "$CACHE/ollama.pid")"; sleep 2
fi
if [ -n "$(relay_is)" ] && ! relay_up; then  # the other brain's relay (the Mac's Ollama, or another Thor): swap it
  alive relay && kill "$(cat "$CACHE/relay.pid")"; pkill -f "tools/(ollama|thor)_relay.py" 2>/dev/null; sleep 1
fi
if [ -n "$THOR" ]; then
  echo "Gemma: the Thor at $THOR"
  python3 "$HERE/tools/thor_relay.py" --thor "$THOR" --check \
    || die "the Thor isn't ready to be Pupper's brain (above). Is this Mac on the Thor's network?"
elif ! command -v ollama >/dev/null; then
  die "no Ollama: brew install ollama (or ollama.com/download)"
elif ! ollama_up; then
  # The context the brain asks for (so the eyes' requests share it and nothing reloads), kept
  # loaded, two at a time (the eyes don't push the brain's prompt out of the cache).
  OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=2 nohup ollama serve > "$LOG/ollama.log" 2>&1 &
  echo $! > "$CACHE/ollama.pid"
  for _ in $(seq 20); do ollama_up && break; sleep 1; done
  ollama_up || die "Ollama didn't start (log $LOG/ollama.log)"
  nohup caffeinate -i -w "$(cat "$CACHE/ollama.pid")" >/dev/null 2>&1 &  # no sleeping while it's the brain
fi
MODEL=$WANT
if [ -z "$THOR" ] && ! ollama show "$WANT" >/dev/null 2>&1; then
  echo "pulling $WANT (once, about 17 GB)..."
  ollama pull "$WANT" || die "couldn't pull $WANT. Gemma 4 needs a recent Ollama: brew upgrade ollama, or update the app"
fi
echo "Gemma: $MODEL (the brain and eyes use this same model)"

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
if ! relay_up; then
  if [ -n "$THOR" ]; then
    nohup python3 -u "$HERE/tools/thor_relay.py" --port $OLLAMA_PORT --thor "$THOR" > "$LOG/relay.log" 2>&1 &
    nohup caffeinate -i -w $! >/dev/null 2>&1 &  # no sleeping while this Mac carries the brain
  else
    nohup python3 -u "$HERE/tools/ollama_relay.py" --port $OLLAMA_PORT --ollama "http://$OLLAMA_HOST" > "$LOG/relay.log" 2>&1 &
  fi
  echo $! > "$CACHE/relay.pid"
  for _ in $(seq 10); do relay_up && break; sleep 0.5; done
  relay_up || die "the Ollama relay didn't start (log $LOG/relay.log)"
fi
if [ -n "$THOR" ]; then
  echo "brain: the Thor ($THOR) behind tools/thor_relay.py :$OLLAMA_PORT, whisper :$WHISPER_PORT ($WHISPER_MODEL) (logs in $LOG)"
else
  echo "brain: Ollama :$OLLAMA_INTERNAL behind the relay :$OLLAMA_PORT, whisper :$WHISPER_PORT ($WHISPER_MODEL) (logs in $LOG)"
fi

# --- 3. the link: the two ports on the Pupper, through ssh, reconnecting whenever it drops
pupper_sees_brain() { "$PUP" sh "curl -sf -m 3 localhost:$OLLAMA_PORT/api/version >/dev/null && curl -sf -m 3 -o /dev/null localhost:$WHISPER_PORT/" >/dev/null 2>&1; }
PORTS="$OLLAMA_PORT $WHISPER_PORT"
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
# The eyes run on the Pupper (robot/start.sh): its camera is there, and frames are too big for its
# WiFi to send here. Eyes an older run started on this Mac are stopped.
if alive eyes; then kill "$(cat "$CACHE/eyes.pid")"; pkill -f "venv/bin/python -u vision/eyes.py"; rm -f "$CACHE/eyes.pid"; fi
if [ -n "$THOR" ]; then
  echo "watch: http://localhost:8000 (Pupper's app), tail -f $LOG/relay.log (each Gemma request, with its time)"
else
  echo "watch: http://localhost:8000 (Pupper's app), tail -f $LOG/ollama.log (each Gemma request),"
  echo "       OLLAMA_HOST=127.0.0.1:$OLLAMA_INTERNAL ollama ps (the model on the GPU)"
fi

# --- 4. the Pupper
echo
run_args=""
if [ "$#" -gt 0 ]; then printf -v run_args ' %q' "$@"; fi
"$PUP" tty "cd ~/pupper-hackathon && ./run_pupper.sh$run_args"
"$PUP" sh "~/pupper-hackathon/robot/stop.sh" >/dev/null 2>&1 \
  || echo "!! couldn't reach the Pupper to stop it: use the e-stop, then ./tools/pup sh pupper-hackathon/robot/stop.sh"
echo "The brain and the link keep running: ./run_pupper.sh on the Pupper works too. To end them: tools/brain_on_mac.sh stop"
