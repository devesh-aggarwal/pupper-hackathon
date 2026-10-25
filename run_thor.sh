#!/bin/bash
# Pupper with Gemma on the Thor:   ./run_thor.sh [volume 0-100] [--laptop]
#
# On the Pupper (on ThorNet): everything runs between the Pupper and the Thor, no Mac. On a Mac: as before.
#
# The same as ./run_pupper.sh on the Mac, except that Gemma (the brain, and the eyes' questions) is
# the Thor's: Gemma 4 31B in vLLM on the team's Jetson Thor, not Ollama on this Mac. whisper still
# runs here, and the Pupper's side doesn't change: tools/thor_relay.py answers on port 11435 in
# Ollama's terms, and the link makes it localhost on the Pupper, as with the Mac's Gemma.
#
# This Mac must reach both the Thor and the Pupper. The Thor is found at $PUPPER_THOR if set, else
# on thornet (http://10.42.0.1:8000), else on LatentSpace (http://192.168.2.5:8000).
#
# The Thor's vLLM needs tool calling on (--enable-auto-tool-choice --tool-call-parser gemma4); it
# is checked first, and if it is off this says how to turn it on. Back to the Mac's Gemma:
# ./run_pupper.sh (it swaps the relay back by itself).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
case "${1:-}" in -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

# On the Pupper (both on the Thor's network, thornet): no Mac at all. The Thor's relay runs here,
# on port 11436 (not 11435, which a Mac's link may hold), and the brain and the eyes use it.
# Speech to text must be on the Thor too: whisper.cpp's server on :8178, or vLLM serving whisper
# on :8001 (PUPPER_WHISPER=<url> says where else).
if [ "$(uname)" = Linux ]; then
  THOR=${PUPPER_THOR:-http://10.42.0.1:8000}
  HOST=$(echo "$THOR" | sed -E 's#^(https?://[^:/]+).*#\1#')
  curl -sf -m 3 -o /dev/null "$THOR/v1/models" || { echo "!! no Thor at $THOR: is the Pupper on its WiFi (ThorNet)? Or PUPPER_THOR=http://<address>:8000 ./run_thor.sh"; exit 1; }
  if [ -z "${PUPPER_WHISPER:-}" ]; then
    if curl -sf -m 3 -o /dev/null "$HOST:8178/"; then PUPPER_WHISPER=$HOST:8178
    elif curl -sf -m 3 -o /dev/null "$HOST:8001/v1/models"; then PUPPER_WHISPER=$HOST:8001/v1
    else
      echo "!! no speech to text on the Thor (nothing on $HOST:8178 or $HOST:8001), so Pupper can't hear."
      echo "   On the Thor, start one of: whisper.cpp's whisper-server -m <ggml model> -l en --convert --host 0.0.0.0 --port 8178"
      echo "                          or: vllm serve openai/whisper-large-v3-turbo --port 8001 --gpu-memory-utilization 0.1"
      echo "   Or say where it is: PUPPER_WHISPER=http://<address>:<port> ./run_thor.sh"
      exit 1
    fi
  fi
  python3 "$HERE/tools/thor_relay.py" --thor "$THOR" --check || exit 1
  PORT=11436
  pkill -f "[t]ools/thor_relay.py --port $PORT" 2>/dev/null
  mkdir -p /tmp/gemini-pupper
  setsid nohup python3 -u "$HERE/tools/thor_relay.py" --port $PORT --thor "$THOR" > /tmp/gemini-pupper/thor_relay.log 2>&1 < /dev/null &
  for _ in $(seq 20); do curl -sf -m 1 -o /dev/null "http://localhost:$PORT/relay" && break; sleep 0.5; done
  echo "Gemma: the Thor ($THOR) through tools/thor_relay.py on :$PORT (log /tmp/gemini-pupper/thor_relay.log); whisper: $PUPPER_WHISPER"
  export PUPPER_OLLAMA=http://localhost:$PORT PUPPER_WHISPER
  exec "$HERE/run_pupper.sh" "$@"
fi

thor_up() { curl -sf -m 3 -o /dev/null "$1/health"; }
if [ -n "${PUPPER_THOR:-}" ]; then
  thor_up "$PUPPER_THOR" || { echo "!! the Thor doesn't answer at $PUPPER_THOR/health"; exit 1; }
else
  for url in http://10.42.0.1:8000 http://192.168.2.5:8000; do
    thor_up "$url" && { PUPPER_THOR=$url; break; }
  done
  [ -n "${PUPPER_THOR:-}" ] || {
    echo "!! no Thor at http://10.42.0.1:8000 (thornet) or http://192.168.2.5:8000 (LatentSpace)."
    echo "   Is this Mac on the Thor's network? Or: PUPPER_THOR=http://<address>:8000 ./run_thor.sh"
    exit 1; }
fi
export PUPPER_THOR
exec "$HERE/tools/brain_on_mac.sh" "$@"
