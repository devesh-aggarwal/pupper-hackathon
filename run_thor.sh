#!/bin/bash
# Pupper with Gemma on the Thor, on the Mac:   ./run_thor.sh [volume 0-100] [--laptop]
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
[ "$(uname)" = Darwin ] || { echo "run this on the Mac (the Pupper runs ./run_pupper.sh, which the Mac starts)"; exit 2; }
case "${1:-}" in -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

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
