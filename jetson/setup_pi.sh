#!/usr/bin/env bash
# Set up the Pi's side of the local brain on a Jetson, once:   jetson/setup_pi.sh
#
#   1. the cable to the Jetson: this end is 10.10.10.2 (jetson/link.sh pi up)
#   2. Pupper's voice: Piper, and espeak-ng as a fallback (robot/setup_voice.sh)
#
# It downloads things, so it needs the internet once (WiFi). Moves nothing. Safe to run again.
# Then: python3 robot/gemma_brain.py --check
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

echo "[1/2] the cable to the Jetson"
"$HERE/link.sh" pi up

echo "[2/2] Pupper's voice"
"$HERE/../robot/setup_voice.sh"

echo
echo "Done. Next: python3 robot/gemma_brain.py --check"
