#!/usr/bin/env bash
# Set up the Pi's side of the local brain, once:   jetson/setup_pi.sh
#
#   1. the cable to the Jetson: this end is 10.10.10.2 (jetson/link.sh pi up)
#   2. Pupper's voice: Piper (a neural text-to-speech voice, fast on the Pi 5), and espeak-ng
#      as a fallback
#
# It downloads things, so it needs the internet once (WiFi). Moves nothing. Safe to run again.
# Then: python3 robot/gemma_brain.py --check
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VOICE=en_US-lessac-medium
VOICES=$HOME/.local/share/piper
URL=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium

echo "[1/2] the cable to the Jetson"
"$HERE/link.sh" pi up

echo "[2/2] Pupper's voice"
command -v espeak-ng >/dev/null || sudo apt-get install -y espeak-ng
python3 -c "import piper" 2>/dev/null || python3 -m pip install --user --break-system-packages piper-tts
mkdir -p "$VOICES"
for f in "$VOICE.onnx" "$VOICE.onnx.json"; do
  [ -f "$VOICES/$f" ] || curl -fL -o "$VOICES/$f" "$URL/$f"
done
echo "hello, I am Pupper" | "$HOME/.local/bin/piper" -m "$VOICES/$VOICE.onnx" -f /tmp/pupper-hello.wav >/dev/null 2>&1 \
  && aplay -q /tmp/pupper-hello.wav && echo "Piper works" || echo "Piper failed: the brain will use espeak-ng"

echo
echo "Done. Next: python3 robot/gemma_brain.py --check"
