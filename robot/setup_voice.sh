#!/usr/bin/env bash
# Pupper's voice for the local brain (robot/gemma_brain.py), once, on the Pi:   robot/setup_voice.sh
# Piper (a neural text-to-speech voice, fast on the Pi 5) and its voice, and espeak-ng as a
# fallback. Downloads, so it needs the internet once. Moves nothing. Safe to run again.
set -euo pipefail
VOICE=en_US-lessac-medium
VOICES=$HOME/.local/share/piper
URL=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium

echo "setting up Pupper's voice (Piper, $VOICE)..."
command -v espeak-ng >/dev/null || sudo apt-get install -y espeak-ng
python3 -c "import piper" 2>/dev/null || python3 -m pip install --user --break-system-packages -q piper-tts
mkdir -p "$VOICES"
for f in "$VOICE.onnx" "$VOICE.onnx.json"; do
  [ -f "$VOICES/$f" ] || curl -fsSL -o "$VOICES/$f" "$URL/$f"
done
if echo "hello, I am Pupper" | "$HOME/.local/bin/piper" -m "$VOICES/$VOICE.onnx" -f /tmp/pupper-hello.wav >/dev/null 2>&1; then
  echo "voice: Piper works"
else
  echo "voice: Piper failed, so Pupper will speak with espeak-ng"
fi
