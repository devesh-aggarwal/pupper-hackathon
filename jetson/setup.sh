#!/usr/bin/env bash
# Set up the Jetson (Thor or Orin Nano) as Pupper's on-device brain, once:   jetson/setup.sh [model]
#
#   1. the cable to the Pi: this end is 10.10.10.1 (jetson/link.sh jetson up)
#   2. Ollama, listening only on the cable (10.10.10.1:11434), with Gemma pulled and kept loaded
#   3. whisper.cpp's server on the GPU (speech to text), only on the cable (10.10.10.1:8080)
#
# It downloads things, so it needs the internet once (WiFi). After that nothing leaves the Jetson
# and the Pi. Safe to run again. The model: gemma4:26b on a Thor (32 GB of memory or more), else
# gemma4:e2b-it-qat (an 8 GB Orin Nano: e4b is 6.6 GB and doesn't fit beside whisper).
# The robot's eyes can use the same Gemma, so one model stays loaded:
#   python3 vision/eyes.py --robot http://10.10.10.2:8000 --ollama http://10.10.10.1:11434 --gemma <model>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ADDR=10.10.10.1
MEM_GB=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)
if [ "$MEM_GB" -ge 32 ]; then BIG=1; else BIG=0; fi
MODEL=${1:-$([ $BIG = 1 ] && echo gemma4:26b || echo gemma4:e2b-it-qat)}
WHISPER_MODEL=${WHISPER_MODEL:-$([ $BIG = 1 ] && echo small.en || echo base.en)}
echo "Jetson with ${MEM_GB} GB: Gemma $MODEL, whisper $WHISPER_MODEL"

echo "[1/3] the cable to the Pi"
"$HERE/link.sh" jetson up

echo "[2/3] Ollama with $MODEL"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh  # it knows Jetsons (JetPack)
sudo mkdir -p /etc/systemd/system/ollama.service.d
# Only on the cable's address; kept loaded; the context the brain asks for (robot/gemma_brain.py
# NUM_CTX), so the eyes' requests (which don't ask) get the same one and Ollama never reloads.
sudo tee /etc/systemd/system/ollama.service.d/pupper.conf >/dev/null <<EOF
[Service]
Environment="OLLAMA_HOST=$ADDR:11434"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_CONTEXT_LENGTH=16384"
EOF
sudo systemctl daemon-reload
sudo systemctl enable ollama >/dev/null 2>&1 || true
sudo systemctl restart ollama
for _ in $(seq 30); do curl -sf "http://$ADDR:11434/api/tags" >/dev/null && break; sleep 1; done
OLLAMA_HOST=$ADDR:11434 ollama pull "$MODEL"

echo "[3/3] whisper.cpp's server ($WHISPER_MODEL, on the GPU)"
command -v cmake >/dev/null && command -v git >/dev/null || sudo apt-get install -y git cmake build-essential
WDIR=$HOME/whisper.cpp
[ -d "$WDIR" ] || git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$WDIR"
export PATH=/usr/local/cuda/bin:$PATH
cmake -S "$WDIR" -B "$WDIR/build" -DGGML_CUDA=1 -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$WDIR/build" -j"$(nproc)" --target whisper-server
[ -f "$WDIR/models/ggml-$WHISPER_MODEL.bin" ] || bash "$WDIR/models/download-ggml-model.sh" "$WHISPER_MODEL"
sudo tee /etc/systemd/system/pupper-whisper.service >/dev/null <<EOF
[Unit]
Description=whisper.cpp server for Pupper (speech to text, on the cable to the Pi)
After=network-online.target

[Service]
User=$USER
ExecStart=$WDIR/build/bin/whisper-server -m $WDIR/models/ggml-$WHISPER_MODEL.bin -l en --host $ADDR --port 8080
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now pupper-whisper
sudo systemctl restart pupper-whisper

echo
echo "Done. On the Pi: jetson/setup_pi.sh, then python3 robot/gemma_brain.py --check"
