#!/bin/bash
# One-time setup of Gemini Pupper on the robot. Safe to run again (e.g. after a git pull
# or after editing the web app: it rebuilds web/dist/).
#   robot/setup.sh    Node in ~/.local/node (no sudo), the web app build, the robot
#                     server's Python packages, and the Gemini API key file
# Then start with robot/start.sh --kiosk.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
NODE_VERSION=v22.23.3  # Debian's nodejs (18) is too old for the build tools
NODE_DIR=$HOME/.local/node
KEY_FILE=${GEMINI_KEY_FILE:-$HOME/.config/gemini-pupper/api_key}

node_major() { "$NODE_DIR/bin/node" -v 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/'; }

echo "[1/4] Node.js"
if [ "$(node_major || echo 0)" -ge 20 ] 2>/dev/null; then
  echo "      $("$NODE_DIR/bin/node" -v) already in $NODE_DIR"
else
  arch=$(uname -m); [ "$arch" = aarch64 ] && arch=arm64; [ "$arch" = x86_64 ] && arch=x64
  echo "      downloading Node $NODE_VERSION ($arch) into $NODE_DIR"
  tmp=$(mktemp -d)
  curl -fL --progress-bar -o "$tmp/node.tar.xz" "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-$arch.tar.xz"
  rm -rf "$NODE_DIR"; mkdir -p "$NODE_DIR"
  tar -xJf "$tmp/node.tar.xz" -C "$NODE_DIR" --strip-components=1; rm -rf "$tmp"
fi
export PATH=$NODE_DIR/bin:$PATH
if ! grep -qs '.local/node/bin' "$HOME/.bashrc"; then
  echo 'export PATH=$HOME/.local/node/bin:$PATH  # Node for Gemini Pupper' >> "$HOME/.bashrc"
  echo "      added Node to PATH in ~/.bashrc (open a new terminal to use npm yourself)"
fi

echo "[2/4] building the web app (npm ci && npm run build)"
cd "$REPO/web"
if grep -qs '^GEMINI_API_KEY=..' .env.local; then
  echo "      note: .env.local has a GEMINI_API_KEY, so the build embeds it in web/dist/. Never share it."
fi
npm ci --no-audit --no-fund --loglevel=error
npm run build --silent 2>&1 | grep -vE "chunks are larger|dynamic import|manualChunks|chunkSizeWarningLimit|^$" || true
[ -f dist/index.html ] || { echo "build failed: no dist/index.html"; exit 1; }
echo "      built web/dist/"

echo "[3/4] Python packages for the robot server"
if python3 -c "import fastapi, uvicorn, psutil, pydantic, websockets, google.genai" 2>/dev/null; then
  echo "      already installed"
else
  python3 -m pip install --user --break-system-packages -q fastapi uvicorn psutil pydantic websockets google-genai
  python3 -c "import fastapi, uvicorn, psutil, pydantic, websockets, google.genai"
  echo "      installed"
fi

echo "[4/4] Gemini API key ($KEY_FILE)"
if [ -s "$KEY_FILE" ]; then
  echo "      already there"
elif [ -t 0 ]; then
  read -rsp "      paste your Gemini API key (hidden), then Enter: " key; echo
  [ -n "$key" ] || { echo "      no key given; kiosk mode needs it. Run setup again later."; exit 0; }
  mkdir -p "$(dirname "$KEY_FILE")"; (umask 077; printf '%s\n' "$key" > "$KEY_FILE")
  echo "      saved (readable only by you)"
else
  echo "      missing. Create it with:  mkdir -p $(dirname "$KEY_FILE") && (umask 077; cat > $KEY_FILE)"
fi

echo "done. Start with:  robot/start.sh --kiosk     (stop with robot/stop.sh)"
