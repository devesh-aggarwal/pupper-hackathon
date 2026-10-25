#!/bin/bash
# One-time setup of Gemini Pupper on the robot. Safe to run again (e.g. after a git pull
# or after editing the web app: it rebuilds web/dist/).
#   robot/setup.sh    Node in ~/.local/node (no sudo), the web app build, the robot
#                     server's Python packages. No cloud key is needed on this branch.
# Then start with ./run_pupper.sh (the Mac brain must be reachable).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
NODE_VERSION=v22.23.3  # Debian's nodejs (18) is too old for the build tools
NODE_DIR=$HOME/.local/node

node_major() { "$NODE_DIR/bin/node" -v 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/'; }

echo "[1/3] Node.js"
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

echo "[2/3] building the web app (npm ci && npm run build)"
cd "$REPO/web"
npm ci --no-audit --no-fund --loglevel=error
npm run build --silent
[ -f dist/index.html ] || { echo "build failed: no dist/index.html"; exit 1; }
echo "      built web/dist/"

echo "[3/3] Python packages for the robot server"
if python3 -c "import fastapi, uvicorn, psutil, pydantic, websockets, yaml" 2>/dev/null; then
  echo "      already installed"
else
  python3 -m pip install --user --break-system-packages -q fastapi uvicorn psutil pydantic websockets pyyaml
  python3 -c "import fastapi, uvicorn, psutil, pydantic, websockets, yaml"
  echo "      installed"
fi

echo "done. No Gemini API key is needed. Start with ./run_pupper.sh after the Mac brain is linked."
