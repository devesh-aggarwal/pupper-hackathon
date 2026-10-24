#!/usr/bin/env bash
# Run Gemini Pupper on your laptop, no Pupper needed:  ./run_local.sh
# The real robot server, lab/ tools and prompt, reminders and web app, with a fake robot
# (every motion is logged, nothing moves). Open http://localhost:8000, paste your Gemini
# key in Settings, press WAKE PUPPER, and talk through your laptop's mic.
#   PORT=8080 ./run_local.sh     another port
#   ./run_local.sh --build       rebuild the web app even if it looks up to date
set -euo pipefail
cd "$(dirname "$0")"
PY_PKGS="fastapi uvicorn psutil pydantic pyyaml"

# NixOS (or anywhere without python3/npm): rerun inside a nix-shell that has them.
if { ! command -v python3 >/dev/null || ! command -v npm >/dev/null; } && [ -z "${PUPPER_IN_NIX:-}" ] \
   && command -v nix-shell >/dev/null; then
  echo "python3 or npm missing: running inside nix-shell (the first time downloads them)"
  exec env PUPPER_IN_NIX=1 nix-shell -p "python3.withPackages(p: with p; [fastapi uvicorn psutil pydantic pyyaml])" \
    nodejs_22 --run "$(printf '%q ' "$0" "$@")"
fi
command -v python3 >/dev/null || { echo "Needs python3"; exit 1; }
if ! python3 -c "import fastapi, uvicorn, psutil, pydantic, yaml" 2>/dev/null; then
  echo "Missing Python packages. Install them with:"
  echo "  python3 -m pip install --user $PY_PKGS"
  exit 1
fi

# Build the web app if it's missing, older than its sources, or asked for.
stale=$( [ -f web/dist/index.html ] && find web -path web/node_modules -prune -o -path web/dist -prune \
         -o -type f -newer web/dist/index.html -print -quit || echo missing)
if [ -n "$stale" ] || [ "${1:-}" = "--build" ]; then
  command -v npm >/dev/null || { echo "Needs Node.js (npm) to build the web app"; exit 1; }
  echo "building the web app (web/dist)..."
  (cd web && { [ -d node_modules ] || npm ci --no-audit --no-fund --loglevel=error; } && npm run build --silent \
     2>&1 | grep -vE "chunks are larger|dynamic import|manualChunks|chunkSizeWarningLimit|^$" || true)
  [ -f web/dist/index.html ] || { echo "web build failed"; exit 1; }
fi

exec python3 -u robot/laptop_server.py
