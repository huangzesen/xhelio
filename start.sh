#!/usr/bin/env bash
# Launch backend + frontend together.
# Usage: ./start.sh          — start the app
#        ./start.sh install   — install the 'xhelio' CLI command
#        ./start.sh uninstall — remove the 'xhelio' CLI command
#
# On first run this creates a Python venv, installs pip and npm
# dependencies, and checks for a .env file.  Subsequent runs skip
# these steps automatically.

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Install/uninstall the 'xhelio' CLI command ----
if [ "${1:-}" = "install" ]; then
  TARGET="/usr/local/bin/xhelio"
  echo "Installing 'xhelio' command -> $TARGET"
  cat > /tmp/xhelio-launcher <<LAUNCHER
#!/usr/bin/env bash
exec "$DIR/start.sh" "\$@"
LAUNCHER
  chmod +x /tmp/xhelio-launcher
  if [ -w "$(dirname "$TARGET")" ]; then
    mv /tmp/xhelio-launcher "$TARGET"
  else
    sudo mv /tmp/xhelio-launcher "$TARGET"
  fi
  echo "Done! You can now run 'xhelio' from anywhere."
  exit 0
fi

if [ "${1:-}" = "uninstall" ]; then
  TARGET="/usr/local/bin/xhelio"
  if [ -f "$TARGET" ]; then
    echo "Removing $TARGET ..."
    if [ -w "$TARGET" ]; then
      rm "$TARGET"
    else
      sudo rm "$TARGET"
    fi
    echo "Done."
  else
    echo "'xhelio' command not found at $TARGET"
  fi
  exit 0
fi

# ---- Python venv + deps ----
if [ ! -d "$DIR/venv" ]; then
  echo "Creating Python virtual environment ..."
  python3 -m venv "$DIR/venv"
fi

if [ ! -f "$DIR/venv/.deps_installed" ]; then
  echo "Installing Python dependencies ..."
  "$DIR/venv/bin/pip" install --upgrade pip -q
  "$DIR/venv/bin/pip" install -r "$DIR/requirements.txt" -q
  touch "$DIR/venv/.deps_installed"
fi

# ---- .env check ----
if [ ! -f "$DIR/.env" ]; then
  echo ""
  echo "No .env file found. Creating one — you need at least one API key."
  echo ""
  echo "Which LLM provider will you use?"
  echo "  1. Gemini  (default)"
  echo "  2. OpenAI / OpenAI-compatible"
  echo "  3. Anthropic Claude"
  echo ""
  read -rp "Provider [1]: " provider_choice
  provider_choice="${provider_choice:-1}"
  case "$provider_choice" in
    2) env_var="OPENAI_API_KEY";;
    3) env_var="ANTHROPIC_API_KEY";;
    *) env_var="GOOGLE_API_KEY";;
  esac
  read -rp "Enter your API key ($env_var): " api_key
  echo "$env_var=$api_key" > "$DIR/.env"
  echo "Saved to .env"
  echo ""
fi

# ---- Frontend deps ----
if [ ! -d "$DIR/frontend/node_modules" ]; then
  echo "Installing frontend dependencies ..."
  npm --prefix "$DIR/frontend" install
fi

# ---- Launch ----

# Kill any stale server on :8000
lsof -ti :8000 | xargs kill 2>/dev/null || true

# Start FastAPI backend in background
echo "Starting FastAPI backend on :8000 ..."
"$DIR/venv/bin/python" "$DIR/api_server.py" &
BACKEND_PID=$!

# Start Vite frontend dev server
echo "Starting React frontend on :5173 ..."
npm --prefix "$DIR/frontend" run dev &
FRONTEND_PID=$!

# Open browser after a short delay (macOS/Linux)
(sleep 2 && {
  if command -v open &>/dev/null; then
    open http://localhost:5173
  elif command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:5173
  fi
}) &

# Clean up both on exit
cleanup() {
  echo ""
  echo "Shutting down ..."
  kill $FRONTEND_PID $BACKEND_PID 2>/dev/null
}
trap cleanup EXIT INT TERM

wait
