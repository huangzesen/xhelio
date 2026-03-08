#!/bin/bash
# scripts/run-live-e2e.sh — Run all live E2E tests and generate report
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_DIR/frontend"
REPORT_DIR="$PROJECT_DIR/docs/reports"
DATE=$(date +%Y%m%d)

echo "=== xhelio Live E2E Test Suite ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Check backend
if ! curl -s http://localhost:8000/api/models > /dev/null 2>&1; then
    echo "ERROR: Backend not running on port 8000."
    echo "Start with: venv/bin/python api_server.py --port 8000"
    exit 1
fi
echo "Backend: OK (port 8000)"

# Check frontend (playwright will start it if needed)
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "Frontend: OK (port 5173)"
else
    echo "Frontend: Not running (Playwright will start it)"
fi

# Create output dirs
mkdir -p "$FRONTEND_DIR/e2e/screenshots"
mkdir -p "$REPORT_DIR"

# Run tests
cd "$FRONTEND_DIR"
echo ""
echo "Running live E2E tests..."
echo "This will take 30-60 minutes (LLM response times + data fetches)."
echo ""

XHELIO_E2E_REAL=1 npx playwright test e2e/tests/e2e/ \
    --timeout=300000 \
    --reporter=line \
    2>&1 | tee "$REPORT_DIR/e2e-output-${DATE}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=== Test Complete ==="
echo "Exit code: $EXIT_CODE"
echo "Console log: $REPORT_DIR/e2e-output-${DATE}.log"
echo "Screenshots: $FRONTEND_DIR/e2e/screenshots/"
echo ""
echo "To view HTML report:"
echo "  cd $FRONTEND_DIR && npx playwright show-report"

# Extract issues from log
echo ""
echo "=== Issues Found ==="
grep '\[ISSUE\]' "$REPORT_DIR/e2e-output-${DATE}.log" 2>/dev/null || echo "No issues logged."

exit $EXIT_CODE
