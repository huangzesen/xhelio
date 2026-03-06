#!/usr/bin/env bash
set -euo pipefail

# Delegate a task from docs/tasks/ to a coding agent in a git worktree.
#
# Usage: scripts/delegate.sh <slug> <agent> [--dry-run]
#   agent: opencode | gemini
#
# What it does (in one shot):
#   1. Finds the task file (_pending_ or _taken_)
#   2. Claims it (renames _pending_ → _taken_, updates status fields)
#   3. Creates a git worktree + branch (or reuses existing)
#   4. Symlinks shared resources (frontend/node_modules)
#   5. Writes .task-prompt.md (task content wrapped in instructions)
#   6. Launches the agent in background, captures PID
#   7. Registers in docs/tasks/logs/delegations.tsv
#   8. Prints a summary line
#
# The task file IS the spec — no rewriting needed.

if [ $# -lt 2 ]; then
    echo "Usage: scripts/delegate.sh <slug> <agent> [--dry-run]"
    echo "  agent: opencode | gemini"
    exit 1
fi

SLUG="$1"
AGENT="$2"
DRY_RUN=false
if [ "${3:-}" = "--dry-run" ]; then
    DRY_RUN=true
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
MAIN_REPO=$(git worktree list | head -1 | awk '{print $1}')

# Agent-specific config
case "$AGENT" in
    opencode)
        PREFIX="oc"
        BRANCH_PREFIX="opencode"
        AGENT_NAME="OpenCode"
        LOG_SUFFIX="opencode"
        ;;
    gemini)
        PREFIX="gm"
        BRANCH_PREFIX="gemini"
        AGENT_NAME="Gemini CLI"
        LOG_SUFFIX="gemini"
        ;;
    *)
        echo "ERROR: Unknown agent: $AGENT (expected: opencode | gemini)"
        exit 1
        ;;
esac

BRANCH="$BRANCH_PREFIX/$SLUG"
WORKTREE_DIR="$REPO_ROOT/../xhelio-$PREFIX-$SLUG"
LOG_DIR="$REPO_ROOT/docs/tasks/logs"
LOG_FILE="$LOG_DIR/$SLUG.$LOG_SUFFIX.log"

# ── 1. Find and claim task ────────────────────────────────────────────

# Check if already done
if [ -f "$REPO_ROOT/docs/tasks/_done_${SLUG}.md" ]; then
    echo "SKIP: $SLUG already done"
    exit 0
fi

# Find pending or taken
TASK_FILE=""
for state in pending taken; do
    f="$REPO_ROOT/docs/tasks/_${state}_${SLUG}.md"
    if [ -f "$f" ]; then
        TASK_FILE="$f"
        break
    fi
done

if [ -z "$TASK_FILE" ]; then
    echo "ERROR: no task found for '$SLUG' in docs/tasks/"
    exit 1
fi

# Claim if pending (rename + update status fields)
if [[ "$TASK_FILE" == *"_pending_"* ]]; then
    NEW_FILE="$REPO_ROOT/docs/tasks/_taken_${SLUG}.md"
    if $DRY_RUN; then
        echo "[dry-run] Would rename: $(basename "$TASK_FILE") → $(basename "$NEW_FILE")"
    else
        mv "$TASK_FILE" "$NEW_FILE"
        # Update status fields using Python (safe with ** markdown — BSD sed breaks on *)
        python3 -c "
import re, sys
path = sys.argv[1]
agent_name = sys.argv[2]
text = open(path).read()
text = re.sub(r'\*\*Status:\*\*\s*\w+', '**Status:** taken', text, count=1)
text = re.sub(r'\*\*Taken by:\*\*.*', f'**Taken by:** {agent_name}', text, count=1)
open(path, 'w').write(text)
" "$NEW_FILE" "$AGENT_NAME"
        TASK_FILE="$NEW_FILE"
    fi
elif $DRY_RUN; then
    echo "[dry-run] Task already taken: $(basename "$TASK_FILE")"
fi

# ── 2. Create worktree (or reuse) ────────────────────────────────────

if [ -d "$WORKTREE_DIR" ]; then
    if $DRY_RUN; then
        echo "[dry-run] Worktree already exists: $WORKTREE_DIR"
    fi
else
    if $DRY_RUN; then
        echo "[dry-run] Would create worktree: $WORKTREE_DIR (branch: $BRANCH)"
    else
        cd "$REPO_ROOT"
        git worktree add "$WORKTREE_DIR" -b "$BRANCH" 2>/dev/null \
            || git worktree add "$WORKTREE_DIR" "$BRANCH" 2>/dev/null \
            || { echo "ERROR: failed to create worktree at $WORKTREE_DIR"; exit 1; }
    fi
fi

# ── 3. Symlink shared resources ──────────────────────────────────────

if ! $DRY_RUN; then
    ln -sf "$MAIN_REPO/frontend/node_modules" "$WORKTREE_DIR/frontend/node_modules" 2>/dev/null || true
fi

# ── 4. Write prompt file ─────────────────────────────────────────────

PROMPT_FILE="$WORKTREE_DIR/.task-prompt.md"

if $DRY_RUN; then
    echo "[dry-run] Would write prompt to: $PROMPT_FILE"
    echo "[dry-run] Task content from: $TASK_FILE"
else
    cat > "$PROMPT_FILE" << 'HEADER'
You are implementing a task in a git worktree of the xhelio project.

## Task Specification

HEADER
    cat "$TASK_FILE" >> "$PROMPT_FILE"
    cat >> "$PROMPT_FILE" << 'FOOTER'

## Instructions

- Read ALL files listed in "Files to Modify" BEFORE writing any changes. Understand the existing code first.
- Follow existing code patterns, naming conventions, and style.
- This is a SolidJS + TypeScript frontend (in frontend/src/) with a Python backend (in agent/).
- Use Tailwind CSS utility classes for styling (the project uses Tailwind).
- Do NOT install new dependencies unless absolutely necessary.
- Do NOT modify files outside the scope listed in the task.
- Do NOT delete or rename files unless the task explicitly says to.
- Do NOT run git commit, git push, or git merge. Just write the code. An auditor will review and commit your changes.
- Do NOT run the full test suite (pytest, npm test). But DO run any commands listed in the task's "Self-Verification" section if present.
- If you encounter pre-existing errors (LSP warnings, type errors) in files you did not modify, ignore them. Only fix errors directly caused by your changes.

## Code Style Rules

- Focus on completing the task correctly. Make all changes specified in the task.
- Follow existing code patterns, naming conventions, and style in each file you modify.
- Do NOT run any code formatter (black, ruff, autopep8, prettier, eslint --fix, etc.) on modified files.
- Cosmetic reformatting is FINE. If you reformat whitespace, line breaks, trailing commas, import ordering, quote style, or comment wording while working, that is perfectly acceptable. Do not waste time reverting cosmetic changes — the reviewer will accept them without question.
- If the task has a "Change Checklist" section, walk through every item and verify you have made ALL changes before finishing. Count them.
FOOTER
fi

# ── 5. Launch agent in background ────────────────────────────────────

mkdir -p "$LOG_DIR"

if $DRY_RUN; then
    echo "[dry-run] Would launch $AGENT_NAME in $WORKTREE_DIR"
    echo "[dry-run] Log: $LOG_FILE"
    echo ""
    echo "DRY RUN COMPLETE: $SLUG → $AGENT_NAME"
    echo "Branch:    $BRANCH"
    echo "Worktree:  $WORKTREE_DIR"
    echo "Log:       $LOG_FILE"
    exit 0
fi

# Verify worktree is accessible before launching agent
if [ ! -f "$WORKTREE_DIR/.git" ] && [ ! -d "$WORKTREE_DIR/.git" ]; then
    echo "ERROR: worktree not ready at $WORKTREE_DIR (missing .git)"
    exit 1
fi

cd "$WORKTREE_DIR"

case "$AGENT" in
    opencode)
        OPENCODE_DANGEROUSLY_SKIP_PERMISSIONS=true opencode run \
            --model minimax/MiniMax-M2.5-highspeed \
            -- "Implement the task described in the attached file. Read the existing source files first, then make all changes." \
            -f .task-prompt.md \
            > "$LOG_FILE" 2>&1 &
        ;;
    gemini)
        # 10-minute timeout to prevent infinite loops on worktree issues
        timeout 600 gemini -p "$(cat .task-prompt.md)" --yolo -o text \
            > "$LOG_FILE" 2>&1 &
        ;;
esac
PID=$!

# ── 6. Register in delegations.tsv ───────────────────────────────────

echo "$SLUG|$AGENT|$PID|$BRANCH|$WORKTREE_DIR|$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_DIR/delegations.tsv"

# ── 7. Report ────────────────────────────────────────────────────────

echo "DELEGATED: $SLUG → $AGENT_NAME (PID $PID)"
echo "Branch:    $BRANCH"
echo "Worktree:  $WORKTREE_DIR"
echo "Log:       $LOG_FILE"
