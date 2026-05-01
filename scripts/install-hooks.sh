#!/bin/bash
# Installation script for Git hooks
set -euo pipefail

FORCE=0
if [ "${1:-}" = "--force" ]; then
    FORCE=1
elif [ "${1:-}" != "" ]; then
    echo "Usage: $0 [--force]"
    exit 2
fi

echo "Installing Git hooks..."

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$REPO_ROOT" ]; then
    echo "Error: Not a git repository."
    exit 1
fi

HOOKS_DIR="$REPO_ROOT/hooks"
GIT_HOOKS_DIR="$(git -C "$REPO_ROOT" rev-parse --absolute-git-dir)/hooks"

if [ ! -d "$HOOKS_DIR" ]; then
    echo "Error: hooks/ directory not found at $HOOKS_DIR"
    exit 1
fi

if [ -f "$HOOKS_DIR/pre-commit" ]; then
    mkdir -p "$GIT_HOOKS_DIR"
    if [ -e "$GIT_HOOKS_DIR/pre-commit" ] && [ "$FORCE" -ne 1 ]; then
        echo "Error: $GIT_HOOKS_DIR/pre-commit already exists. Re-run with --force to replace it."
        exit 1
    fi
    cp "$HOOKS_DIR/pre-commit" "$GIT_HOOKS_DIR/pre-commit"
    chmod +x "$GIT_HOOKS_DIR/pre-commit"
    echo "Installed pre-commit hook"
else
    echo "Warning: hooks/pre-commit not found"
fi

echo "Done! Git hooks are now active."
