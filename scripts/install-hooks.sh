#!/bin/bash
# Installation script for Git hooks
set -e

HOOKS_DIR="hooks"
GIT_HOOKS_DIR=".git/hooks"

echo "Installing Git hooks..."

if [ ! -d ".git" ]; then
    echo "Error: Not a git repository. Run this script from the repository root."
    exit 1
fi

if [ ! -d "$HOOKS_DIR" ]; then
    echo "Error: hooks/ directory not found"
    exit 1
fi

if [ -f "$HOOKS_DIR/pre-commit" ]; then
    cp "$HOOKS_DIR/pre-commit" "$GIT_HOOKS_DIR/pre-commit"
    chmod +x "$GIT_HOOKS_DIR/pre-commit"
    echo "Installed pre-commit hook"
else
    echo "Warning: hooks/pre-commit not found"
fi

echo "Done! Git hooks are now active."
