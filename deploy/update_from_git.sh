#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This script must be run inside a git repository."
  exit 1
fi

echo "Fetching latest commits..."
git fetch --all --prune

echo "Pulling current branch (fast-forward only)..."
git pull --ff-only

echo "Rebuilding image and restarting service..."
docker compose build --pull vote-sim-ui
docker compose up -d --remove-orphans vote-sim-ui

echo "Deployment updated successfully."
