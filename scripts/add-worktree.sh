#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <branch-name>" >&2
  exit 1
fi

name="$1"
path="${name//\//__}"
root="$(cd "$(dirname "$(git rev-parse --git-common-dir)")" && pwd)"
worktree="$root/.worktrees/$path"

# Refresh remote refs (non-fatal, skips if offline)
git fetch --quiet --all >/dev/null 2>&1 || true

mkdir -p "$(dirname "$worktree")"

if git show-ref --verify --quiet "refs/heads/$name"; then
  # Local branch already exists
  if git worktree list --porcelain | awk '/^branch/{print $2}' | grep -qFx "refs/heads/$name"; then
    existing=$(git worktree list --porcelain | awk "/^branch refs\\/heads\\/${name}$/{found=1} found&&/^worktree/{print \$2;exit}")
    echo "error: branch '$name' is already checked out in a worktree ($existing)" >&2
    exit 1
  fi

  echo "Branch '$name' already exists but is not checked out in any worktree."
  read -rp "Check it out in a new worktree? [y/N] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi

  git worktree add "$worktree" "$name"
elif matching_remote="$(git remote | while read -r r; do
       git show-ref --verify --quiet "refs/remotes/$r/$name" && echo "$r" && break
     done)" && [ -n "$matching_remote" ]; then
  # Remote branch exists — checkout from it
  echo "Branch '$name' exists on remote '$matching_remote'."
  read -rp "Check it out in a new worktree? [y/N] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi

  git worktree add -b "$name" "$worktree" "$matching_remote/$name"
else
  # Branch doesn't exist — create it
  git worktree add -b "$name" "$worktree"
fi

# Copy gitignored .env files from the original worktree
for f in $(git -C "$root" ls-files --others --ignored --exclude-standard -- ".env*"); do
  cp "$root/$f" "$worktree/$f"
done

# Activate direnv for the new worktree if direnv is installed and current directory is allowed
# `foundRC.allowed`: 0 -> allowed, 2 -> denied
if command -v direnv >/dev/null 2>&1 && command -v jq >/dev/null 2>&1; then
  if direnv status --json |
    jq -e --arg p "$(pwd -P)/.envrc" '.state.foundRC.path == $p and .state.foundRC.allowed == 0' >/dev/null; then
    echo "==> current direnv is allowed, propagating to worktree"
    direnv allow "$worktree"
  else
    echo "==> current direnv not allowed (or no .envrc), skip"
  fi
fi

# Install all dependencies for a fresh worktree, fail quietly
(cd "$worktree" && uv sync >/dev/null 2>&1 || true) &
(cd "$worktree/plugin" && mvn dependency:resolve >/dev/null 2>&1 || true) &
(cd "$worktree/web" && pnpm install >/dev/null 2>&1 || true) &
wait

echo "Worktree '$name' created at '$worktree', you can start working on it:"
echo "  cd $worktree"
