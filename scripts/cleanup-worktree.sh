#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

dry_run=false
if [ $# -eq 1 ] && [ "$1" = "--dry-run" ]; then
  dry_run=true
elif [ $# -ne 0 ]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 1
fi

root="$(cd "$(dirname "$(git rev-parse --git-common-dir)")" && pwd)"

# Phase 1: check all worktrees and collect removable ones
removable=()
while IFS= read -r wt; do
  [ -n "$wt" ] || continue
  name="$(basename "$wt")"
  branch="${name//__//}"

  if ! git show-ref --quiet --heads "$branch"; then
    status="gone"
  else
    pr_state=$(gh pr view "$branch" --json state --jq '.state' 2>/dev/null || echo "NO_PR")
    case "$pr_state" in
      MERGED)  status="merged" ;;
      CLOSED)  status="closed" ;;
      *)       status="active" ;;
    esac
  fi

  [ "$status" = "active" ] && continue

  echo "[$status] $name (branch: $branch)"
  removable+=("$wt:$branch")
done < <(git worktree list --porcelain | grep "^worktree " | cut -d' ' -f2 | grep "^$root/.worktrees/")

if [ ${#removable[@]} -eq 0 ]; then
  echo "No worktrees to clean up."
  exit 0
fi

# Phase 2: confirm removal
if [ "$dry_run" = true ]; then
  echo "(dry run) no changes made."
  exit 0
fi

echo ""
read -r -p "Remove all ${#removable[@]} worktree(s)? [y/N] " answer
if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
  echo "Aborted."
  exit 0
fi

# Phase 3: remove
for entry in "${removable[@]}"; do
  wt="${entry%%:*}"
  branch="${entry#*:}"
  git worktree remove --force "$wt"
  git branch -D "$branch" 2>/dev/null || true
  echo "Removed $(basename "$wt")"
done

git worktree prune
