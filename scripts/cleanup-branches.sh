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
current_branch="$(git branch --show-current)"
default_branch="$(git remote show origin 2>/dev/null | sed -n '/HEAD branch/s/.*: //p' || echo main)"

# Phase 1: check all local branches and collect removable ones
removable=()
while IFS= read -r branch; do
  [ -n "$branch" ] || continue
  [[ "$branch" == "$current_branch" || "$branch" == "$default_branch" ]] && continue

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

  wt_name="${branch//\//__}"
  wt_path="$root/.worktrees/$wt_name"
  if git worktree list --porcelain | grep -q "^worktree $wt_path$"; then
    removable+=("$branch:has-worktree:$wt_path")
    echo "[$status] $branch (has worktree: $wt_name)"
  else
    removable+=("$branch:no-worktree:")
    echo "[$status] $branch"
  fi
done < <(git branch --format='%(refname:short)')

if [ ${#removable[@]} -eq 0 ]; then
  echo "No branches to clean up."
  exit 0
fi

# Phase 2: confirm removal
if [ "$dry_run" = true ]; then
  echo "(dry run) no changes made."
  exit 0
fi

echo ""
read -r -p "Remove all ${#removable[@]} branch(es)? [y/N] " answer
if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
  echo "Aborted."
  exit 0
fi

# Phase 3: remove
for entry in "${removable[@]}"; do
  branch="${entry%%:*}"
  rest="${entry#*:}"
  has_wt="${rest%%:*}"
  wt_path="${rest#*:}"

  if [ "$has_wt" = "has-worktree" ]; then
    git worktree remove --force "$wt_path"
    echo "Removed worktree $(basename "$wt_path")"
  fi
  git branch -D "$branch" 2>/dev/null || true
done

git worktree prune
