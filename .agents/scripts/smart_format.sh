#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

if [[ -n "$CLAUDE_PROJECT_DIR" ]]; then
  input=$(cat)
  filepath=$(echo "$input" | jq -r '.tool_input.file_path // empty')
fi

if [[ -z "$filepath" ]]; then
  echo "Error: no file path provided" >&2
  exit 2
fi

if [[ ! -f "$filepath" ]]; then
  echo "Error: File does not exist: $filepath" >&2
  exit 2
fi

case "$filepath" in
*.py | *.pyi | *.pyx)
  ruff format "$filepath" 2>/dev/null || true
  ;;
*.js | *.ts | *.jsx | *.tsx | *.vue | *.json | *.css | *.html | *.md | *.yml | *.yaml)
  prettier --write "$filepath" 2>/dev/null || true
  ;;
*.nix)
  nixfmt "$filepath" 2>/dev/null || true
  ;;
*)
  echo "No formatting tool configured for file type: ${filepath##*.}"
  ;;
esac
