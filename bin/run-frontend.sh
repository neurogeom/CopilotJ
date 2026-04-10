#!/bin/sh
cd "$(dirname "$0")"/../web
test -d node_modules || pnpm install
pnpm run dev
