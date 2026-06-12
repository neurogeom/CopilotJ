#!/bin/sh
cd "$(dirname "$0")"/../web || exit
test -d node_modules || pnpm install
pnpm run dev
