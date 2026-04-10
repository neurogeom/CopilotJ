#!/bin/sh
cd "$(dirname "$0")"/..
uv run --group all python -m copilotj.server --host 127.0.0.1 --port 8786
