# just is a command runner, Justfile is very similar to Makefile, but simpler.
#
# JDK configuration (set via flake.nix env vars):
#   JAVA_HOME   -> JDK 21 (builds the single JAR: core -> Java 8 bytecode, MCP -> Java 17)
#   JAVA8_HOME  -> JDK 8  (only for ad-hoc Java 8 downgrade checks; not required to build)

default:
  @just --list

dev-server:
  python -m copilotj.server

# Dev: run the plugin in a forked Java 21 JVM (MCP enabled via the embedded bundle)
dev-plugin:
  @just build-plugin
  cd plugin && mvn exec:exec

# Build: single self-contained plugin JAR (Java 8 core + embedded Java 17 MCP bundle).
# The same JAR runs on Java 8 Fiji (core only, MCP degrades) and Java 17+ Fiji (full MCP).
build-plugin:
  cd plugin && mvn package -DskipTests

clean-plugin:
  cd plugin && mvn clean

copy-plugin-deps:
  cd plugin && mvn dependency:copy-dependencies -DoutputDirectory=target/deps

dev-web:
  cd web && pnpm run dev

build-web:
  cd web && pnpm run build

run-workflow:
  scripts/run-workflow.sh

test:
  uv run --with pytest \
    pytest \
      --doctest-modules --ignore=examples \
      --pyargs copilotj

test-cov:
  uv run --with pytest --with pytest-cov \
    pytest \
      --doctest-modules --ignore=examples \
      --cov=copilotj --cov-report=xml --cov-report=html \
      --pyargs copilotj

# Build knowledge base from source data
build-kb:
  python scripts/rag_builder.py --build

# Rebuild FAISS index from JSONL export
rebuild-kb:
  python scripts/rag_builder.py --rebuild

# Show knowledge base status
status-kb:
  python scripts/rag_builder.py --status

# Create a git worktree under .worktrees/<name> with a new branch
add-worktree name:
  scripts/add-worktree.sh {{name}}

# Interactively remove branches that are merged or closed
cleanup-branches *args:
  scripts/cleanup-branches.sh {{args}}

# --- Linting ---

lint-web:
  cd web && pnpm dlx vue-tsc -b
  cd web && prettier --check .

lint-python:
  uv run --with ruff ruff check
  uv run --with ruff ruff format --check

lint: lint-python lint-web

# --- Formatting ---

format-web:
  cd web && prettier --write .

format-python:
  uv run --with ruff ruff check --fix
  uv run --with ruff ruff format

format: format-python format-web
