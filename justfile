# just is a command runner, Justfile is very similar to Makefile, but simpler.
#
# JDK configuration (set via flake.nix env vars):
#   JAVA_HOME   -> JDK 21 (default, used by dev-plugin / build-plugin)
#   JAVA8_HOME  -> JDK 8  (fallback, used by dev-plugin-stable / build-plugin-stable)

default:
  @just --list

# Dev: MCP plugin (Java 21)
dev-plugin:
  @just build-plugin
  cd plugin && \
    mvn exec:exec -P mcp

# Dev: core ImageJ plugin (Java 8)
dev-plugin-stable:
  cd plugin && \
    JAVA_HOME="$JAVA8_HOME" \
    mvn compile exec:java -D"exec.mainClass=copilotj.DefaultCopilotJBridgeService" \
      -D"ij.debug=true" -D"scijava.log.level=debug" -D"copilotj.maxRetryWaitSecond=1" \
      -D"copilotj.sourcePath={{justfile_directory()}}"

dev-plugin-full-stable: clean-plugin dev-plugin-stable

# Build: MCP fat JAR (Java 21)
build-plugin:
  cd plugin && mvn package -P mcp -DskipTests

# Build: core ImageJ plugin JAR (Java 8)
build-plugin-stable: clean-plugin
  cd plugin && JAVA_HOME="$JAVA8_HOME" mvn package

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
