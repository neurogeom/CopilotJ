# just is a command runner, Justfile is very similar to Makefile, but simpler.

default:
  @just --list

dev-server:
  python -m copilotj.server

dev-plugin:
  cd plugin && \
    mvn exec:java -D"exec.mainClass=copilotj.DefaultCopilotJBridgeService" \
      -D"ij.debug=true" -D"scijava.log.level=debug" -D"copilotj.maxRetryWaitSecond=1"

dev-plugin-full: clean-plugin dev-plugin

build-plugin: clean-plugin
  cd plugin && mvn package

clean-plugin:
  cd plugin && mvn clean

compile-plugin:
  cd plugin && mvn compile

copy-plugin-deps:
  cd plugin && mvn dependency:copy-dependencies -DoutputDirectory=target/deps

dev-web:
  cd web && pnpm run dev

build-web:
  cd web && pnpm run build

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

# Create a git worktree under .worktrees/<name> with a new branch
add-worktree name:
  scripts/add-worktree.sh {{name}}
