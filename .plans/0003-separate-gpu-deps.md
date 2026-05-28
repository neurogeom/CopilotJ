# Plan: Separate GPU dependencies into a dedicated group

## Context

The project has ~3.2 GB of installed dependencies, with `tensorflow` (1.1 GB) and `torch` (347 MB, GPU build) being the heaviest. The goal is:
1. Default install to be CPU-only (lightweight, works everywhere)
2. GPU packages moved to an optional `gpu` dependency group
3. `cellpose` stays in core deps but uses CPU-only PyTorch

uv supports per-index PyTorch source configuration via `[tool.uv.sources]` + `[[tool.uv.index]]` ([docs](https://docs.astral.sh/uv/guides/integration/pytorch/)).

## Decision log

| Decision | Choice | Why |
|---|---|---|
| `tensorflow` | Move to `gpu` group | 1.1 GB, only needed by stardist |
| `stardist` | Move to `gpu` group | Hard-depends on TF; user confirmed |
| `biapy` | Move to `gpu` group | Heavy PyTorch framework; user confirmed |
| `cellpose` | Stay in core | User explicitly wants to keep it, CPU-only torch |
| `torch` | Stay in core, CPU index | cellpose/timm/etc need it |
| `csbdeep` | Stay in core | Only uses `normalize()` utility, no TF needed at runtime |
| `timm`, `torchmetrics`, `pytorch-msssim` | Stay in core | PyTorch-based, CPU OK |
| `stardist` import style | Must change to lazy | Currently top-level import at `py_tools.py:15` — would crash without `gpu` group |
| `ollama` SDK | Move to `project.dependencies` | 60KB, keep `OllamaChatCompletionClient` as-is |

## Import analysis

These packages are **Agent tool functions**, imported lazily inside function bodies (safe to move):

- `biapy` → `py_tools.py:580`: `from biapy import BiaPy` (already lazy, try/except)

These have **top-level imports** in `py_tools.py` (must be made lazy before moving):

- `stardist` → `py_tools.py:15`: `from stardist.models import StarDist2D` (top-level, will crash)

These are **referenced in prompts/configs** but don't require Python-level imports:

- `tool_agent.toml` registers `stardist_segmentation` and `biapy_tool` as tools
- `leader_prompts.py` mentions Cellpose/Stardist/BiaPy in system prompts

## Changes

### 1. `pyproject.toml` — Add CPU-only PyTorch index + source routing

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu" },
]
```

### 2. `pyproject.toml` — Move 3 packages to `[dependency-groups] gpu`

Remove from `project.dependencies`:
- `tensorflow>=2.18.1`
- `stardist>=0.9.1`
- `biapy>=3.5.6`

### 3. `pyproject.toml` — Move `ollama` to default dependencies

Move `ollama>=0.4.8` from `[dependency-groups] all` into `project.dependencies`. Delete the `all` group.

### 4. `copilotj/multiagent/py_tools.py` — Make stardist import lazy

Current (line 15):
```python
from stardist.models import StarDist2D
```

Change to lazy import inside `stardist_segmentation()`:
```python
try:
    from stardist.models import StarDist2D
except ImportError:
    raise RuntimeError("StarDist is not installed. Install with: uv sync --group gpu")
```

### 5. Resulting `pyproject.toml` structure

```toml
[project]
dependencies = [
  # ... all current deps EXCEPT tensorflow, stardist, biapy ...
  # PLUS ollama>=0.4.8 (moved from [all] group)
]

[dependency-groups]
dev = ["pytest>=8.3.4", "pytest-cov>=6.0.0", "ruff>=0.11.4"]
gpu = [
  "tensorflow>=2.18.1",
  "stardist>=0.9.1",
  "biapy>=3.5.6",
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu" },
]
```

## Files to modify

| File | Change |
|---|---|
| `pyproject.toml` | Add index, source routing, move 3 packages to gpu group, move ollama to deps, delete `all` group |
| `copilotj/multiagent/py_tools.py` | Make stardist import lazy (line 15) |

## Verification

1. `uv lock` — resolves with CPU torch
2. `uv sync` — installs without tensorflow/stardist/biapy
3. `uv sync --group gpu` — additionally installs the 3 GPU packages
4. `uv run python -c "import torch; print(torch.__version__)"` — shows `+cpu` suffix
5. `uv run python -c "import cellpose; print('ok')"` — cellpose works
6. `uv run python -c "from copilotj.multiagent.py_tools import cellpose_segmentation; print('ok')"` — no crash without stardist
