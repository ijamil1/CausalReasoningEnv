# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Before beginning work in this repository, read `AGENTS.md` and follow all scoped AGENTS guidance.

## Project Overview

This is a workspace for building RL environments using Prime Intellect's **verifiers** framework. Environments built with Verifiers are self-contained Python modules. Environments are installable Python packages that expose a `load_environment()` function returning a `vf.Environment`.
The **verifiers** source code is located at: /Users/irfanjamil/.local/share/uv/tools/prime/lib/python3.12/site-packages/verifiers

## Commands

```bash
# Install dependencies (uses uv with Python >=3.12)
uv sync

# Linting and formatting
uv run ruff check --fix .

# Tests
uv run pytest tests/
```

## Prime CLI Reference (v0.5.31)

The `prime` CLI (at `/Users/irfanjamil/.local/bin/prime`) is Prime Intellect's tool for the full environment lifecycle: scaffold, install, evaluate, publish, and train.

### Environment Management (`prime env`)

```bash
# Initialize — scaffold a new environment from template
prime env init <name>                  # creates ./environments/<name>/ with .py, pyproject.toml, README
prime env init <name> -p /other/dir   # custom output directory

# Install — install an environment into the current Python env (uses uv by default so environment package is installed into .venv)
prime env install <name>                         # local install from ./environments/
prime env install <name> -p /path/to/envs        # local install from custom path
prime env install owner/env-name                 # install from Prime Hub
prime env install owner/env-name@0.2.3           # specific version from Hub
prime env install env1 env2 env3                 # install multiple at once
prime env install <name> --with pip              # use pip instead of uv

# Uninstall
prime env uninstall <name>

# Push — publish environment to the Prime Hub registry
prime env push                        # push from current directory
prime env push -p ./environments/my_env  # push from specific path
prime env push --auto-bump            # auto-bump patch version before push
prime env push -v PUBLIC              # set visibility (PUBLIC or PRIVATE)
prime env push -o <owner> -t <team>   # push under a specific owner/team

# Pull — download environment source for local inspection
prime env pull owner/env-name                  # latest version
prime env pull owner/env-name@0.2.3            # specific version
prime env pull owner/env-name -t ./local-dir   # custom target directory

# Explore
prime env list                        # list public environments on the Hub
prime env list --mine                 # list your own environments
prime env list --search "math"        # search by name/description
prime env list --sort stars           # sort by popularity
prime env info owner/env-name         # show details and install commands
prime env status owner/env-name       # show CI action status

# Versioning & CI actions
prime env version list owner/env-name          # list all versions
prime env version delete owner/env-name <hash> # delete a version by content hash
prime env action list owner/env-name           # list CI jobs
prime env action logs owner/env-name <id>      # get CI job logs
prime env action retry owner/env-name          # retry failed CI action
```

### Evaluation (`prime eval`)

```bash
# Run an eval (default model: openai/gpt-4.1-mini via Prime Inference)
prime eval run <env-name>
prime eval run <env-name> -m openai/gpt-4.1-mini   # specify model
prime eval run <env-name> -n 50                     # number of examples
prime eval run <env-name> -r 3                      # rollouts per example
prime eval run <env-name> -t 4096                   # max tokens
prime eval run <env-name> -T 0.7                    # temperature
prime eval run <env-name> -c 64                     # max concurrent requests
prime eval run <env-name> -s                        # save results to disk
prime eval run <env-name> -a '{"key":"value"}'      # pass env args as JSON
prime eval run <env-name> -b https://api.example.com/v1  # custom API base URL
prime eval run <env-name> -k MY_API_KEY_VAR         # override API key env var
prime eval run <env-name> --skip-upload              # don't upload results to Hub

# Manage eval results
prime eval list                       # list past evaluations
prime eval list -e <env-name>         # filter by environment
prime eval push                       # push local results to Prime Evals Hub
prime eval push outputs/evals/...     # push specific result directory
prime eval tui                        # launch TUI for viewing results
```

### Other Useful Commands

```bash
# Workspace setup
prime lab setup                       # set up verifiers workspace (downloads AGENTS.md, configs)
prime lab setup --prime-rl            # also install prime-rl and download training configs

# Inference
prime inference models                # list models available on Prime Inference

# Account
prime login                           # authenticate
prime whoami                          # show current user
```

## Repository Structure

- **`environments/`** — Each subdirectory is a standalone environment package with its own `pyproject.toml`. Must export `load_environment() -> vf.Environment`. See `environments/CLAUDE.md` for the full verifiers environment API reference.
- **`configs/vf-rl/`** — TOML training configs for `prime train` (model, GPU allocation, batch sizes, env ID).
- **`configs/endpoints.py`** — `ENDPOINTS` dict mapping shorthand names to model/URL/API-key triples (Prime Inference and OpenAI). API keys are referenced by env var name (`PRIME_API_KEY`, `OPENAI_API_KEY`).
- **`configs/zero3.yaml`** — DeepSpeed ZeRO-3 accelerate config for distributed training.

## Environment Architecture

Environments use the `verifiers` library (`import verifiers as vf`). Key types in order of complexity:

1. **`vf.SingleTurnEnv`** — one prompt, one response, reward scored.
2. **`vf.MultiTurnEnv`** — multi-turn rollout loop; override `env_response()` for custom protocols.
3. **`vf.ToolEnv`** — adds stateless tool calling to multi-turn.
4. **`vf.StatefulToolEnv`** — per-rollout state injected into tool calls (sandboxes, sessions).
5. **`vf.MCPEnv`** — MCP server integration for tools.
6. **`vf.SandboxEnv` / `vf.PythonEnv`** — containerized execution via Prime Sandboxes.

Each environment needs a **dataset** (HuggingFace `Dataset` with `prompt` or `question` column) and a **rubric** (`vf.Rubric` with async reward functions). Reward functions declare needed data via argument names (`completion`, `answer`, `info`, `state`, `parser`, etc.).

## Key Conventions

- Reward functions are async, return `float` (typically 0.0–1.0), and are composed via `vf.Rubric(funcs=[...], weights=[...])`.
- Use `vf.ensure_keys([...])` in `load_environment()` to validate required API keys early.
- Use `vf.XMLParser` for structured output parsing, shared between environment and rubric.
- Lifecycle decorators: `@vf.stop` (stop conditions), `@vf.cleanup` (per-rollout), `@vf.teardown` (shutdown).
- Eval defaults go in the environment's `pyproject.toml` under `[tool.verifiers.eval]`.
