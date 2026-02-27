# CausalReasoningEnv

A workspace for building causal reasoning RL training environments using Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework.

The goal is a **four-flavor causal reasoning benchmark** paired with a multi-turn tool-use training environment that trains models toward it. See [`BENCHMARK_DESIGN.md`](BENCHMARK_DESIGN.md) for the full design rationale, data generation specs, prompt sketches, and reward rubrics.

## Environment

→ **[`environments/CausalReasoningEnv/`](environments/CausalReasoningEnv/)** — the main package. See its [README](environments/CausalReasoningEnv/README.md) for what's implemented, how to install, and how to run eval.

## Repository Structure

```
environments/
  CausalReasoningEnv/         # Main package — load_environment(weights) → vf.EnvGroup
    CausalReasoningEnv.py     #   Entry point; routes weights to active flavor sub-envs
    flavor1.py                #   Flavor 1: adjustment set identification (fully implemented)
    flavor2.py                #   Flavor 2: ATE from observational data (stub)
    flavor3.py                #   Flavor 3: analytical ATE from SCM (stub)
    flavor4.py                #   Flavor 4: estimate SCM from data (stub)
    data_generation/
      flavor1_gen.py          #   DAG generation + dataset builder for Flavor 1
      flavor2_gen.py          #   stub
      flavor3_gen.py          #   stub
      flavor4_gen.py          #   stub
    pyproject.toml
    README.md

configs/
  lab/
    phase1.toml               # Curriculum phase 1: Flavor 1 only
    phase2.toml               # Curriculum phase 2: Flavors 1 + 3
    phase3.toml               # Curriculum phase 3: Flavors 1 + 3 + 2
    phase4.toml               # Curriculum phase 4: all four flavors

BENCHMARK_DESIGN.md           # Full benchmark and training environment design doc
```

## Setup

```bash
uv sync
prime env install CausalReasoningEnv
```
