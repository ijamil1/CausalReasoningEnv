# CausalReasoningEnv

A workspace for building causal reasoning RL training environments using Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework.

The goal is a **three-flavor causal reasoning benchmark** paired with a multi-turn tool-use training environment that trains models toward it. See [`BENCHMARK_DESIGN.md`](BENCHMARK_DESIGN.md) for the full design rationale, data generation specs, answer formats, and reward rubrics.

## Environment

→ **[`environments/CausalReasoningEnv/`](environments/CausalReasoningEnv/)** — the main package. See its [README](environments/CausalReasoningEnv/README.md) for what's implemented, how to install, and how to run eval.

## Repository Structure

```
environments/
  CausalReasoningEnv/         # Main package — load_environment(weights) → vf.EnvGroup
    CausalReasoningEnv.py     #   Entry point; routes weights to active flavor sub-envs
    flavor1.py                #   Flavor 1: adjustment set identification (fully implemented)
    flavor2.py                #   Flavor 2: ATE estimation — linear SCM path-tracing (Sub-case A)
                              #             + nonparametric ATE from discrete data (Sub-case B)
    flavor3.py                #   Flavor 3: estimate structural equation from data
    prompts.py                #   Shared causal knowledge block + system prompt builder
    data_generation/
      flavor1_gen.py          #   DAG generation + dataset builder for Flavor 1
      flavor2_gen.py          #   Data generation for Flavor 2 (Sub-cases A and B)
      flavor3_gen.py          #   Data generation for Flavor 3
    pyproject.toml
    README.md

configs/
  lab/
    phase1.toml               # Curriculum phase 1: Flavor 1 only
    phase2.toml               # Curriculum phase 2: Flavors 1 + 2
    phase3.toml               # Curriculum phase 3: all three flavors

BENCHMARK_DESIGN.md           # Full benchmark and training environment design doc
```

## Setup

```bash
uv sync
prime env install CausalReasoningEnv
```
