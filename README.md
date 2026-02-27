# CausalReasoningEnv

A workspace for building causal reasoning RL training environments and benchmarks using Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework.

The long-term goal is a **five-flavor causal reasoning benchmark** that tests end-to-end causal analysis competency — from reading DAGs and applying d-separation, to estimating ATEs from data and writing identification formulas — paired with a multi-turn tool-use training environment (`CausalReasoningEnv_2`) that trains models toward this benchmark.

## Setup

```bash
# Install dependencies
uv sync

# Install an environment locally
prime env install CausalReasoningEnv_1
```

## Environments

| Environment | Status | Description |
| ----------- | ------ | ----------- |
| [CausalReasoningEnv_1](environments/CausalReasoningEnv_1/) | ✅ Built | Single-turn environment: given a DAG, identify the minimal adjustment set that blocks all backdoor paths from treatment X to outcome Y. Problems are stratified by difficulty (standard, collider, ancestor). |
| CausalReasoningEnv_2 | 🚧 Planned | Multi-turn tool-use environment covering all five benchmark flavors via `vf.EnvGroup`. Uses Python execution tools for estimation tasks and graph tools (d-separation check, adjustment set finder) for identification tasks. |

## Benchmark Design — Five Flavors

See [BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md) for full design rationale, data generation specs, prompt sketches, and reward rubric definitions.

| Flavor | Task | Key Skill Tested |
| ------ | ---- | ---------------- |
| 1 — Adjustment Set | Given DAG + X, Y: find minimal adjustment set | d-separation, backdoor criterion, collider logic |
| 2 — ATE from Data | Given DAG + observational data: estimate ATE/CATE numerically | Identification + estimation pipeline; tool-use for regression |
| 3 — Analytical ATE | Given DAG + fully specified SCM: compute exact E[Y\|do(X=x)] | do() operator, graph mutilation, causal vs. observational conditioning |
| 4 — Estimate SCM | Given DAG + data: estimate structural equations | Causal Markov condition; regress on parents, not correlated variables |
| 5 — Identification Formula | Given DAG + equation forms (no params): write backdoor adjustment formula | Symbolic identification; express E[Y\|do(X=x)] as observable distribution |

## Usage

```bash
# Run evaluation
prime eval run CausalReasoningEnv_1

# Run evaluation with a specific model
prime eval run CausalReasoningEnv_1 -m openai/gpt-4.1-mini -n 50

# Push to Prime Hub
prime env push -p ./environments/CausalReasoningEnv_1
```

## Repository Structure

```
environments/
  CausalReasoningEnv_1/   # Flavor 1: adjustment set identification (built)
  CausalReasoningEnv_2/   # All 5 flavors via EnvGroup (planned)
configs/
  vf-rl/                  # Training configs (TOML)
  endpoints.py            # Model endpoint shorthands
  zero3.yaml              # DeepSpeed ZeRO-3 config
BENCHMARK_DESIGN.md       # Full benchmark and training env design doc
```