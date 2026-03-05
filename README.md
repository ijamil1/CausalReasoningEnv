# CausalReasoningEnv

A workspace for building causal reasoning RL training environments using Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework.

The environment trains models to identify causal adjustment strategies and compute Average Treatment Effects (ATE) via probability query tools over discrete CPT-based DAGs. See [`BENCHMARK_DESIGN.md`](BENCHMARK_DESIGN.md) for the full design rationale.

## Environment

→ **[`environments/CausalReasoningEnv/`](environments/CausalReasoningEnv/)** — the main package. See its [README](environments/CausalReasoningEnv/README.md) for design details, install instructions, and eval commands.

### Two-Phase Design

**Phase 1 (declaration):** Model reasons about the DAG and writes `<set>…</set>` to declare its identification set — scored independently of computation.

**Phase 2 (tool use + answer):** Model calls `marginal()` / `conditional()` tools and writes `<answer>ATE=…</answer>` or `<answer>not_identifiable</answer>` to end the episode.

### Reward (weights: 0.05 / 0.30 / 0.15 / 0.50)

`format_compliance` / `set_valid` / `minimality` / `ate_accuracy`

## Repository Structure

```
environments/
  CausalReasoningEnv/              # Main package — load_environment() → CausalATEEnv
    CausalReasoningEnv.py          #   Entry point
    env.py                         #   CausalATEEnv — tools, rubric, two-phase stop logic
    prompts.py                     #   SYSTEM_PROMPT — declaration format, tool docs
    data_generation/
      gen.py                       #   DAG generation, problem sampling, dataset builder
      generate_datasets.py         #   CLI: regenerate and upload datasets
    pyproject.toml
    README.md

configs/
  lab/
    phase1.toml                    # Training config: CausalReasoningEnv

BENCHMARK_DESIGN.md                # Full benchmark design doc
IMPLEMENTATION_PLAN.md             # Detailed implementation spec (declaration + tool use design)
```

## Setup

```bash
uv sync
prime env install CausalReasoningEnv
```
