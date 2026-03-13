# CausalReasoningEnv

A workspace for building causal reasoning RL training environments using Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework.

The environment trains models to identify causal effects via backdoor adjustment, frontdoor criterion, or instrumental variables, and compute ATE/LATE via probability query tools over discrete CPT-based DAGs. See [`BENCHMARK_DESIGN.md`](BENCHMARK_DESIGN.md) for the full design rationale.

## Environment

→ **[`environments/CausalReasoningEnv/`](environments/CausalReasoningEnv/)** — the main package. See its [README](environments/CausalReasoningEnv/README.md) for design details, install instructions, and eval commands.

### Two-Phase Design

**Phase 1 (declaration + tools, Turn 1):** Model reasons about the DAG, calls `declare(method, nodes)` to commit to an identification approach, and makes all probability tool calls in the same response. Rollout terminates early on format violations or invalid declarations.

**Phase 2 (answer, Turn 2):** Model receives tool results and writes `<answer>ATE=X.XXXX</answer>` (backdoor/frontdoor) or `<answer>LATE=X.XXXX</answer>` (IV).

### Reward (weights: 0.05 / 0.125 / 0.125 / 0.10 / 0.50 / 0.10)

`format_compliance` / `method_validity` / `set_validity` / `minimality` / `ate_accuracy_binary` / `ate_accuracy_l2`

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
