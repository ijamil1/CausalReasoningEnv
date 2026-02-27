# CausalReasoningEnv

Multi-flavor causal reasoning benchmark and RL training environment, built with Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework.

The environment trains models to reason about causal graphs: identifying adjustment sets, applying the do() operator, estimating treatment effects from data, and fitting structural equations. Tasks are organized into four flavors of increasing complexity, combined via `vf.EnvGroup` with a curriculum controlled by sampling weights.

See [`BENCHMARK_DESIGN.md`](../../BENCHMARK_DESIGN.md) for full design rationale, data generation specs, prompt sketches, and reward rubric definitions.

---

## Install

```bash
prime env install CausalReasoningEnv
```

Or from source:

```bash
prime env install CausalReasoningEnv -p ./environments
```

---

## What's Implemented

### Flavor 1 — Adjustment Set Identification ✅

**Task:** Given a DAG with treatment node X and outcome node Y, identify the minimal valid adjustment set Z — the smallest set of non-descendants of X whose conditioning blocks all backdoor paths from X to Y (d-separation in the backdoor graph).

**Prompt:** The model receives both a text description of the DAG (edge list, X, Y) and a rendered PNG image. Two in-context examples with full chain-of-thought reasoning are included in the system prompt.

**Expected output:**
```xml
<reasoning>
[step-by-step backdoor path analysis]
</reasoning>
<answer>{2, 5}</answer>
```

**Problem types (stratified):**
- `standard` — all parents of X are needed in the adjustment set
- `ancestor` — parent redundancy via ancestor absorption (smaller set suffices)
- `collider` — parent redundancy via collider structure (some parents blocked by default)

**Data generation:** Random DAGs (Erdős–Rényi, forward edges only), 6–12 nodes. Each accepted problem has: Y is a descendant of X, Y is a leaf, at least 4 backdoor paths exist with at least one of length ≥ 5 nodes, and a valid minimal d-separator exists. The minimal adjustment set is computed using `networkx.algorithms.d_separation.find_minimal_d_separator`.

**Reward rubric:**
- `format_reward` (weight 0.05) — response contains exactly one parseable `<answer>` block
- `valid_adjustment_set` (weight 0.15) — predicted set is a valid (not necessarily minimal) adjustment set
- `correct_adjustment_set` (weight 0.80) — predicted set exactly matches the minimal adjustment set

**Environment type:** `vf.SingleTurnEnv` subclass (`Flavor1Env`).

---

### Flavor 3 — Analytical ATE from SCM 🚧 stub

**Task:** Given a DAG and fully specified structural equations (functional form + parameters + noise distributions), compute ATE = E[Y|do(X=1)] − E[Y|do(X=0)] analytically.

- Linear SCMs (75%): exact numeric answer via Wright's path-tracing
- Nonlinear SCMs (25%): substituted symbolic expression evaluable by the grader

**Status:** `flavor3.py` and `data_generation/flavor3_gen.py` are stubs. Not yet instantiated during training.

---

### Flavor 2 — ATE from Observational Data 🚧 stub

**Task:** Given a DAG and observational data (CSV), estimate ATE = E[Y|do(X=1)] − E[Y|do(X=0)] and CATE using stratified nonparametric counting over the valid adjustment set. All variables are discrete. Step 0 is always to check whether ATE is estimable from the available data.

**Status:** `flavor2.py` and `data_generation/flavor2_gen.py` are stubs. Will require tool use (`run_python`, `load_data`).

---

### Flavor 4 — Estimate SCM from Data 🚧 stub

**Task:** Given a DAG and observational data, estimate the structural equation for each node by regressing on its causal parents (per the DAG). Tests whether the model uses the DAG to select parent regressors rather than naively including all correlated variables.

**Status:** `flavor4.py` and `data_generation/flavor4_gen.py` are stubs.

---

## Usage

### Run eval (Flavor 1 only, default)

```bash
prime eval run CausalReasoningEnv
prime eval run CausalReasoningEnv -m openai/gpt-4.1-mini -n 50 -r 1
```

### Run eval with specific flavor weights

```bash
# Flavor 1 only
prime eval run CausalReasoningEnv -a '{"weights": [1.0, 0.0, 0.0, 0.0]}'

# Flavors 1 + 3 (phase 2 curriculum)
prime eval run CausalReasoningEnv -a '{"weights": [0.4, 0.6, 0.0, 0.0]}'
```

Weight index order: `[w_F1, w_F3, w_F2, w_F4]` — matches curriculum progression from graph-only tasks to data+tool tasks. Flavors with weight 0 are not instantiated.

### Train with curriculum configs

```bash
# Phase 1: Flavor 1 only
prime train --config ../../configs/lab/phase1.toml

# Advance to phase 2 from checkpoint
prime train --config ../../configs/lab/phase2.toml --resume checkpoints/step_XXXX/
```

---

## File Structure

```
CausalReasoningEnv/
  CausalReasoningEnv.py       # load_environment(weights) → vf.EnvGroup or Flavor1Env
  flavor1.py                  # Flavor1Env + load_flavor1() — fully implemented
  flavor2.py                  # Flavor2Env stub
  flavor3.py                  # Flavor3Env stub
  flavor4.py                  # Flavor4Env stub
  data_generation/
    flavor1_gen.py            # DAG generation, problem sampling, dataset builder
    flavor2_gen.py            # stub
    flavor3_gen.py            # stub
    flavor4_gen.py            # stub
  pyproject.toml
  README.md                   # this file
```

---

## Dependencies

Declared in `pyproject.toml`: `verifiers`, `networkx`, `matplotlib`, `datasets`, `scipy`, `pandas`, `statsmodels`.
