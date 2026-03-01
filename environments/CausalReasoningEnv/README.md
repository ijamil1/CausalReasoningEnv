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

### Flavor 1 — Causal Identification ✅

**Task:** Given a DAG with treatment node X, outcome node Y, and a mix of observed/latent nodes: (a) determine the identifiability status of ATE, and (b) produce the appropriate answer — minimal backdoor adjustment set, frontdoor mediator, or a non-identifiability declaration.

**Prompt:** The model receives a text description of the DAG (nodes classified as observed/latent, edge list, X, Y) and a rendered PNG image (blue=X, orange=Y, gray=observed, light purple=latent). The system prompt provides comprehensive causal inference knowledge (d-separation, backdoor/frontdoor criteria, identifiability conditions) without worked examples.

**Expected output formats:**
```xml
<reasoning>[step-by-step causal analysis]</reasoning>
<answer>{2, 5}</answer>                  <!-- backdoor: adjustment set -->
<answer>frontdoor: {3}</answer>          <!-- frontdoor: mediator node -->
<answer>not_identifiable</answer>        <!-- no valid identification strategy -->
```

**Problem types (6 stratified buckets):**
- `identifiable_standard` (~20%) — non-empty minimal adjustment set; all observed parents of X are required
- `identifiable_ancestor` (~15%) — non-empty set; redundancy because a dropped parent has an ancestor already in the set
- `identifiable_collider` (~20%) — non-empty set; redundancy via a collider structure on the backdoor path
- `identifiable_frontdoor` (~10%) — latent confounder blocks all backdoor adjustment; frontdoor criterion applies via a mediator M
- `empty` (~15%) — empty adjustment set; X and Y already d-separated in the backdoor graph
- `not_identifiable` (~20%) — latent L→X and L→Y with X→Y direct edge; neither backdoor nor frontdoor criterion applies

**Data generation:** Random DAGs (Erdős–Rényi, forward edges only), 8–12 nodes. Non-X/Y nodes randomly marked latent (p=0.3). All minimal adjustment sets enumerated and stored as `minimal_adjustment_sets` in `info`. Datasets hosted on HuggingFace: `irfanjamil/causal-reasoning-flavor1` (250 train / 100 test). Loaded via `load_dataset()` in `load_flavor1()`.

**Reward rubric:**
- `format_compliance` (weight 0.10) — response contains exactly one parseable `<answer>` block
- `status_check` (weight 0.10) — correct identification method declared (backdoor / frontdoor / not_identifiable)
- `answer_correctness` (weight 0.80) — exact match against any element of `minimal_adjustment_sets` = 1.0; valid but non-minimal = 0.5; wrong type or invalid = 0.0

**Environment type:** `vf.SingleTurnEnv` subclass (`Flavor1Env`). Uses `setup_state()` to inject a base64-encoded PNG of the DAG into the user message as a multimodal image.

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
  prompts.py                  # shared prompt components (CAUSAL_KNOWLEDGE, build_system_prompt)
  data_generation/
    flavor1_gen.py            # DAG generation, problem sampling, dataset builder
    generate_datasets_flavor1.py  # standalone script: regenerate + save datasets to disk
    profile_datasets_flavor1.py   # dataset profiling and distribution visualization
    upload_flavor1_datasets.py    # one-off: push local Arrow files to HuggingFace Hub
    flavor2_gen.py            # stub
    flavor3_gen.py            # stub
    flavor4_gen.py            # stub
  datasets/flavor1/           # local Arrow copies (train/ + eval/) — source of truth for HF upload
  pyproject.toml
  README.md                   # this file
```

---

## Dependencies

Declared in `pyproject.toml`: `verifiers`, `networkx`, `matplotlib`, `datasets`, `scipy`, `pandas`, `statsmodels`.
