# CausalReasoningEnv

Multi-flavor causal reasoning benchmark and RL training environment, built with Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework.

The environment trains models to reason about causal graphs: identifying adjustment sets, applying the do() operator, estimating treatment effects from data, and fitting structural equations. Tasks are organized into three flavors of increasing complexity, combined via `vf.EnvGroup` with a curriculum controlled by sampling weights.

See [`BENCHMARK_DESIGN.md`](../../BENCHMARK_DESIGN.md) for full design rationale, data generation specs, answer formats, and reward rubric definitions.

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

**Prompt:** The model receives a text description of the DAG (nodes classified as observed/latent, edge list, adjacency, X, Y). The system prompt provides comprehensive causal inference knowledge (d-separation, backdoor/frontdoor criteria, identifiability conditions) without worked examples.

**Expected output formats:**
```xml
<reasoning>[step-by-step causal analysis]</reasoning>
<answer>{2, 5}</answer>           <!-- backdoor: minimal adjustment set -->
<answer>{}</answer>               <!-- backdoor: empty adjustment set -->
<answer>{3}</answer>              <!-- frontdoor: mediator node -->
<answer>not_identifiable</answer> <!-- no valid identification strategy -->
```

**Problem types (6 stratified buckets):**
- `identifiable_standard` (~20%) — non-empty minimal adjustment set; all observed parents of X are required
- `identifiable_ancestor` (~15%) — non-empty set; redundancy because a dropped parent has an ancestor already in the set
- `identifiable_collider` (~20%) — non-empty set; redundancy via a collider structure on the backdoor path
- `identifiable_frontdoor` (~10%) — latent confounder blocks all backdoor adjustment; frontdoor criterion applies via a mediator M
- `empty` (~15%) — empty adjustment set; X and Y already d-separated in the backdoor graph
- `not_identifiable` (~20%) — latent L→X and L→Y with X→Y direct edge; neither backdoor nor frontdoor criterion applies

**Data:** Random DAGs (Erdős–Rényi, 8–12 nodes). All minimal adjustment sets enumerated and stored. Hosted on HuggingFace: `irfanjamil/causal-reasoning-flavor1` (250 train / 100 test).

**Reward rubric:**
- `format_compliance` (weight 0.10) — one parseable `<answer>` block
- `status_check` (weight 0.00) — correct identification strategy declared (monitoring only)
- `answer_quality` (weight 0.90) — graded: exact match on minimal set = 1.0; valid but non-minimal = scaled partial; wrong = 0.0
- `answer_correctness` (weight 0.00) — binary exact-match metric (monitoring only)

**Environment type:** `vf.SingleTurnEnv`

---

### Flavor 2 — ATE Estimation 🚧 To be implemented

**Overview:** Problems always include a DAG. Depending on the problem type, the model may also receive a linear SCM (structural equations), observational data, or neither. The prompt is neutral — it does not label the problem type. The model must determine from context what information is available and apply the appropriate method.

---

#### SCM-only problems (~20% of Flavor 2 problems)

**What the model receives:** A fully specified linear SCM (structural equations with numeric coefficients) and the DAG. No data.

**Task:** Compute exact ATE = E[Y|do(X=1)] − E[Y|do(X=0)] by tracing all directed paths from X to Y and summing the products of edge coefficients (Wright's rule).

**Why X is always a root node:** Binary X is incompatible with a linear structural equation that has parents. Therefore X ~ Bernoulli(p) with no parents — no confounders, always identifiable. The task is purely path computation.

**Variable types:** X binary, all other nodes continuous (linear Gaussian), Y continuous.

**Problem sub-types:**
- `standard` (~40%): 1–2 directed X→Y paths, ATE ≠ 0
- `mediated` (~30%): ≥2 paths through mediators; model must sum all, not just direct
- `canceling` (~20%): ≥2 opposing-sign paths; each path contribution ≥ 0.4; ATE ≈ 0
- `no_path` (~10%): no directed X→Y path; ATE = 0 (identifiable, trivially zero)

**Answer format:**
```xml
<reasoning>[path enumeration and coefficient products]</reasoning>
<ate_type>exact</ate_type>
<extra_nodes>{}</extra_nodes>
<answer>ATE=0.27</answer>
```

**Reward:** `format_compliance` (0.05) + `status_check` (0.10) + `formula_quality` (0.15) + `answer_quality` (0.70).

---

#### Data-only problems (~80% of Flavor 2 problems)

**What the model receives:** A DAG (with observed/latent node labels) and N rows of observational data (CSV, observed columns only). No SCM.

**Task:** (a) Determine whether ATE is identifiable from this DAG. (b) If identifiable: estimate ATE nonparametrically from the data.

**Variable types:** X binary, Y binary, all other nodes binary or ternary discrete.

**Problem types:**
- `backdoor_standard` (~35%): non-empty adjustment set Z, all observed, full support. Model estimates ATE.
- `backdoor_empty` (~20%): empty adjustment set — X and Y already d-separated in backdoor graph. Model estimates ATE.
- `frontdoor` (~20%): latent U→X, U→Y; valid frontdoor mediator M. Model applies two-step frontdoor formula.
- `not_identifiable` (~25%): latent confounder, no valid backdoor or frontdoor. Model declares `not_identifiable`.

**Ground truth stored per problem:**
- `true_ATE`: exact CPT enumeration — for dataset quality checks only, not for grading.
- `data_ATE`: ATE implied by the N-row sample via exact frequency counting. This is the grading target.

**Answer format:** The prompt is neutral; the model answers based on available information.

```xml
<!-- Backdoor (non-empty adjustment set) -->
<ate_type>empirical</ate_type>
<extra_nodes>{3, 5}</extra_nodes>
<answer>ATE=0.24</answer>

<!-- Backdoor empty / no extra nodes -->
<ate_type>empirical</ate_type>
<extra_nodes>{}</extra_nodes>
<answer>ATE=0.18</answer>

<!-- Frontdoor -->
<ate_type>empirical</ate_type>
<extra_nodes>{4}</extra_nodes>
<answer>ATE=0.18</answer>

<!-- Not identifiable -->
<answer>not_identifiable</answer>
```

**Reward:** `format_compliance` (0.05) + `status_check` (0.10) + `formula_quality` (0.15) + `answer_quality` (0.70).
`formula_quality`: 0.5 for correct `<ate_type>` + 0.5 for correct `<extra_nodes>` set. `answer_quality` is graded against `data_ATE`.

**Tools available:** `check_d_separation`, `load_data`, `run_python`, `find_adjustment_sets` (training only).

---

### Flavor 3 — Estimate SCM from Data 🚧 To be implemented

**Task:** Given a DAG and observational data, estimate the structural equation for Y — i.e., regress Y on its causal parents as identified from the DAG. Tests whether the model reads the DAG to select the correct parent regressors rather than naively including all correlated variables.

**Prompt:** System prompt provides causal Markov condition knowledge. User provides DAG edge list and a data CSV. Model must identify Y's parents from the DAG, run regression via `run_python`, and report structural coefficients.

**Variable types:** Linear Gaussian SCM, continuous variables, N=1000 rows.

**Key hardener:** Data contains distractor variables (correlated with Y's parents due to shared ancestors, but not direct parents). Model must use the DAG to filter.

**Expected output format:**
```xml
<answer>Y = 0.73·parent1 + −1.17·parent2 + N(0, 0.41)</answer>
```

**Reward:** `format_compliance` (0.05) + `answer_quality` (0.95).
Coefficient accuracy (0.60): mean relative error per parent ≤ 20%. Parent set selection (0.35): binary — exactly the DAG parents of Y, no more, no less.

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
prime eval run CausalReasoningEnv -a '{"weights": [1.0, 0.0, 0.0]}'

# Flavors 1 + 2 (phase 2 curriculum)
prime eval run CausalReasoningEnv -a '{"weights": [0.5, 0.5, 0.0]}'

# All three flavors (phase 3 curriculum)
prime eval run CausalReasoningEnv -a '{"weights": [0.4, 0.4, 0.2]}'
```

Weight index order: `[w_F1, w_F2, w_F3]`. Flavors with weight 0 are not instantiated.

### Train with curriculum configs

```bash
# Phase 1: Flavor 1 only
prime train --config ../../configs/lab/phase1.toml

# Advance to phase 2 from checkpoint
prime train --config ../../configs/lab/phase2.toml --resume checkpoints/step_XXXX/

# Advance to phase 3
prime train --config ../../configs/lab/phase3.toml --resume checkpoints/step_YYYY/
```

---

## File Structure

```
CausalReasoningEnv/
  CausalReasoningEnv.py       # load_environment(weights) → vf.EnvGroup
                              # weights = [w_F1, w_F2, w_F3]; default [1.0, 0.0, 0.0]
  flavor1.py                  # Flavor1Env + load_flavor1() — fully implemented
  flavor2.py                  # Flavor2Env + load_flavor2() — to be implemented
  flavor3.py                  # Flavor3Env + load_flavor3() — to be implemented
  prompts.py                  # CAUSAL_KNOWLEDGE block + build_system_prompt()
  data_generation/
    flavor1_gen.py            # DAG generation, problem sampling, dataset builder (done)
    flavor2_gen.py            # Sub-case A (linear SCM) + Sub-case B (discrete CPT) generation
    flavor3_gen.py            # Linear Gaussian SCM generation for Flavor 3
    generate_datasets_flavor1.py   # standalone: regenerate + save F1 datasets to disk
    profile_datasets_flavor1.py    # dataset profiling + distribution visualization
    upload_flavor1_datasets.py     # one-off: push local Arrow files to HuggingFace Hub
  datasets/flavor1/           # local Arrow copies (train/ + eval/)
  pyproject.toml
  README.md
```

---

## Dependencies

Declared in `pyproject.toml`: `verifiers`, `networkx`, `datasets`, `scipy`, `pandas`, `statsmodels`.
