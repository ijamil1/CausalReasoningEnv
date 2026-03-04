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

**Overview:** Two sub-cases combined into a single `vf.ToolEnv`. X is always binary. All problems require the model to reason about do() and correctly apply the relevant identification formula.

---

#### Sub-case A — Linear SCM, Analytical Path-Tracing (~20% of Flavor 2 problems)

**What the model receives:** A fully specified linear SCM (structural equations with numeric coefficients) and the DAG. No data.

**Task:** Compute ATE = E[Y|do(X=1)] − E[Y|do(X=0)] as a numeric value by tracing all directed paths from X to Y and summing the products of edge coefficients (Wright's rule).

**Why X is always a root node:** Binary X is incompatible with a linear structural equation that has parents (a linear function of continuous parents is continuous, not binary). Therefore X ~ Bernoulli(p) with no parents. Since X has no parents, there are no confounders — ATE is always identifiable. The task is purely path computation.

**Variable types:** X binary, all other nodes continuous (linear Gaussian), Y continuous.

**CATE:** Not asked in Sub-case A. Without interaction terms, CATE(Z=z) = ATE for all z.

**Problem sub-types:**
- `standard` (~40%): 1–2 directed X→Y paths, ATE ≠ 0
- `mediated` (~30%): ≥2 paths through mediators; model must sum all, not just direct
- `canceling` (~20%): ≥2 opposing-sign paths; each path contribution ≥ 0.4; ATE ≈ 0
- `no_path` (~10%): no directed X→Y path; ATE = 0 (identifiable, trivially zero)

**Ground truth:** Wright's path-tracing sum, confirmed by 1M-sample simulation.

**Answer format:**
```xml
<reasoning>[path enumeration and coefficient products]</reasoning>
<answer>ATE=0.27</answer>
```

**Reward:** `format_compliance` (0.05) + `status_check` (0.15) + `answer_quality` (0.80).
ATE tolerance: 10% relative error. ATE=0 case: full credit iff |ATE_hat| ≤ 0.05.

---

#### Sub-case B — Discrete SCM, Nonparametric ATE from Data (~80% of Flavor 2 problems)

**What the model receives:** A DAG (with observed/latent node labels) and N=5000 rows of observational data (CSV, observed columns only). No SCM.

**Task:** (a) Determine whether ATE is identifiable from this DAG and estimable from this data. (b) If estimable: estimate ATE nonparametrically. (c) If backdoor-identifiable: estimate CATE for a specified covariate stratum.

**Key principle:** The prompt provides the backdoor and frontdoor identification formulas abstractly but does NOT prescribe the estimation method. For discrete data, the principled estimator is frequency counting (empirical conditional probabilities). Whether the model imposes a parametric form vs. counts directly is part of what is measured.

**Variable types:** X binary, Y binary, all other nodes binary or ternary discrete.

**Identifiability / estimability distinction:**
- `not_identifiable` — structural failure: latent confounder blocks all adjustment sets and no frontdoor mediator exists. No data can fix this. Model declares `not_identifiable`.
- `not_estimable` — empirical failure: identification is structurally possible but data lacks overlap (one treatment arm absent in ≥1 stratum). Model declares `not_estimable`.

**Problem types:**
- `backdoor_standard` (~30%): non-empty adjustment set Z, all observed, full support. Model estimates ATE and CATE(z₀).
- `backdoor_empty` (~15%): empty adjustment set — X and Y already d-separated in backdoor graph. Model estimates ATE and CATE(z₀).
- `frontdoor` (~15%): latent U→X, U→Y; valid frontdoor mediator M. Model applies two-step frontdoor formula. CATE not asked.
- `not_identifiable` (~20%): latent confounder, no valid backdoor or frontdoor. Model declares `not_identifiable`.
- `missing_support` (~20%): valid adjustment set exists, but ≥1 stratum has no X=1 (or X=0) obs. Model declares `not_estimable`.

**Ground truth ATE:** Exact CPT enumeration (not from sampled data).
- Backdoor: `ATE = Σ_z [P(Y=1|X=1,Z=z) − P(Y=1|X=0,Z=z)] · P(Z=z)`
- Frontdoor: two-step formula over mediator M

**Answer formats:**
```xml
<!-- Identifiable, backdoor -->
<answer>status=identifiable, ATE=0.24, CATE=0.31</answer>

<!-- Identifiable, frontdoor (no CATE) -->
<answer>status=identifiable, ATE=0.18</answer>

<!-- Not identifiable (structural) -->
<answer>status=not_identifiable, reason=latent U blocks all adjustment sets; no valid frontdoor mediator</answer>

<!-- Not estimable (overlap failure) -->
<answer>status=not_estimable, reason=no X=1 observations for stratum Z=2</answer>
```

**Reward:** `format_compliance` (0.05) + `status_check` (0.15) + `answer_quality` (0.80).
ATE tolerance: 30% relative error. CATE tolerance: 40% relative error (backdoor only).
Non-estimable/non-identifiable: correct flag + correct reason = 1.0; numeric estimate = 0.0.

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
