# Causal Reasoning Benchmark & Training Environment Design

## Context

The existing `CausalReasoningEnv_1` benchmarks only one causal task — minimal adjustment set identification — and does so with heavy scaffolding (detailed ICL examples, step-by-step instructions). The goal is to expand this into a full causal reasoning benchmark covering three distinct "flavors" of tasks that together test end-to-end causal analysis competency, then design a multi-turn tool-use training environment with a principled reward rubric to train models toward this benchmark.

---

## Part I: Benchmark Analysis — Four Flavors

### Flavor 1 — Causal Identification ✅ Implemented

**Task:** Given a DAG with treatment X, outcome Y, and a mix of observed/latent nodes: (a) determine the identifiability status of ATE (backdoor-identifiable, frontdoor-identifiable, empty, or not-identifiable), and (b) produce the appropriate answer — minimal backdoor adjustment set, frontdoor mediator, or non-identifiability declaration.

**What it tests:** d-separation, backdoor criterion, collider logic, non-descendant constraint, and crucially — the ability to diagnose *when* no valid adjustment set exists.

**Critiques & Pointers:**
- **Hand-holding is heavy.** The current system prompt teaches the backdoor criterion step-by-step. For a *benchmark*, this measures "can the model follow instructions" not "does it understand causal graphs." The benchmark variant should strip the ICL examples to definitional knowledge only — no worked examples.
- **Multiple valid minimal sets.** `find_minimal_d_separator` returns *one* minimal set, but other minimal sets may exist. Grading on exact match against a single gold answer will penalize correct alternative answers. Fix: at generation time, enumerate all minimal adjustment sets and store the full set of valid answers.
- **Jaccard reward can mislead training.** If a model returns `{all parents}`, it gets partial Jaccard credit but is wrong on minimality. Add a **minimality penalty**: score is 0 if `|predicted| > |gold|` and the predicted set is valid.
- **The `valid_adjustment_set` reward is currently weighted 0.0.** For training it should carry some weight to teach the validity constraint before minimality.
- **Collider chain trap.** A node C may be a collider on path P1 (conditioning on C blocks P1) but a non-collider on path P2 (conditioning on C opens P2). These cases directly test collider logic — the model must avoid C or find an alternative set that handles both paths simultaneously.
- **No-set and empty-set cases.** ~20% of problems should require "no valid adjustment set exists" or "no adjustment needed" as the correct answer. Models are biased toward producing a non-empty set; these cases test whether the model correctly diagnoses the graph rather than finding something to adjust for.

**Problem Types (six stratified buckets — implemented in `flavor1_gen.py`):**
```
identifiable_standard (~20%):
  Non-empty minimal adjustment set; all observed parents of X are required
  (|min_set| ≥ |observed_parents(X)|).

identifiable_ancestor (~15%):
  Non-empty set; |min_set| < |observed_parents(X)|; redundancy because a
  dropped observed parent has an ancestor already in min_set.

identifiable_collider (~20%):
  Non-empty set; |min_set| < |observed_parents(X)|; redundancy via a
  collider structure on the backdoor path through that parent.
  NOTE: type (e) collider-chain problems are subsumed here — they are a
  subset of identifiable_collider.

identifiable_frontdoor (~10%):
  Latent confounder U→X and U→Y blocks all valid backdoor adjustment sets.
  A mediator M exists (X→M→Y) with no unblocked backdoor paths from X to M
  and all backdoor paths from M to Y blocked by X. Frontdoor criterion applies.
  Model must identify M and declare "frontdoor" as the identification strategy.

empty (~15%):
  Empty minimal adjustment set.  X and Y are already d-separated in the
  backdoor graph (all paths blocked by unconditioned colliders).  Requires
  ≥1 undirected path in G_bd (result is structural, not trivial).

not_identifiable (~20%):
  No valid observed adjustment set exists.  A synthetic latent node L is
  added with direct edges L→X and L→Y.  X→Y is a direct edge and the ONLY
  causal path (no X→M→Y mediator — ruling out the front-door criterion).
  Model must declare ATE is not identifiable.
```

**Data Generation (current implementation in `flavor1_gen.py`):**
```
Observed / latent node assignment:
- Every problem includes observed_nodes and latent_nodes fields.
- X and Y are always observed.
- For identifiable/empty types: non-X, non-Y nodes are independently marked
  latent (Bernoulli p=0.3) BEFORE the minimal adjustment set is computed.
  The d-separator is then restricted to observed_nodes − {X, Y}.
  If no observed adjustment set exists, the sample is discarded.
- For not_identifiable: a fresh latent node L is appended with L→X and L→Y.

Sampling filters (all types):
- Directed X→Y path required.
- Y is a leaf node (no outgoing edges in G).
- identifiable: ≥4 undirected backdoor paths in G_bd, ≥1 of length ≥5.
- empty: ≥1 undirected path in G_bd.
- not_identifiable: X→Y must be the only causal path (no mediator);
  find_minimal_d_separator(G_bd, X, Y, restricted=observed−{X,Y}) = None.

Uniqueness: (frozenset(edges), X, Y) signatures deduplicated across split.
Train/eval split: stratified per bucket (~71%/29% for n_train=250, n_eval=100).
```

**Reward (implemented in `flavor1.py`):**
```
- format_compliance (0.10): exactly one parseable <answer> block
- status_check (0.10): correct identification method declared
    ("backdoor" for identifiable/empty, "frontdoor", or "not_identifiable")
- answer_correctness (0.80):
    - backdoor: 1.0 if predicted set ∈ minimal_adjustment_sets;
                0.5 if valid d-separator but non-minimal;
                0.0 otherwise
    - frontdoor: 1.0 if predicted mediator matches mediator_node
    - empty: 1.0 if predicted set is {}
    - not_identifiable: 1.0 if type declared as not_identifiable
```

**Benchmark Prompt Sketch:**
```
System: [Comprehensive causal inference knowledge prompt — see Prompt Design Strategy section]

User: "DAG: [edges]. X=[x], Y=[y].
(a) Is ATE identifiable by backdoor adjustment? State yes or no and why.
(b) If yes: what is the minimal valid adjustment set?
    If no: which structural condition fails?
<answer>identifiable=[yes/no], Z=[set or "none"], reason=[...]</answer>"
```

---

### Flavor 2 — ATE Estimation (Analytical + Nonparametric) 🚧 To be implemented

**Overview:** This flavor combines two complementary ATE computation tasks under a single multi-turn tool-use environment.

- **Sub-case A (~20% of problems):** The model is given a fully specified linear SCM and must compute ATE analytically via directed path-tracing (Wright's rule). No data is provided.
- **Sub-case B (~80% of problems):** The model is given a DAG and observational data (no SCM). The model must first determine whether ATE is identifiable/estimable, then estimate it nonparametrically from the data if estimable.

**Environment type:** `vf.ToolEnv` (multi-turn, 8–12 turns). Tools: `load_data`, `check_d_separation`, `run_python`, and `find_adjustment_sets` (training scaffold only — not available at eval time).

---

#### Sub-case A — Linear SCM, Analytical Path-Tracing (~20%)

**What it tests:** Does the model correctly apply do(X=x) by mutilating incoming edges to X and tracing ALL directed paths from X to Y, multiplying edge coefficients along each path and summing across paths? A model that reports only the direct structural coefficient β_XY from Y's equation — ignoring mediated paths — will fail on mediated and canceling-path problems.

**Why X must be a root node:** In a linear SCM, every node V satisfies `V = Σβᵢ·parentᵢ + ε`. Any node with parents inherits a continuous distribution from them and is itself continuous-valued. There is no way to make such a node binary while preserving the linear structural equation: thresholding introduces a step function (nonlinear), and a logistic link produces a Bernoulli mechanism (not a linear SCM). Therefore, binary X requires X to be exogenous — a root node with no parents (including no latent parents). Since X is a root node, no confounders of X exist (a confounder U→X, U→Y would make U a parent of X). Consequently, ATE is always identifiable in Sub-case A — the identifiability challenge is absent. The task is purely analytical path computation.

**Variable types:**
- X: binary root node, `X ~ Bernoulli(p)`, p ∈ [0.3, 0.7]
- All other nodes: linear continuous, `V = Σ β_i · parent_i + ε`,  ε ~ N(0, σ²)
- Y: continuous leaf node (no outgoing edges)
- No latent variables

Note: CATE is not asked in Sub-case A. In a linear additive model without interaction terms, CATE(Z=z) = ATE for all z — there is no treatment effect heterogeneity.

**SCM parameter generation:**
```
- 5–10 total nodes
- β_i ~ Uniform([0.2, 1.5] ∪ [−1.5, −0.2])  — non-round values (e.g. 0.73, −1.17, 0.84)
- σ ~ Uniform(0.1, 0.5) per node
- Y is always a leaf node (no outgoing edges)
```

**Problem sub-types:**
```
standard   (~40%): 1–2 directed X→Y paths, ATE ≠ 0, no sign cancellation.
                   Example: X→Y (direct coeff 0.73) + X→M→Y (mediator chain).
                   Model must sum both path contributions, not just report β_XY.

mediated   (~30%): ≥2 directed paths through mediators; no direct X→Y edge required.
                   Key test: model must identify ALL directed paths, not only the
                   direct link from X to Y.

canceling  (~20%): ≥2 directed paths with opposing signs.
                   Requirement: each path's absolute contribution ≥ 0.4; ATE within
                   ±0.05 of 0. A model reporting only the direct path coefficient fails.

no_path    (~10%): no directed X→Y path exists; ATE = 0 by construction.
                   ATE is still identifiable — intervening produces zero change.
                   Test: does the model correctly conclude ATE=0 rather than
                   hallucinating a path or refusing to answer?
```

**Ground truth ATE:**
```
ATE = Σ_{all directed paths P: X →...→ Y}  Π_{(u,v) ∈ P}  β_{u,v}

This is Wright's path-tracing rule applied to the mutilated graph do(X=x).
Since X is a root node, do(X=x) removes no edges (X has no parents).
Trace all directed paths from X to Y; multiply the structural coefficients
along each path; sum across paths.

ATE_true = 0 for the no_path sub-type.
All ATEs are also confirmed via simulation: 1M samples under do(X=0) and do(X=1).
```

**What the model sees:**
```
- DAG edge list
- Full structural equations in text form, e.g.:
    X  ~ Bernoulli(0.45)
    Z1 = N(0, 1)
    Z2 = 0.73·Z1 + N(0, 0.4)
    M  = 1.17·X + 0.61·Z2 + N(0, 0.3)
    Y  = −0.84·X + 0.95·M + N(0, 0.5)
- X and Y designated
- No data (load_data is not applicable for Sub-case A)
```

**Answer format:**
```xml
<reasoning>[path enumeration and coefficient product computation]</reasoning>
<answer>ATE=0.27</answer>
```

**Reward:**
```
format_compliance (0.05): parseable <answer>ATE=...</answer> tag
status_check      (0.15): Sub-case A is always identifiable; full credit if model
                          correctly does not declare not_identifiable / not_estimable.
answer_quality    (0.80): max(0, 1 − |ATE_hat − ATE_true| / (0.1 · |ATE_true|))
                          Tight tolerance (10%) — exact algebraic answer expected.
                          Special case ATE_true = 0: full credit iff |ATE_hat| ≤ 0.05.
```

---

#### Sub-case B — Discrete SCM, Nonparametric ATE from Data (~80%)

**What it tests:** (1) Identification status diagnosis — can the model determine whether ATE is identifiable or estimable from the given DAG and data? (2) Nonparametric ATE estimation — given an identifiable case, can the model correctly apply the backdoor or frontdoor formula using discrete frequency counts from the data? (3) CATE for specified covariate strata (backdoor cases only).

**Key design principle — do not prescribe the estimation method:** The model is given the DAG and data, NOT the SCM. The principled estimation method for discrete data is nonparametric frequency counting — estimating P(Y=1|X=x, Z=z) as the empirical conditional frequency. Imposing a parametric model (logistic regression, linear probability model) introduces assumptions the data does not require and that the true SCM may violate. The system prompt provides the backdoor and frontdoor identification formulas abstractly but does NOT say "use stratified counting." Whether the model reaches for counting vs. a parametric form is part of what is being measured.

**Identifiability vs. estimability distinction:**
- `not_identifiable`: a structural property of the DAG — latent confounders block all valid adjustment sets and no frontdoor mediator exists. No amount of data resolves this.
- `not_estimable`: identification is possible in principle, but the available data lacks sufficient overlap. Both cases require the model to withhold a numeric ATE estimate, but for different structural reasons. Models should distinguish these with different status strings.

**Variable types:**
- X: binary (can have parents — binary X with discrete parents is valid via CPTs)
- Y: binary → ATE = P(Y=1|do(X=1)) − P(Y=1|do(X=0))
- All other variables: binary or ternary (2–3 categories)

**Problem types:**
```
backdoor_standard  (~30%): non-empty minimal adjustment set Z; all variables in Z
                           are observed; all strata Z=z have observations for both
                           X=0 and X=1. Model: identify Z, apply backdoor formula,
                           estimate ATE and CATE for a specified stratum z₀.

backdoor_empty     (~15%): empty adjustment set — X and Y are d-separated in the
                           backdoor graph (no conditioning needed). Model: recognize
                           no adjustment is required, estimate ATE directly as
                           P(Y=1|X=1)−P(Y=1|X=0), and estimate CATE.

frontdoor          (~15%): latent U→X, U→Y; no valid backdoor adjustment set.
                           A valid frontdoor mediator M exists (X→M→Y with all three
                           frontdoor conditions satisfied). Model: identify M, apply
                           the two-step frontdoor formula for ATE.
                           CATE is NOT asked for frontdoor cases.

not_identifiable   (~20%): latent U→X, U→Y; no valid backdoor adjustment set exists;
                           no valid frontdoor mediator (conditions violated). ATE
                           cannot be identified from observational data regardless of
                           sample size. Model: declare not_identifiable with a brief
                           structural explanation. Producing a numeric estimate is
                           penalized at 0.

missing_support    (~20%): ATE is structurally identifiable (valid adjustment set Z
                           exists, all variables observed), but ≥1 stratum Z=z in
                           the minimal adjustment set has no observations for one
                           treatment arm (X=1 or X=0). ATE cannot be estimated from
                           this data. Model: declare not_estimable with an overlap
                           explanation. Producing a numeric estimate is penalized at 0.
```

**SCM generation:**
```
1. Generate DAG (5–10 nodes, Erdős–Rényi with forward topological ordering).
   Apply the same type-specific structural filters used in Flavor 1:
   - backdoor_standard/empty: ≥1 undirected backdoor path; minimal adjustment set
     exists and all variables in it are observed.
   - frontdoor: add latent U with U→X and U→Y; verify valid mediator M on X→M→Y
     satisfying all three frontdoor conditions.
   - not_identifiable: add latent U with U→X and U→Y; verify no valid observed
     adjustment set exists and no valid frontdoor mediator exists.
   - missing_support: generate a valid backdoor_standard problem, then post-hoc
     zero out all X=1 rows (or X=0 rows) for ≥1 stratum of the adjustment set.

2. Assign variable types:
   - X and Y: always binary.
   - Other nodes: randomly assign binary (60%) or ternary (40%) per node.
   - Latent nodes appear in the causal structure but are NEVER included in the
     data CSV.

3. Parameterize as conditional probability tables (CPTs):
   - P(V=v | pa(V)) for each node; all probabilities in [0.1, 0.9] (avoid degeneracy).
   - X CPT: P(X=1 | pa(X)) ∈ [0.2, 0.8] for all parent combinations.
   - Verify identifiability conditions hold before sampling data.

4. Sample N=5000 observations from the joint distribution.
   Rationale: with binary/ternary adjustment variables and |Z| ≤ 2, there are at
   most 3² = 9 strata; N=5000 gives ~555 expected obs/stratum — sufficient for
   stable frequency estimates.

5. Drop latent variable columns from the data CSV.

6. Compute ground truth ATE via exact enumeration over CPTs (not from sampled data):
   Backdoor:
     ATE = Σ_z [P(Y=1|X=1,Z=z) − P(Y=1|X=0,Z=z)] · P(Z=z)
     CATE(z₀) = P(Y=1|X=1,Z=z₀) − P(Y=1|X=0,Z=z₀)

   Frontdoor (M = mediator node):
     ATE = Σ_m P(M=m|X=1) · Σ_x' P(Y=1|X=x',M=m)·P(X=x')
         − Σ_m P(M=m|X=0) · Σ_x' P(Y=1|X=x',M=m)·P(X=x')

   not_identifiable / missing_support: true_ATE = None, true_CATE = None

7. Store per problem:
   edges, observed_nodes, latent_nodes, data_csv (str), X, Y, problem_type,
   identifiability_status ("identifiable" | "not_identifiable" | "not_estimable"),
   true_ATE (None if non-estimable), true_CATE (None for frontdoor/not_identifiable/
   missing_support), adjustment_set (or mediator_node for frontdoor).
   CPTs are NOT stored in the info dict — the model never sees them.
```

**What the model sees:**
```
- DAG edge list
- List of observed nodes and latent nodes
- Full data CSV (N=5000 rows, observed columns only), accessible via load_data tool
- X and Y designated
- For backdoor_standard and backdoor_empty: a CATE question specifying Z=z₀
```

**Answer format:**
```xml
<!-- Identifiable, backdoor, ATE + CATE -->
<reasoning>[d-separation analysis, adjustment set identification, counting steps]</reasoning>
<answer>status=identifiable, ATE=0.24, CATE=0.31</answer>

<!-- Identifiable, frontdoor, ATE only -->
<answer>status=identifiable, ATE=0.18</answer>

<!-- Not identifiable (structural failure) -->
<answer>status=not_identifiable, reason=latent U blocks all adjustment sets and no valid frontdoor mediator exists</answer>

<!-- Not estimable (overlap failure) -->
<answer>status=not_estimable, reason=no X=1 observations for stratum Z=2</answer>
```

**Reward:**
```
format_compliance (0.05): parseable <answer> block
status_check      (0.15): correct status string declared
                          (identifiable / not_identifiable / not_estimable): 1.0
                          wrong status: 0.0

answer_quality    (0.80):
  For identifiable (backdoor_standard, backdoor_empty, frontdoor):
    ATE accuracy:  max(0, 1 − |ATE_hat − ATE_true| / (0.3 · |ATE_true|))      [0.55]
    Special case: ATE_true = 0 → full credit iff |ATE_hat| ≤ 0.05
    CATE accuracy (backdoor_standard, backdoor_empty only):
      max(0, 1 − |CATE_hat − CATE_true| / (0.4 · |CATE_true|))                [0.15]
    Correct adjustment set / mediator identified in answer                     [0.10]

  For not_identifiable or not_estimable:
    Correct flag + reason matching the actual failure type: 1.0
    Numeric estimate produced (even if numerically close to true value): 0.0
```

---

### Flavor 3 — DAG + Observational Data → Estimate the SCM 🚧 To be implemented

**Task:** Given a DAG (structure only) and observational data, estimate the structural equations — i.e., the functional form and parameters for each node given its causal parents.

**What it tests:** Whether the model understands that structural equations are estimated by regressing each node on its *causal parents* (per the DAG), not on all correlated variables. The key causal insight is variable selection: use parents, not correlated variables.

**Design principles:**
- **Make the selection problem hard.** Include nodes with high correlation to non-parents (due to shared ancestors). Test whether the model uses the DAG to correctly select parent regressors vs. naively including all correlated variables.
- **Best tested as a multi-step tool-use task.** Model should: (1) read DAG, (2) for each node list its parents, (3) run regression via `run_python`, (4) report coefficients.
- **Focus on a single target node.** Rather than asking for the full SCM, ask "estimate the structural equation for Y specifically." This requires reading Y's parents from the DAG but NOT conditioning on Y's children or Y's non-parent ancestors. This tests DAG-reading + causal Markov condition understanding.

**Data Generation:**
```
1. Generate DAG + linear Gaussian SCM
2. Sample N=1000 rows
3. Include distractor variables: variables correlated with Y's parents via shared
   ancestors, but not direct parents of Y. These appear in the data and will tempt
   the model to include them as regressors.
4. True structural coefficients stored per node
5. Store: edges, data CSV, true_structural_equations: {node: {parent: coeff, noise_var: σ}}
```

**Benchmark Prompt Sketch:**
```
System: [Comprehensive causal inference knowledge prompt]

User: "DAG: [edges]. Data: [CSV snippet; full data via load_data].
Estimate the structural equation for node Y. Use only Y's causal parents
as regressors (per the DAG). Report coefficients for each parent and the
noise standard deviation.
<answer>Y = [coeff1]·[parent1] + [coeff2]·[parent2] + N(0, [sigma])</answer>"
```

**Evaluation:**
```
format_compliance (0.05)
answer_quality    (0.95):
  Per-parent coefficient accuracy: mean over parents of
    max(0, 1 − |β_hat − β_true| / (0.2 · |β_true|))                          [0.60]
  Correct parent set selected (binary check per node):
    1.0 if model regressed on exactly the DAG parents of Y; 0.0 otherwise     [0.35]
```

---


## Part II: Benchmark Design Principles

### Prompt Design Strategy

**All three flavors use the same comprehensive system prompt.** The model is not told which flavor it is solving or which algorithm to apply. The same prompt is used regardless of task type.

**Definitional, not prescriptive.** The system prompt provides complete causal inference knowledge — d-separation, backdoor criterion, frontdoor criterion, do-calculus, ATE/CATE definitions, identifiability conditions — but does NOT say "apply X for this task." Knowledge is provided; the model must determine which knowledge is applicable.

**Zero worked examples.** No few-shot demonstrations. The model applies knowledge to novel inputs via structural reasoning, not template-matching to demonstrated procedures.

**Identification status is always the first output.** All three flavors ask: "Is the target quantity identifiable/estimable from the given information?" This is the hardest question and the one most resistant to prompt hacking, because it requires case-by-case structural analysis of the specific graph, not application of a memorized algorithm.

**Why this design is robust to prompt optimization:**
A prompt that provides all relevant causal knowledge still requires the model to:

1. Read the specific DAG structure correctly
2. Determine which identification strategy is applicable (structural reasoning)
3. Verify that all conditions for that strategy are met on this specific instance
4. Execute the method — or correctly diagnose why no method applies

Step 3 (condition verification) is the primary source of difficulty. Near-miss structures — where conditions are almost but not quite met — require genuine structural analysis that no checklist can resolve without doing the actual graph reasoning.

---

### Advanced Identification Structures

These structures are seeded as harder variants (~10% of problems) across Flavors 1 and 2. They test whether the model correctly determines identification status when standard backdoor adjustment fails.

#### Frontdoor Criterion
```
Structure: U→X, U→Y (U unobserved), X→M→Y, no direct X→Y edge
Identification: frontdoor formula applies
  E[Y|do(X=x)] = Σ_m P(M=m|X=x) · Σ_x' P(Y|X=x', M=m) · P(X=x')
Required conditions:
  (1) No unblocked backdoor path from X to M
  (2) All backdoor paths from M to Y are blocked by X
  (3) No direct X→Y edge
```

#### Frontdoor + Direct Path → Not Identifiable
```
Structure: same as above but with additional direct X→Y edge
Result: frontdoor criterion fails (condition 3 violated);
        backdoor criterion fails (U is unobserved and blocks no valid set).
        ATE is NOT identifiable from observational data.
Model must: recognize both strategies fail and declare non-identifiability.
This is a near-miss condition failure — the structure looks almost like a
frontdoor case but one condition is violated. Tests condition verification,
not algorithm recall.
```

#### Multiple Frontdoor Mediators
```
Structure: U→X, U→Y, X→M1→M2→Y (two-mediator chain), no direct X→Y
Identification: chained frontdoor formula (more complex; tests extension of criterion)
```

#### Instrumental Variable Structure (monitoring only, not graded)
```
Structure: IV→X, U→X, U→Y, X→Y (IV is valid instrument, no IV→Y direct edge)
Status: IV identification is complex to reward automatically; include as a
        monitoring-only problem type. Log whether model identifies IV as the
        correct strategy.
```

**Common feature of advanced structures:** They require the model to check multiple conditions, any one of which could be violated. The hard cases are those where all-but-one condition holds — the model must find the specific failure, not just apply the formula.

---

### Cross-Cutting Hardeners

These design principles apply across all three flavors to ensure tasks are difficult regardless of prompt quality.

```
1. Identification status as primary output (~20% of problems require "not_identifiable"
   or "not_estimable"). Every flavor begins with identifiability/estimability diagnosis.
   Attempting a numeric estimate when the answer is "not identifiable/estimable" is
   always penalized at 0, even if the estimate happens to be numerically close.

2. ATE = 0 traps (~10% of problems)
   True effect is exactly zero due to canceling directed paths (Flavor 2A) or X
   genuinely has no causal effect on Y (no directed X→Y path). Models are biased
   toward finding a nonzero effect. Zero is rewarded at full credit if |estimate| ≤ 0.05.

3. Near-miss condition failures (~15% of problems)
   An identification strategy almost applies but one structural condition is
   violated. Examples:
   - Mediator M has an unblocked backdoor path → frontdoor fails
   - Direct X→Y edge exists → frontdoor fails
   - Required adjustment variable is unobserved → backdoor fails
   - A collider on one path is a non-collider on another → no valid set
   Model must diagnose the specific failure, not just declare "not identifiable."

4. Non-round structural coefficients (Flavors 2A, 3)
   Use β values like 0.73, 1.17, −0.84 rather than 0.5, 1.0, −1.0.
   Prevents pattern-matching to textbook values.

5. CATE questions (Flavor 2B, backdoor cases only)
   Subquestions test whether effect heterogeneity is correctly attributed to
   the right covariates. CATE(z₀) requires the model to estimate the stratum-specific
   conditional effect, not just report the marginal ATE.

6. Large DAGs (~20% of problems, all flavors)
   10–16 node DAGs. Path tracking at scale is genuinely hard regardless of
   algorithm knowledge — the number of paths grows combinatorially and manual
   enumeration is error-prone.
```

---

## Part III: Training Environment Design

### Architecture: Multi-Turn Tool Environment

Use `vf.ToolEnv` as the base. All three flavors are trained jointly via `vf.EnvGroup` with separate sub-environments per flavor. Flavor 1 is a `vf.SingleTurnEnv`; Flavors 2 and 3 are `vf.ToolEnv` instances.

For Flavors 2 and 3 (which require computation over data), expose a `run_python` tool for pandas/numpy operations and a `load_data` tool for accessing the CSV. Flavor 2 Sub-case A (linear SCM) uses the same tool set but `load_data` is not applicable — the problem is fully specified in the prompt.

**Max turns:** 8–12 (enough for: d-separation check → adjustment set confirmation → data loading → frequency counting → ATE computation → answer)

### Tools to Expose

```python
async def check_d_separation(edges: list[list[int]], X: int, Y: int, Z: list[int]) -> str:
    """Check if Z d-separates X from Y in the backdoor graph (X's outgoing edges removed).
    Args:
        edges: List of [u, v] directed edges.
        X: Treatment node index.
        Y: Outcome node index.
        Z: Proposed conditioning set (list of node indices).
    Returns: "d-separated" or "not d-separated" with a brief explanation.
    """

async def find_adjustment_sets(edges: list[list[int]], X: int, Y: int) -> str:
    """Find all minimal valid adjustment sets for (X, Y) in the given DAG.
    Returns: JSON list of minimal adjustment sets (may be empty list if none exist).
    NOTE: Available during training only as a scaffold. Removed at eval time.
    """

async def run_python(code: str) -> str:
    """Execute Python code in a persistent session and return stdout + stderr.
    Use for: frequency counting over data, path arithmetic, ATE computation.
    Available packages: pandas, numpy, statsmodels, sklearn, scipy.
    """

async def load_data(format: str = "head") -> str:
    """Load the observational dataset for this problem.
    Args:
        format: 'head' (first 10 rows), 'describe' (summary stats), 'full' (all rows as CSV)
    Not applicable for Flavor 2 Sub-case A (no data provided).
    """
```

**Design note:** `find_adjustment_sets` is a training scaffold only. Use it early in training to provide dense reward signal on Flavor 1 and Flavor 2B. After convergence on graph reasoning tasks, remove the tool or penalize its use to force the model to internalize graph reasoning.

### Reward Rubric

Each flavor has its own rubric, combined via `vf.EnvGroup`. Reward weights are per-flavor; the EnvGroup aggregates by sampling weight.

#### Flavor 1 — Adjustment Set Identification
```
format_compliance (0.10): one parseable <answer> block
status_check      (0.00): monitoring only (correct identification strategy declared)
answer_quality    (0.90): graded answer correctness
  - not_identifiable: 1.0 if predicted type is not_identifiable
  - empty: 1.0 if predicted set is {}; partial credit for valid non-empty superset
  - frontdoor: 1.0 if predicted set matches {mediator_node}
  - identifiable (backdoor): 1.0 if predicted set ∈ minimal_adjustment_sets;
                              partial credit for valid but non-minimal sets
answer_correctness (0.00): binary exact-match metric (monitoring only, weight 0)
```
See `flavor1.py` for the full reward function implementations.

#### Flavor 2 — ATE Estimation

**Sub-case A (linear SCM, no data):**
```
format_compliance (0.05): parseable <answer>ATE=...</answer> tag
status_check      (0.15): Sub-case A is always identifiable; reward 1.0 if model
                          does not declare not_identifiable / not_estimable
answer_quality    (0.80): max(0, 1 − |ATE_hat − ATE_true| / (0.1 · |ATE_true|))
                          Special case ATE_true=0: full credit iff |ATE_hat| ≤ 0.05
```

**Sub-case B (discrete data):**
```
format_compliance  (0.05): parseable <answer> block with correct field names
status_check       (0.15): correct status string
                           (identifiable / not_identifiable / not_estimable): 1.0

answer_quality     (0.80):
  identifiable (backdoor_standard, backdoor_empty, frontdoor):
    ATE accuracy: max(0, 1 − |ATE_hat − ATE_true| / (0.3·|ATE_true|))         [0.55]
    Special case ATE_true=0: full credit iff |ATE_hat| ≤ 0.05
    CATE accuracy (backdoor_standard, backdoor_empty only):
      max(0, 1 − |CATE_hat − CATE_true| / (0.4·|CATE_true|))                  [0.15]
    Correct adjustment set / mediator identified                               [0.10]

  not_identifiable or not_estimable:
    Correct flag + reason matching the actual failure type: 1.0
    Numeric estimate produced regardless: 0.0
```

**Implementation note for Sub-case A vs. B dispatch:** The `info` dict for each problem includes a `subcase` field (`"A"` or `"B"`). Reward functions must branch on this field.

#### Flavor 3 — Estimate SCM from Data
```
format_compliance (0.05): parseable <answer> tag
answer_quality    (0.95):
  Per-parent coefficient accuracy: mean over parents of
    max(0, 1 − |β_hat − β_true| / (0.2 · |β_true|))                           [0.60]
  Correct parent set selected (Y's DAG parents, no more, no less): 1.0 / 0.0  [0.35]
```

#### Monitoring Metrics (weight 0 — all flavors)
```python
async def used_graph_tool(completion) -> float:
    # Did the model call check_d_separation or find_adjustment_sets?

async def used_python_tool(completion) -> float:
    # Did the model call run_python?

async def num_tool_calls(completion) -> float:
    # Total number of tool calls (efficiency metric)

async def identified_adjustment_set_before_estimation(completion, info) -> float:
    # Flavor 2B: did the model state a valid adjustment set in its reasoning
    # before calling run_python for ATE estimation? Parsed from <reasoning> block.
```

### Curriculum Strategy

**Three-phase curriculum via TOML configs.** All phases use the same env `id = "CausalReasoningEnv"`. Curriculum is controlled by the `weights` arg. Flavors with weight 0 are not instantiated (lazy loading).

```toml
# configs/lab/phase1.toml  — Flavor 1 only
[env]
id = "CausalReasoningEnv"
args = {"weights": [1.0, 0.0, 0.0]}

# configs/lab/phase2.toml  — Flavor 1 + Flavor 2
[env]
id = "CausalReasoningEnv"
args = {"weights": [0.5, 0.5, 0.0]}

# configs/lab/phase3.toml  — all three flavors
[env]
id = "CausalReasoningEnv"
args = {"weights": [0.4, 0.4, 0.2]}
```

Weight index order: `[w_F1, w_F2, w_F3]`.

`load_environment` receives the weights and builds the EnvGroup:

```python
def load_environment(weights=None):
    if weights is None:
        weights = [1.0, 0.0, 0.0]  # default: F1 only

    all_envs = [load_flavor1, load_flavor2, load_flavor3]

    active = [(fn(), w) for fn, w in zip(all_envs, weights) if w > 0]
    return vf.EnvGroup([e for e, _ in active], weights=[w for _, w in active])
```

To advance the curriculum: monitor per-flavor reward in training logs. When the current phase has plateaued, resume from checkpoint with the next config:

```bash
prime train --config configs/lab/phase1.toml
# ... monitor F1 reward, wait for plateau ...
prime train --config configs/lab/phase2.toml --resume checkpoints/step_XXXX/
# ... monitor F1 + F2 reward ...
prime train --config configs/lab/phase3.toml --resume checkpoints/step_YYYY/
```

**Tool scaffolding → removal:** During Phase 1, `find_adjustment_sets` is available. After the model converges on Flavor 1 graph reasoning, remove the tool or penalize its use to force internalized graph structure analysis before proceeding to Phase 2.

---

## Implementation Notes

### Repository and package rename

- ✅ [2026-02-27] Rename repo: `CausalReasoningEnv_1` → `CausalReasoningEnv` (updated on GitHub; README updated)
- ✅ [2026-02-27] Remove `environments/CausalReasoningEnv_1/` (code migrated directly into `environments/CausalReasoningEnv/` — skipped intermediate `CausalReasoningFlavor1/` step)

### Target file structure

Current state and what needs to be built:

```
environments/
  CausalReasoningEnv/
    pyproject.toml                       ✅ exists
    CausalReasoningEnv.py                ✅ exists — needs weights updated to [F1, F2, F3]
    prompts.py                           ✅ exists — shared CAUSAL_KNOWLEDGE + build_system_prompt
    flavor1.py                           ✅ fully implemented
    flavor2.py                           ← REPLACE stub with full Flavor 2 implementation
    flavor3.py                           ← REPLACE stub with Flavor 3 implementation
                                           (previously flavor4.py content)
    flavor4.py                           ← DELETE (merged into new flavor2.py)
    data_generation/
      flavor1_gen.py                     ✅ fully implemented
      flavor2_gen.py                     ← REPLACE stub with full Flavor 2 generation
      flavor3_gen.py                     ← REPLACE stub with Flavor 3 generation
                                           (previously flavor4_gen.py content)
      flavor4_gen.py                     ← DELETE (merged into new flavor2_gen.py)
      generate_datasets_flavor1.py       ✅ exists
      profile_datasets_flavor1.py        ✅ exists
      upload_flavor1_datasets.py         ✅ exists
```

### Completed work (Flavor 1)

- ✅ [2026-02-27] Migrated from `CausalReasoningEnv_1`; repo and package renamed
- ✅ [2026-02-28] Shared `prompts.py` module: `CAUSAL_KNOWLEDGE`, `build_system_prompt`
- ✅ [2026-02-28] 6-bucket stratified generation in `flavor1_gen.py` (including frontdoor and not_identifiable types)
- ✅ [2026-02-28] Reward functions: `format_compliance`, `status_check`, `answer_quality`, `answer_correctness`
- ✅ [2026-03-01] Datasets generated and uploaded to HuggingFace: `irfanjamil/causal-reasoning-flavor1` (250 train / 100 eval)
- ✅ [2026-03-01] `load_flavor1()` loads directly from HuggingFace via `load_dataset()`
- [ ] Verify `flavor1.py` reward behavior (run `prime eval` spot-check)

### Work remaining

**Cleanup (do first):**
- [ ] Delete `flavor4.py` (old Flavor 4 stub — superseded by new Flavor 3)
- [ ] Delete `data_generation/flavor4_gen.py` (same reason)
- [ ] Delete `flavor3.py` (old Flavor 3 stub — merged into new Flavor 2)
- [ ] Delete `data_generation/flavor3_gen.py` (same reason)
- [ ] Update `CausalReasoningEnv.py`: change weights from 4-element to 3-element list;
      update `all_envs` to `[load_flavor1, load_flavor2, load_flavor3]`
- [ ] Update `configs/lab/phase1.toml`: weights `[1.0, 0.0, 0.0]`
- [ ] Update `configs/lab/phase2.toml`: weights `[0.5, 0.5, 0.0]` (F1 + F2)
- [ ] Update `configs/lab/phase3.toml`: weights `[0.4, 0.4, 0.2]` (all three)
- [ ] Delete `configs/lab/phase4.toml` (no longer needed)

**Flavor 2 implementation — `flavor2_gen.py` and `flavor2.py`:**
See the full Flavor 2 spec in Part I above (Sub-case A and Sub-case B sections).

`flavor2_gen.py` must implement:
- `generate_flavor2_problems(n_train, n_eval, seed)` → `(list[dict], list[dict])`
  - Sub-case A (~20%): linear SCM generator → problem_type in {standard, mediated, canceling, no_path}
  - Sub-case B (~80%): discrete CPT SCM generator → problem_type in {backdoor_standard,
    backdoor_empty, frontdoor, not_identifiable, missing_support}
  - Each problem dict must include the fields listed in the SCM generation steps above
- `build_dataset(problems, format_fn)` → HuggingFace `Dataset`

`flavor2.py` must implement:
- `format_problem_2a(...)` — renders Sub-case A problem (structural equations + DAG)
- `format_problem_2b(...)` — renders Sub-case B problem (DAG + data snippet)
- `parse_answer_2(content)` — parses `<answer>ATE=...` or `<answer>status=..., ATE=..., CATE=...`
- Reward functions: `format_compliance`, `status_check`, `answer_quality`
  (see Reward Rubric section above for weight and scoring details)
- `load_flavor2()` → `vf.ToolEnv` with the tools listed in Part III

**Flavor 3 implementation — `flavor3_gen.py` and `flavor3.py`:**
See the Flavor 3 spec in Part I above.

`flavor3_gen.py` must implement:
- `generate_flavor3_problems(n_train, n_eval, seed)` → `(list[dict], list[dict])`
- `build_dataset(problems, format_fn)` → HuggingFace `Dataset`

`flavor3.py` must implement:
- `format_problem_3(...)` — renders problem (DAG + data)
- `parse_answer_3(content)` — parses structural coefficient answer
- Reward functions: `format_compliance`, `answer_quality`
- `load_flavor3()` → `vf.ToolEnv`

### Dependencies
- ✅ `scipy`, `pandas`, `statsmodels` — in `pyproject.toml`
- ✅ `networkx`, `datasets` — in `pyproject.toml`

### Verification plan
- [ ] `python -c "from CausalReasoningEnv import load_environment; env = load_environment(); print(env)"` — confirms F1-only default loads
- [ ] `prime eval run CausalReasoningEnv -a '{"weights": [1.0, 0.0, 0.0]}' -n 10 -m openai/gpt-4.1-mini` — spot-check F1 reward
- [ ] `prime eval run CausalReasoningEnv -a '{"weights": [0.0, 1.0, 0.0]}' -n 20 -m openai/gpt-4.1-mini` — spot-check F2 reward
- [ ] Manually inspect 5 Sub-case A problems: verify ATE matches path-tracing by hand
- [ ] Manually inspect 5 Sub-case B problems per type: verify true_ATE matches CPT enumeration
- [ ] Check reward edge cases: ATE=0, not_estimable, frontdoor formula, missing strata
- [ ] Confirm `vf.EnvGroup` routes to correct sub-environment and aggregates metrics correctly
