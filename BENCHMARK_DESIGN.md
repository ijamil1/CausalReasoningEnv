# Causal Reasoning Benchmark & Training Environment Design

## Context

The existing `CausalReasoningEnv_1` benchmarks only one causal task — minimal adjustment set identification — and does so with heavy scaffolding (detailed ICL examples, step-by-step instructions). The goal is to expand this into a full causal reasoning benchmark covering four distinct "flavors" of tasks that together test end-to-end causal analysis competency, then design a multi-turn tool-use training environment with a principled reward rubric to train models toward this benchmark.

---

## Part I: Benchmark Analysis — Four Flavors

### Flavor 1 — Adjustment Set Identification (ALREADY BUILT)

**Task:** Given a DAG with treatment X and outcome Y, determine: (a) Is ATE identifiable by backdoor adjustment from the observed variables? (b) If yes, give the minimal valid adjustment set Z. If no, explain which structural condition prevents it.

**What it tests:** d-separation, backdoor criterion, collider logic, non-descendant constraint, and crucially — the ability to diagnose *when* no valid adjustment set exists.

**Critiques & Pointers:**
- **Hand-holding is heavy.** The current system prompt teaches the backdoor criterion step-by-step. For a *benchmark*, this measures "can the model follow instructions" not "does it understand causal graphs." The benchmark variant should strip the ICL examples to definitional knowledge only — no worked examples.
- **Multiple valid minimal sets.** `find_minimal_d_separator` returns *one* minimal set, but other minimal sets may exist. Grading on exact match against a single gold answer will penalize correct alternative answers. Fix: at generation time, enumerate all minimal adjustment sets and store the full set of valid answers.
- **Jaccard reward can mislead training.** If a model returns `{all parents}`, it gets partial Jaccard credit but is wrong on minimality. Add a **minimality penalty**: score is 0 if `|predicted| > |gold|` and the predicted set is valid.
- **The `valid_adjustment_set` reward is currently weighted 0.0.** For training it should carry some weight to teach the validity constraint before minimality.
- **Collider chain trap.** A node C may be a collider on path P1 (conditioning on C blocks P1) but a non-collider on path P2 (conditioning on C opens P2). These cases directly test collider logic — the model must avoid C or find an alternative set that handles both paths simultaneously.
- **No-set and empty-set cases.** ~20% of problems should require "no valid adjustment set exists" or "no adjustment needed" as the correct answer. Models are biased toward producing a non-empty set; these cases test whether the model correctly diagnoses the graph rather than finding something to adjust for.

**Problem Types (five stratified buckets — implemented in `flavor1_gen.py`):**
```
identifiable_standard (~42%):
  Non-empty minimal adjustment set; all observed parents of X are required
  (|min_set| ≥ |observed_parents(X)|).

identifiable_ancestor (~14%):
  Non-empty set; |min_set| < |observed_parents(X)|; redundancy because a
  dropped observed parent has an ancestor already in min_set.

identifiable_collider (~14%):
  Non-empty set; |min_set| < |observed_parents(X)|; redundancy via a
  collider structure on the backdoor path through that parent.
  NOTE: type (e) collider-chain problems from the original design are
  subsumed here — they are a subset of identifiable_collider.

empty (~15%):
  Empty minimal adjustment set.  X and Y are already d-separated in the
  backdoor graph (all paths blocked by unconditioned colliders).  Requires
  ≥1 undirected path in G_bd (result is structural, not trivial).

not_identifiable (~15%):
  No valid observed adjustment set exists.  A synthetic latent node L is
  added with direct edges L→X and L→Y.  X→Y is a direct edge and the ONLY
  causal path (no X→M→Y mediator — ruling out the front-door criterion).
  Model must state ATE is not identifiable and explain the blocking failure.
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

**Reward (current implementation in `flavor1.py`):**
```
- format_reward (0.05): exactly one parseable <answer> block
- valid_adjustment_set (0.15): predicted set is a valid (not necessarily
  minimal) adjustment set
- correct_adjustment_set (0.80): predicted set exactly matches the minimal
  adjustment set
NOTE: reward functions need updating to handle the three identifiability
statuses and the new not_identifiable / empty answer formats.
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

### Flavor 2 — DAG + Observational Data → Estimate ATE via Counting

**Task:** Given a DAG and observational data: (a) determine whether ATE is estimable from the available data, (b) if yes, estimate ATE = E[Y(do(X=1))] - E[Y(do(X=0))] using nonparametric counting, and (c) estimate CATE for specified covariate values.

**What it tests:** Identification status diagnosis; nonparametric backdoor estimation; recognizing data limitations (unobserved variables, missing treatment support, auxiliary variables); CATE estimation and effect heterogeneity.

**Key design intent:** The model is given the DAG and data only — **not the SCM or its functional form**. The underlying SCM is nonlinear. The correct approach is to apply the backdoor formula nonparametrically: estimate E[Y | X=x, Z=z] by grouped averaging over the data, then marginalize over P(Z=z). Step 0 is always to check whether ATE is even estimable — some problems require the model to flag non-identifiability rather than attempt estimation.

**Variable types:** All variables are discrete.
- **X: binary (0/1)** — ATE = E[Y|do(X=1)] - E[Y|do(X=0)] is unambiguous with binary treatment.
- **Z (adjustment covariates): discrete (multi-category)** — counting is exact; no binning or kernel methods required.
- **Y: binary** — P(Y=1|X=x, Z=z) estimated directly by grouped frequency.

Using discrete variables means OLS is model-misspecified, so the model gains nothing by assuming linearity; stratified counting is exact.

**Problem Types (stratified sampling at generation time):**
```
(a) Standard: valid adjustment set exists and all required variables are observed;
    enough data for all strata. Model estimates ATE and CATE.

(b) Multi-dimensional Z: minimal adjustment set has ≥2 variables, each with 3+ categories
    → ≥9 strata; some strata sparse (< 5 obs). Model must handle sparse strata correctly
    (report estimate with caveat, or flag insufficient data for that stratum).

(c) Unobserved adjustment variable: one variable in the minimal adjustment set is
    absent from the data CSV. Model should:
      - First check if an alternative valid adjustment set exists using observed variables.
      - If no alternative exists: flag ATE is not estimable. Do NOT compute a biased estimate.

(d) Missing treatment support: all rows have X=0 (or X=1), so P(Y|X=1, Z=z) cannot be
    estimated from data. Model must flag ATE cannot be computed due to lack of overlap.

(e) Auxiliary variables: data CSV contains 1–3 variables not present in the DAG,
    labeled "auxiliary_1", "auxiliary_2", etc. Model must use the DAG as a filter
    and ignore these variables entirely. Adjusting for auxiliaries is penalized.
```

**Critiques & Pointers:**
- **Two failure modes.** Log separately: (i) did the model correctly diagnose identifiability status? (ii) given correct status, did it compute ATE accurately? These are independent sources of failure.
- **Model rejection risk.** Mitigated by using discrete variables (counting is exact) and by framing the prompt around stratified averaging without parametric assumptions.
- **Confounding is required for estimable problems.** Every type (a)/(b)/(e) problem must have at least one active backdoor path. The naive marginal difference must differ from true ATE by ≥15%.
- **Sparse strata are a genuine test.** For type (b), some Z-combinations will have very few observations. The model must handle this gracefully — either reporting the estimate with a reliability caveat or noting the strata are too sparse to estimate reliably.

**Data Generation:**
```
1. Sample problem type according to stratification weights above
2. Generate DAG + discrete nonlinear SCM:
   - Root nodes: Bernoulli or categorical
   - Non-root nodes: P(V=1|parents) = sigmoid(Σ β_i · parent_i + β_ij · parent_i · parent_j)
3. Sample N=2000 rows
4. Apply type-specific data modification:
   - Type (b): ensure ≥2 Z variables with ≥3 categories each; verify sparse strata exist
   - Type (c): remove one required adjustment variable column from the CSV
   - Type (d): filter data to keep only rows where X=0 (or X=1)
   - Type (e): generate 1–3 independent Bernoulli/categorical variables; append with
               "auxiliary_N" column names; tell model these are auxiliary
5. True ATE/CATE: exact enumeration over the true interventional distribution
   ATE = Σ_z [P(Y=1|X=1,Z=z) - P(Y=1|X=0,Z=z)] · P(Z=z)
   CATE(z0) = P(Y=1|X=1,Z=z0) - P(Y=1|X=0,Z=z0)
6. Store: edges, data CSV, X, Y, problem_type, identifiability_status,
   true_ATE (null for types c/d), true_CATE_cases, adjustment_set
   (SCM parameters NOT stored — model never sees them)
```

**Evaluation:**
```
- Correct identifiability_status: 0.20
- For estimable (types a/b/e):
    ATE numeric accuracy: max(0, 1 - |ATE_hat - ATE_true| / (0.5 · |ATE_true|))  [0.50]
    CATE accuracy: same formula, 20% tolerance  [0.15]
    Correct adjustment set used: binary check  [0.15]
- For not-estimable (types c/d):
    Correct flag + correct reason: 0.80
    Attempting numeric estimate anyway: 0.0 regardless of proximity to true value
```

**Benchmark Prompt Sketch:**
```
System: [Comprehensive causal inference knowledge prompt — see Prompt Design Strategy section]

User: "DAG: [edges]. Treatment X=[x] (binary), Outcome Y=[y] (binary).
Observed variables in data: [list — may omit some DAG nodes].
[If type (e):] Also present but not in the DAG: auxiliary_1, auxiliary_2 (ignore these).
Data (first 5 rows shown; full data available via load_data tool): [CSV snippet]

(a) Is ATE estimable from this data and DAG? State yes/no and why.
(b) If yes: estimate ATE = E[Y(do(X=1))] - E[Y(do(X=0))]. Use stratified counting
    over the adjustment set — do not assume a parametric functional form.
(c) If yes: estimate CATE for Z = [z_values].
(d) If yes: what is CATE(Z=z1) - CATE(Z=z2)?
<answer>estimable=[yes/no], reason=[...], ATE=[...], CATE=[...]</answer>"
```

---

### Flavor 3 — DAG + Fully Specified SCM → Compute ATE

**Task:** Given a DAG and complete structural equations (functional form + parameter values + noise distributions), compute ATE = E[Y|do(X=1)] - E[Y|do(X=0)].

- **Linear SCMs (75%):** Produce the exact numeric ATE via algebraic reasoning (Wright's path-tracing).
- **Nonlinear SCMs (25%):** Produce a substituted symbolic formula for ATE that the grader can evaluate numerically.

**What it tests:** Understanding of the do() operator — specifically that do(X=x) mutilates the graph by removing all edges into X and fixing X=x. Tests whether the model truly understands causal vs. observational conditioning, and whether it can derive ATE from a fully specified SCM without data.

**Differentiation from Flavor 2:**

| | Flavor 2 | Flavor 3 |
|---|---|---|
| Given | DAG + data (no SCM) | DAG + full SCM (no data) |
| Method | Nonparametric counting from data | Analytical derivation from SCM |
| Linear output | Numeric ATE | Numeric ATE |
| Nonlinear output | Numeric ATE | **Substituted symbolic formula** |

**Critiques & Pointers:**
- **Linear-Gaussian shortcut.** For linear Gaussian SCMs, capable models may pattern-match to "extract the direct path coefficient." Include DAGs with confounders + mediators where this is wrong. The ATE via do() is the sum of direct path coefficients along all X→Y directed paths (after mutilation); the naive OLS coefficient on X picks up confounding bias. The core test: model correctly applies path-tracing, not OLS.
- **Confusing E[Y|X=x] with E[Y|do(X=x)].** This is the central conceptual test. All problems must have confounding so these differ.
- **Canceling paths.** Explicitly include ~15% of problems where X→Y (coeff +β) and X→M→Y (indirect, coeff −β·γ) nearly cancel, giving ATE ≈ 0. Both individual paths are non-trivially non-zero (|β| ≥ 0.5, |β·γ| ≥ 0.4). Models that just report the direct coefficient will be wrong; path-tracing is required.
- **Nonlinear formula output.** The model writes the ATE as an explicit expectation with SCM functions substituted in — e.g., `ATE = E_{Z1~N(0,1)}[tanh(2·1 + 0.5·Z1) - tanh(2·0 + 0.5·Z1)]`. This is distinct from the abstract identification formula (which Flavor 1 already tests) and is numerically evaluable by the grader.
- **Counterfactuals extension.** An advanced variant: E[Y_{X=1} | X=0, Y=y_obs] — requires twin network / abduction step. Optional hard tier for future work.

**Data Generation:**
```
1. Generate DAG with confounders (at least one active backdoor path)
2. Parameterize SCM:
   - 75% linear: V = Σ(β_i · parent_i) + N(0, σ_V), β_i ~ Uniform(0.2, 1.5)
     (use non-round coefficients: e.g., 0.73, 1.17, −0.84)
   - 25% nonlinear: V = tanh(Σ(β_i · parent_i)) + N(0, 0.2) or quadratic mixtures
3. ~15% of problems: enforce canceling paths (ATE target ≈ 0)
   - Set β_direct and β_indirect such that their sum is within ±0.05 of 0
   - Verify |β_direct| ≥ 0.5 and |β_indirect| ≥ 0.4 (both paths non-trivial)
   - Store has_canceling_paths: bool in info dict
4. Linear ATE: via Wright's path-tracing (sum of directed path products X→Y)
5. All ATE ground truth: simulation (1M samples under do(X=0) and do(X=1))
6. True CATE(Z1=z1): condition on Z1=z1 in the intervention distribution
7. Store: edges, SCM equations as text, true_ATE, true_CATE_cases,
   scm_type ("linear" | "nonlinear"), has_canceling_paths
```

**Benchmark Prompt Sketches:**

*Linear version — numeric ATE + CATE:*
```
System: [Comprehensive causal inference knowledge prompt — see Prompt Design Strategy section]

User: "DAG: [edges].
Structural equations:
  Z1 = N(0, 1)
  Z2 = 0.73·Z1 + N(0, 0.5)
  X  = 1.17·Z2 + N(0, 0.3)   [do(X=x): this equation is replaced by X=x]
  Y  = 0.84·X − 0.81·M + N(0, 0.4)   [M is a mediator on X→M→Y]
  M  = 0.97·X + N(0, 0.2)

(a) Compute ATE = E[Y|do(X=1)] - E[Y|do(X=0)]
(b) Compute CATE for Z2 = 1.0
<answer>ATE=[...], CATE=[...]</answer>"
```

*Nonlinear version — substituted symbolic formula + CATE:*
```
System: [Comprehensive causal inference knowledge prompt — see Prompt Design Strategy section]

User: "DAG: [edges].
Structural equations:
  Z1 = N(0, 1)
  X  = ... [do(X=x): this equation is replaced by X=x]
  Y  = tanh(2.0·X + 0.5·Z1) + N(0, 0.1)

(a) Write ATE = E[Y|do(X=1)] - E[Y|do(X=0)] as an explicit expectation expression
    with the structural equations substituted in.
(b) Write CATE for Z1 = 0.5 as an explicit expression.
<answer>ATE=[formula], CATE=[formula]</answer>"
```

---

### Flavor 4 — DAG + Observational Data → Estimate the SCM

**Task:** Given a DAG (structure only) and observational data, estimate the structural equations — i.e., the functional form and parameters for each node given its causal parents.

**What it tests:** Whether the model understands that structural equations are estimated by regressing each node on its *causal parents* (per the DAG), not on all correlated variables.

**Critiques & Pointers:**
- **Risk of triviality.** If the functional form is given (linear), this reduces to "run OLS for each node on its parents" — essentially a statistics exercise, not a causal reasoning test. The causal insight is the variable selection step (use parents, not correlated variables).
- **Make the selection problem hard.** Include nodes with high correlation to non-parents (due to shared ancestors). Test whether the model uses the DAG to correctly select parent regressors vs. naively including all correlated variables.
- **This flavor best tested as a multi-step tool-use task.** Model should: (1) read DAG, (2) for each node list its parents, (3) run regression, (4) report coefficients.
- **Evaluation.** For each node, compute mean relative error of estimated structural coefficients. Also check: did the model regress on the correct set of parents? (Binary check per node — selection accuracy.)
- **Could be reframed more interestingly.** Instead of "estimate the full SCM," ask "estimate the structural equation for X specifically" — which requires knowing X's parents from the DAG but NOT adjusting for X's children or X's non-parent ancestors. This tests DAG-reading + causal Markov condition understanding.
- **Consider merging with Flavor 2.** Flavor 4 (estimate SCM) is a natural precursor to Flavor 2 (use SCM to compute ATE). Could present as a two-step problem.

**Data Generation:**
```
1. Generate DAG + linear Gaussian SCM (same as Flavor 2/3)
2. Sample N=1000 rows
3. Include distractor variables: report correlations in data that don't correspond to
   parent relationships
4. True structural coefficients stored per node
5. Store: edges, data CSV, true_structural_equations: {node: {parent: coeff, noise_var: σ}}
```

**Benchmark Prompt Sketch:**
```
System: "You are a causal inference expert. The DAG tells you each node's
causal parents. Structural equations are: V = Σ(β_i · parent_i) + ε, ε ~ N(0, σ).
Estimate structural coefficients by regressing each node on its DAG parents only."

User: "DAG: [edges]. Data: [CSV].
Estimate the structural equation for node Y. Report: coefficients for each parent
and the noise standard deviation.
<answer>Y = [coeff1]·[parent1] + [coeff2]·[parent2] + N(0, [sigma])</answer>"
```

---


## Part II: Benchmark Design Principles

### Prompt Design Strategy

**All four flavors use the same comprehensive system prompt.** The model is not told which flavor it is solving or which algorithm to apply. The same prompt is used regardless of task type.

**Definitional, not prescriptive.** The system prompt provides complete causal inference knowledge — d-separation, backdoor criterion, frontdoor criterion, do-calculus, ATE/CATE definitions, identifiability conditions — but does NOT say "apply X for this task." Knowledge is provided; the model must determine which knowledge is applicable.

**Zero worked examples.** No few-shot demonstrations. The model applies knowledge to novel inputs via structural reasoning, not template-matching to demonstrated procedures.

**Identification status is always the first output.** All flavors ask: "Is the target quantity identifiable/estimable from the given information?" This is the hardest question and the one most resistant to prompt hacking, because it requires case-by-case structural analysis of the specific graph, not application of a memorized algorithm.

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

These design principles apply across all four flavors to ensure tasks are difficult regardless of prompt quality.

```
1. Identification status as primary output (~20% of problems require "not estimable")
   Every flavor begins with identifiability diagnosis. "ATE is not estimable" is
   always a valid answer. Attempting a numeric estimate when the answer is
   "not estimable" is always penalized, even if the estimate happens to be close.

2. ATE = 0 and CATE = 0 traps (~10% of problems)
   True effect is exactly zero due to canceling directed paths, or X genuinely
   has no causal effect on Y. Models are biased toward finding a nonzero effect.
   Zero is rewarded at full credit if |estimate| ≤ 0.05.

3. Near-miss condition failures (~15% of problems)
   An identification strategy almost applies but one structural condition is
   violated. Examples:
   - Mediator M has an unblocked backdoor path → frontdoor fails
   - Direct X→Y edge exists → frontdoor fails
   - Required adjustment variable is unobserved → backdoor fails
   - A collider on one path is a non-collider on another → no valid set
   Model must diagnose the specific failure, not just declare "not identifiable."

4. Auxiliary variables in data (Flavors 2, 4)
   1–3 variables in the data CSV labeled "auxiliary_N" are not present in the DAG.
   The model must use the DAG as a filter; including auxiliaries in adjustment
   is penalized even if it produces an accurate numeric result.

5. Non-round structural coefficients (Flavors 3, 4)
   Use β values like 0.73, 1.17, −0.84 rather than 0.5, 1.0, −1.0.
   Prevents pattern-matching to textbook values.

6. CATE and difference-in-CATE (Flavors 2, 3)
   Optional subquestions test whether effect heterogeneity is correctly attributed
   to the right covariates. "What is CATE(Z=z1) − CATE(Z=z2)?" requires the model
   to compute and compare conditional effects, not just report a marginal average.

7. Large DAGs (~20% of problems, all flavors)
   12–16 node DAGs. Path tracking at scale is genuinely hard regardless of
   algorithm knowledge — the number of paths grows combinatorially and manual
   enumeration is error-prone.
```

---

## Part III: Training Environment Design

### Architecture: Multi-Turn Tool Environment

Use `vf.ToolEnv` (or `vf.StatefulToolEnv` if code execution state needs to persist across turns) as the base. All four flavors can be trained jointly via `vf.EnvGroup` with separate sub-environments per flavor.

For Flavors 2 and 4 (which require computation over data), embed a Python REPL via `vf.PythonEnv` or expose a `run_python` tool backed by Prime Sandboxes. For Flavors 1 and 3 (graph-theoretic or analytic), a simpler `vf.ToolEnv` suffices.

**Max turns:** 8–12 (enough for plan → adjust_set_check → regression → ATE_calc → finalize)

### Tools to Expose

```python
async def check_d_separation(edges: list[list[int]], X: int, Y: int, Z: list[int]) -> str:
    """Check if Z d-separates X from Y in the given DAG (with X's outgoing edges removed).
    Args:
        edges: List of [u, v] directed edges.
        X: Treatment node.
        Y: Outcome node.
        Z: Proposed conditioning set.
    Returns: "d-separated" or "not d-separated" with explanation.
    """

async def find_adjustment_sets(edges: list[list[int]], X: int, Y: int) -> str:
    """Find all minimal valid adjustment sets for X → Y in the given DAG.
    Returns: JSON list of minimal adjustment sets and whether they exist.
    NOTE: Available during training only; removed for benchmark evaluation.
    """

async def get_descendants(edges: list[list[int]], node: int) -> str:
    """Return all descendants of a node in the DAG (for the no-descendant check).
    """

async def run_python(code: str) -> str:
    """Execute Python code and return stdout + stderr.
    Use for: OLS regression (statsmodels/sklearn), ATE estimation, data manipulation.
    pandas, numpy, statsmodels, sklearn are available.
    """

async def load_data(format: str = "head") -> str:
    """Load the observational dataset for this problem.
    Args:
        format: 'head' (first 10 rows), 'describe' (summary stats), 'full' (all rows as CSV)
    """
```

**Design note:** Don't expose `find_adjustment_sets` during evaluation (benchmark mode) — it gives away the answer. Use it during training as a scaffold to get dense reward signal early, then phase it out via a curriculum (or apply a reward penalty for using it).

### Reward Rubric

Each flavor gets its own rubric, combined via `vf.EnvGroup`. Within each flavor, the rubric has 4 layers:

#### Layer 1: Format Compliance (weight 0.05 across all flavors)
```python
async def format_compliance(completion) -> float:
    # Check: exactly one <answer>...</answer> tag, parseable content
    return 1.0 if parse_answer(completion) is not None else 0.0
```

#### Layer 2: Identification Status + Intermediate Process (weight 0.20)
- **All flavors:** Correct identifiability/estimability status declaration (prerequisite).
  If status is wrong, total score is capped at 0.20 regardless of numeric accuracy.
- **Flavor 1:** Was the proposed adjustment set valid (even if not minimal)?
- **Flavor 2:** Did the model use a valid adjustment set over observed variables?
- **Flavor 3:** Did the model correctly identify which paths survive the do() mutilation?
- **Flavor 4:** Did the model correctly identify each node's parents from the DAG?

```python
async def identifiability_status_check(completion, info) -> float:
    # Check model's declared status against info["identifiability_status"]
    # Returns 1.0 for correct, 0.0 for wrong

async def validity_check(completion, info, state) -> float:
    # Re-use valid_adjustment_set logic from existing code
    # Extended per flavor — returns partial credit for correct intermediate steps
```

#### Layer 3: Answer Correctness (weight 0.80)
- **Flavor 1 (identifiable):** Exact match against any element of `all_minimal_adjustment_sets` = 1.0. Valid but non-minimal = 0.25. Jaccard partial credit with minimality penalty.
- **Flavor 1 (not identifiable):** Correct diagnosis + structural explanation = 1.0. Produces a set anyway = 0.0.
- **Flavor 2 (estimable):** ATE relative error: `max(0, 1 - |ATE_hat - ATE_true| / (0.5·|ATE_true|))` [0.50]; CATE same formula 20% tolerance [0.15]; correct adjustment set used [0.15].
- **Flavor 2 (not estimable):** Correct flag + correct reason = 1.0. Numeric estimate attempted anyway = 0.0.
- **Flavor 3 (linear):** Relative error formula, tight tolerance (±1%). ATE [0.60] + CATE [0.20] where applicable.
- **Flavor 3 (nonlinear):** Grader evaluates substituted symbolic formula via Monte Carlo (1M samples), ±5% tolerance. ATE formula [0.60] + CATE formula [0.20] where applicable.
- **Flavor 4:** Mean relative error across all structural coefficients [0.60]; per-node parent selection accuracy [0.20].

#### Layer 4: Monitoring Metrics (weight 0 — observability only)
```python
async def used_graph_tool(completion) -> float:
    # Did the model call check_d_separation or find_adjustment_sets?

async def used_python_tool(completion) -> float:
    # Did the model call run_python?

async def num_tool_calls(completion) -> float:
    # Total number of tool calls (efficiency metric)

async def identified_correct_adjustment_set_before_estimation(completion, info) -> float:
    # For Flavor 2: did the model log a valid adjustment set in its reasoning
    # before running regression? Parsed from <reasoning> block.
```

### Reward Shaping Strategy

**Curriculum by flavor difficulty:**
1. Phase 1 (warm-up): Flavor 1 only. Establish graph reasoning and format compliance.
2. Phase 2: Add Flavor 3. Model learns do() operator with exact algebraic answers.
3. Phase 3: Add Flavor 2 (data → counting ATE). Tool use + nonparametric estimation pipeline.
4. Phase 4: Add Flavor 4 (data → SCM). Full end-to-end: parent selection + regression.

**Advancing the curriculum — manual phase transitions via TOML args:**

All phases use the same single environment (`CausalReasoningEnv_2`) and the same `env_id`.
Curriculum is controlled by passing different `weights` args to `load_environment()` via
separate TOML configs. Sub-environments with weight 0 are not instantiated (lazy loading),
so only the active flavors load their datasets.

```toml
# configs/vf-rl/phase1.toml
[env]
id = "CausalReasoningEnv_2"
args = {"weights": [1.0, 0.0, 0.0, 0.0]}   # F1 only

# configs/vf-rl/phase2.toml
[env]
id = "CausalReasoningEnv_2"
args = {"weights": [0.4, 0.6, 0.0, 0.0]}   # F1 + F3

# configs/vf-rl/phase3.toml
[env]
id = "CausalReasoningEnv_2"
args = {"weights": [0.3, 0.4, 0.3, 0.0]}   # F1 + F3 + F2

# configs/vf-rl/phase4.toml
[env]
id = "CausalReasoningEnv_2"
args = {"weights": [0.25, 0.3, 0.25, 0.2]} # all four flavors
```

`load_environment` receives the weights and builds the EnvGroup accordingly:

```python
def load_environment(weights=None):
    if weights is None:
        weights = [1.0, 0.0, 0.0, 0.0]  # default: F1 only

    all_envs = [load_flavor1, load_flavor3, load_flavor2, load_flavor4]
    # order: [F1, F3, F2, F4] — matches weight index

    active = [(fn(), w) for fn, w in zip(all_envs, weights) if w > 0]
    return vf.EnvGroup([e for e, _ in active], weights=[w for _, w in active])
```

To advance the curriculum: monitor per-flavor reward in training logs, decide manually
when the current phase has plateaued, then resume from checkpoint with the next config:

```bash
prime train --config configs/vf-rl/phase1.toml
# ... monitor F1 reward, wait for plateau ...
prime train --config configs/vf-rl/phase2.toml --resume checkpoints/step_XXXX/
```

**Tool scaffolding → removal:** Early training, `find_adjustment_sets` tool available. After convergence on Flavor 1, remove tool (or penalize its use), forcing internalized graph reasoning.

**Process reward option:** Use a lightweight LLM judge (e.g., `gpt-4.1-mini` via `vf.JudgeRubric`) to score reasoning quality: "Did the model correctly identify the backdoor paths? Did it correctly apply the do() operator?" Weight 0.1.

---

## Implementation Notes

### Repository and package rename

- ✅ [2026-02-27] Rename repo: `CausalReasoningEnv_1` → `CausalReasoningEnv` (updated on GitHub; README updated)
- ✅ [2026-02-27] Remove `environments/CausalReasoningEnv_1/` (code migrated directly into `environments/CausalReasoningEnv/` — skipped intermediate `CausalReasoningFlavor1/` step)

### Target file structure

✅ [2026-02-27] File structure created:

```
environments/
  CausalReasoningEnv/                    ← new main package
    pyproject.toml                       ✅ created
    CausalReasoningEnv.py                ← load_environment() → EnvGroup  ✅ created
    flavor1.py                           ← Flavor1Env + load_flavor1()  ✅ created (migrated from CausalReasoningEnv_1)
    flavor2.py                           ← Flavor2Env + load_flavor2()  ✅ stub created
    flavor3.py                           ← Flavor3Env + load_flavor3()  ✅ stub created
    flavor4.py                           ← Flavor4Env + load_flavor4()  ✅ stub created
    data_generation/
      flavor1_gen.py                     ✅ created (generation logic ported from original)
      flavor2_gen.py                     ✅ stub created
      flavor3_gen.py                     ✅ stub created
      flavor4_gen.py                     ✅ stub created
```

### Migration plan for CausalReasoningFlavor1

- ✅ [2026-02-27] Port `_make_dag`, `_try_sample_problem`, `generate_stratified_dag_problems` → `data_generation/flavor1_gen.py`
- ✅ [2026-02-27] Port `Flavor1Env` class + `load_flavor1()` → `flavor1.py`
- ✅ [2026-02-27] Port `_render_dag_b64`, `format_problem`, `valid_adjustment_set`, `parse_answer` → inline in `flavor1.py`
- ✅ [2026-02-27] Delete `environments/CausalReasoningEnv_1/`
- [ ] Verify `flavor1.py` produces equivalent reward behavior to the original (run `prime eval` spot-check)

### New files to create
- ✅ [2026-02-27] `environments/CausalReasoningEnv/CausalReasoningEnv.py`
- ✅ [2026-02-27] `environments/CausalReasoningEnv/pyproject.toml`
- ✅ [2026-02-27] `environments/CausalReasoningEnv/flavor1.py` through `flavor4.py`
- ✅ [2026-02-27] `environments/CausalReasoningEnv/data_generation/flavor1_gen.py` through `flavor4_gen.py`
- ✅ [2026-02-27] `configs/lab/phase1.toml` — F1 only (`weights: [1.0, 0.0, 0.0, 0.0]`)
- ✅ [2026-02-27] `configs/lab/phase2.toml` — F1 + F3 (`weights: [0.4, 0.6, 0.0, 0.0]`)
- ✅ [2026-02-27] `configs/lab/phase3.toml` — F1 + F3 + F2 (`weights: [0.3, 0.4, 0.3, 0.0]`)
- ✅ [2026-02-27] `configs/lab/phase4.toml` — all flavors (`weights: [0.25, 0.3, 0.25, 0.2]`)
- Note: configs placed in `configs/lab/` (not `configs/vf-rl/`) — update path references if needed

### New dependencies needed
- [ ] `scipy`, `pandas`, `statsmodels` — data generation and estimation (needed for Flavors 2–4)
- [ ] `sympy` — optional symbolic SCM manipulation (Flavor 3 linear case)

### Verification plan
- [ ] `python -c "from CausalReasoningEnv import load_environment; env = load_environment(); print(env)"` — confirms environment loads
- [ ] `prime eval run CausalReasoningEnv -a '{"weights": [1.0, 0.0, 0.0, 0.0]}' -n 10 -m openai/gpt-4.1-mini` — spot-check F1 reward matches original
- [ ] `prime eval run CausalReasoningEnv -n 10 -m openai/gpt-4.1-mini` — spot-check reward distributions per flavor (all phases)
- [ ] Manually inspect 5 problems per flavor: verify ground truth ATEs match simulation, verify prompts are parseable
- [ ] Check reward function edge cases: empty adjustment set, unparseable answers, zero-variance outcomes
- [ ] Confirm `vf.EnvGroup` routes to correct sub-environment and aggregates metrics correctly
