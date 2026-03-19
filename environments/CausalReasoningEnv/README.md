# CausalReasoningEnv

Multi-turn causal reasoning benchmark and RL training environment, built with Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework.

The environment trains models to identify causal effects via backdoor adjustment, frontdoor criterion, or instrumental variables, and compute ATE/LATE via probability query tools over discrete CPT-based DAGs.

---

## Design

### Two-Phase Rollout

**Phase 1 — Declaration + Tools (Turn 1):** The model reasons about the DAG structure, calls `declare(method, nodes)` to commit to an identification method and relevant node set, and makes all needed probability tool calls in the same response. A global cap of 4 total tool calls is enforced (declare + up to 3 probability tools). The rollout terminates early if: `declare` is missing, arguments are unparseable, the method is unknown, no probability calls are made, the parallel limit is exceeded, or the declared method/set is invalid.

**Phase 2 — Answer (Turn 2):** The model receives tool results and writes a final answer. For backdoor/frontdoor: `<answer>ATE=X.XXXX</answer>`. For IV: `<answer>LATE=X.XXXX</answer>`.

### Problem Types

| Type | Fraction | Identification strategy |
|------|----------|------------------------|
| `backdoor_standard` | ~35% | Non-empty minimal adjustment set Z; verified no frontdoor |
| `backdoor_empty` | ~15% | Empty adjustment set (no backdoor confounding); verified no frontdoor |
| `frontdoor` | ~40% | Latent confounder; valid mediator set M (possibly multi-node); verified no backdoor |
| `iv` | ~10% | No backdoor/frontdoor; fresh exogenous IV node Z→X + latent confounder L→X,Y |

Each problem has **exactly one valid identification method** by construction (mutual exclusivity enforced at generation time).

The stored `minimal_set` holds the minimal adjustment set (backdoor), minimal mediator set (frontdoor), or single-element instrument list (IV).

### Reward Rubric

| Component | Weight | Description |
|-----------|--------|-------------|
| `format_compliance` | 0.07 | 0.0 on Turn-1 format violations, answer in Turn 1, or missing answer after valid declaration |
| `method_validity` | 0.145 | 1.0 if declared method matches the problem's identification method |
| `set_validity` | 0.145 | 1.0 if declared node set correctly identifies the effect; gated on `method_validity` |
| `minimality` | 0.0 | 1.0 if set equals `minimal_set`; k/\|declared\| if valid superset; gated on both validity scores (metrics only, no reward weight) |
| `ate_accuracy_binary` | 0.50 | 1.0 if \|answer − true target\| < 0.001; works for ATE and LATE |
| `process_correctness` | 0.14 | Graded score for correct intermediate computation steps |

### Tools

```
declare(method, nodes)
  Declare identification method and the relevant node set. REQUIRED in Turn 1.
  method: "backdoor", "frontdoor", or "iv". Example: "backdoor"
  nodes:  adjustment set (backdoor), mediator set (frontdoor), or [instrument] (iv).
          Pass node IDs as integers. Example: [1, 3]

marginal(variables)
  Returns the full joint PMF P(V1, V2, ...) for all value combinations.
  Input:  variables — list of node IDs as integers, e.g. [2, 3]
  Output: P(node2=0, node3=-1) = 0.1234 ...

conditional(query, given)
  Returns P(query | given) for all strata of the conditioning variables.
  Input:  query — list of node IDs as integers, e.g. [4]
          given — list of node IDs as integers, e.g. [0, 2]
  Output: P(node4=0 | node0=0, node2=-1) = 0.7234 ...
```

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

## Quickstart

### Run eval

```bash
prime eval run CausalReasoningEnv
prime eval run CausalReasoningEnv -m openai/gpt-4.1-mini -n 50 -r 1
```

### Regenerate dataset

```bash
cd environments/CausalReasoningEnv
python data_generation/generate_datasets.py --n 1000 --save-local
python data_generation/generate_datasets.py --n 1000 --push-hub
```

### Train

```bash
prime train --config configs/lab/phase1.toml
```

---

## File Structure

```
CausalReasoningEnv/
  CausalReasoningEnv.py           # load_environment() entry point
  env.py                          # CausalATEEnv — tools, rubric, stop logic
  prompts.py                      # SYSTEM_PROMPT — declaration format, tool docs
  data_generation/
    gen.py                        # DAG generation, problem sampling, dataset builder
    generate_datasets.py          # CLI: regenerate and upload datasets
  pyproject.toml
  README.md
```

---

## Data Fields

Each problem stored in the HuggingFace dataset has these fields in `info`:

| Field | Type | Description |
|-------|------|-------------|
| `problem_type` | str | `backdoor_empty` / `backdoor_standard` / `frontdoor` / `iv` |
| `identification_methods` | list[str] | Single-element list: `["backdoor"]` / `["frontdoor"]` / `["iv"]` |
| `edges` | list[list[int]] | DAG edges `[[u, v], ...]` for graph reconstruction |
| `nodes` | list[int] | All node IDs |
| `X`, `Y` | int | Treatment and outcome node IDs |
| `observed_nodes` | list[int] | Observable nodes |
| `latent_nodes` | list[int] | Latent (hidden) nodes |
| `domains` | dict | Node ID → list of actual domain values (e.g. `[-1, 0]` or `[0,1,2,3,4]`) |
| `cpts` | dict | Serialized CPTs; keys are pipe-delimited parent value strings |
| `topo_order` | list[int] | Topological order |
| `parents_map` | dict | Node ID → list of parent IDs |
| `true_ATE` | float or None | Exact ATE via enumeration; non-None for backdoor/frontdoor |
| `true_LATE` | float or None | Exact LATE via Wald estimator; non-None for IV |
| `minimal_set` | list[int] | Minimal adjustment set (backdoor), mediator set (frontdoor), or `[iv_instrument]` |
| `iv_instrument` | int or None | Node ID of the IV instrument; non-None for IV problems |

---

## Dependencies

Declared in `pyproject.toml`: `verifiers>=0.1.9.post3`, `networkx>=3.0`, `datasets`.
