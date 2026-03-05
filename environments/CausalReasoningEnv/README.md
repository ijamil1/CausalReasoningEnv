# CausalReasoningEnv

Multi-turn causal reasoning benchmark and RL training environment, built with Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework.

The environment trains models to identify causal adjustment strategies and compute Average Treatment Effects (ATE) via probability query tools over discrete CPT-based DAGs.

---

## Design

### Two-Phase Rollout

**Phase 1 — Declaration (Turn 1):** The model reasons about the DAG structure and writes exactly one `<set>` tag declaring its identification set before making any tool calls. Scored independently of computation.

**Phase 2 — Tool use + Answer:** The model calls tools and writes `<answer>` to end the episode. A global cap of 5 tool calls is enforced; the optimal maximum is 2.

### Problem Types

| Type | Fraction | Identification strategy | Optimal tool calls |
|------|----------|------------------------|-------------------|
| `backdoor_empty` | ~20% | Empty adjustment set (no confounding) | 1 |
| `backdoor_standard` | ~35% | Non-empty minimal adjustment set Z | 2 |
| `frontdoor` | ~20% | Latent confounder; valid mediator set M (possibly multi-node) | 2 |
| `not_identifiable` | ~25% | No valid backdoor or frontdoor strategy | 0 |

### Identification Set Design

Each problem has **exactly one valid identification method** by construction:
- Backdoor problems: verified to have no valid frontdoor set.
- Frontdoor problems: verified to have no valid backdoor adjustment set (latent confounder L→X, L→Y).

The stored `minimal_set` field holds the minimal adjustment set (for backdoor) or minimal mediator set (for frontdoor), found via `find_minimal_d_separator` and `minimum_node_cut` respectively.

### Reward Rubric

| Component | Weight | Description |
|-----------|--------|-------------|
| `format_compliance` | 0.05 | Valid `<answer>` block present |
| `set_valid` | 0.30 | Declared `<set>` satisfies the identification criterion |
| `minimality` | 0.15 | Graded: 1.0 if minimal, k/|declared| if valid superset |
| `ate_accuracy` | 0.50 | Final answer within ±0.01 of true ATE (or correct not_identifiable) |

### Tools

```
marginal(variables)
  Returns the full joint PMF P(V1, V2, ...) for all value combinations.
  Input:  variables — list of node IDs as strings, e.g. ["2", "3"]
  Output: P(node2=0, node3=0) = 0.1234 ...

conditional(query, given)
  Returns P(query | given) for all strata of the conditioning variables.
  Input:  query, given — lists of node IDs as strings
  Output: P(node4=0 | node0=0, node2=0) = 0.7234 ...
```

### Declaration Format

```
<set>2, 3</set>    ← identification set {node2, node3}
<set>{}</set>       ← empty identification set (no confounding)
<set></set>         ← ATE is not identifiable
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
| `problem_type` | str | `backdoor_empty` / `backdoor_standard` / `frontdoor` / `not_identifiable` |
| `edges` | list[list[int]] | DAG edges `[[u, v], ...]` for graph reconstruction |
| `nodes` | list[int] | All node IDs |
| `X`, `Y` | int | Treatment and outcome node IDs |
| `observed_nodes` | list[int] | Observable nodes |
| `latent_nodes` | list[int] | Latent (hidden) nodes |
| `domains` | dict | Node ID → list of values |
| `cpts` | dict | Serialized conditional probability tables |
| `topo_order` | list[int] | Topological order |
| `parents_map` | dict | Node ID → list of parent IDs |
| `identifiability_status` | str | `"identifiable"` or `"not_identifiable"` |
| `true_ATE` | float or None | Exact ATE via do-calculus enumeration |
| `minimal_set` | list[int] or None | Minimal adjustment or mediator set |
| `optimal_turns` | int | Minimum tool calls (0/1/2/2) |

---

## Dependencies

Declared in `pyproject.toml`: `verifiers>=0.1.9.post3`, `networkx>=3.0`, `datasets`.
