# CausalReasoningEnv_1

A causal inference reasoning environment where the model must identify the **minimal adjustment set** for a given DAG.

## Task Description

Each problem presents:
- A randomly generated **Directed Acyclic Graph (DAG)** as a node/edge list with a full adjacency summary
- A **rendered image of the DAG** (blue node = treatment X, orange node = outcome Y) injected alongside the text
- A **treatment node X** and an **outcome node Y**

The model must identify the smallest set of nodes Z (the minimal adjustment set) that blocks all backdoor paths from X to Y, enabling unbiased estimation of the causal effect of X on Y.

### What is a backdoor path?
A backdoor path is any undirected path between X and Y that has an arrow *into* X — it represents a confounding channel. Conditioning on the adjustment set Z closes all such paths via d-separation.

### Constraints on Z
- Z must contain only non-descendants of X
- Z must not open new paths by conditioning on colliders
- Z must be minimal (no proper subset also qualifies)

## Dataset

DAGs are generated procedurally using stratified rejection sampling, filtered to ensure:
- Y is a descendant of X (a causal effect exists)
- Y is a leaf node (no outgoing edges)
- At least 4 backdoor paths exist, with at least one of length ≥ 5 nodes

**Difficulty stratification** — problems are classified by the relationship between the minimal adjustment set and the parents of X:
- **standard** (~60%): all parents of X appear in the minimal set (ratio = 1)
- **collider** (~20%): some parent is omitted because a collider on its backdoor path blocks it by default
- **ancestor** (~20%): some parent is omitted because an ancestor of that parent is already in the minimal set, blocking its confounding contribution

Both train and eval splits preserve this distribution. All problems are unique by (edges, X, Y) signature, and train/eval are disjoint.

Generation parameters are fixed — **250 train / 100 eval**, nodes 8–12, edge probability 0.41, seed=42.

### Pre-building datasets

Run the module directly to generate and save both splits as HuggingFace Datasets (avoids regeneration on every `load_environment()` call):

```bash
uv run python environments/CausalReasoningEnv_1/CausalReasoningEnv_1.py
```

Datasets are saved to `environments/CausalReasoningEnv_1/datasets/{train,eval}`. Calling `load_environment()` with no arguments will auto-load from disk if the datasets exist, or regenerate them from scratch otherwise:

```python
# Auto-load from disk (recommended after running __main__)
env = load_environment()

# Or pass datasets explicitly
from datasets import load_from_disk
from CausalReasoningEnv_1 import load_environment, _TRAIN_DATASET_PATH, _EVAL_DATASET_PATH

env = load_environment(
    train_dataset=load_from_disk(str(_TRAIN_DATASET_PATH)),
    eval_dataset=load_from_disk(str(_EVAL_DATASET_PATH)),
)
```

## Environment Architecture

**Type**: `vf.SingleTurnEnv` subclass (`CausalReasoningEnv`)

The model receives one prompt and produces one response — no multi-turn loop, no tools.

### Image injection via `setup_state`

At the start of each rollout, `setup_state` reconstructs the `nx.DiGraph` from `state["info"]`, renders it as a PNG using a topological layer layout (sources at top, sinks at bottom), base64-encodes it, and replaces the user message's plain-string content with a `[text, image_url]` multimodal content list. This keeps the HuggingFace dataset lean (only edge lists are stored) while delivering a fresh rendered image to the model at rollout time.

The system message is also upgraded to multimodal at rollout time to inject the pre-rendered ICL Example 2 DAG image alongside its worked solution.

### `load_environment`

Accepts optional pre-built `train_dataset` and `eval_dataset` arguments. If omitted, auto-loads from disk if datasets exist; otherwise regenerates from scratch using the fixed parameters below.

| Parameter | Value |
|---|---|
| `num_train` | `250` |
| `num_eval` | `100` |
| `min_nodes` | `8` |
| `max_nodes` | `12` |
| `edge_prob` | `0.41` |
| `seed` | `42` |
| `target_ratio_lt1` | `0.40` |
| `target_ancestor_fraction` | `0.50` |

## Response Format

```
<reasoning>
Step-by-step analysis of the graph, backdoor paths, and justification
for why the chosen set is minimal and valid.
</reasoning>
<answer>{node_id1, node_id2, ...}</answer>
```

Empty adjustment set: `<answer>{}</answer>`

## Reward Functions

| Function | Weight | Description |
|---|---|---|
| `adjustment_set_accuracy` | 0.9 | Exact match → 1.0; partial overlap → Jaccard similarity (intersection / union); unparseable → 0.0 |
| `valid_adjustment_set` | 0.0 | (Metric only) 1.0 if the predicted set is a valid adjustment set (not necessarily minimal): no descendants of X included, and Z d-separates X from Y in the backdoor graph |
| `format_compliance` | 0.1 | 1.0 if `<answer>{...}</answer>` is correctly present and parseable, else 0.0 |

## Installation & Evaluation

```bash
prime env install CausalReasoningEnv_1
prime eval run CausalReasoningEnv_1 -n 20
```
