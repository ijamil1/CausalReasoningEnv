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

DAGs are generated procedurally using rejection sampling, filtered to ensure:
- Y is a descendant of X (a causal effect exists)
- Y is a leaf node (no outgoing edges)
- At least 2 backdoor paths exist, with at least one of length > 3

Splits and DAG generation parameters are fixed constants — **250 train / 100 eval**, nodes 5–8, seed=42 — so the dataset is identical across every run. All problems are unique by (edges, X, Y) signature.

## Environment Architecture

**Type**: `vf.SingleTurnEnv` subclass (`CausalReasoningEnv`)

The model receives one prompt and produces one response — no multi-turn loop, no tools.

### Image injection via `setup_state`

At the start of each rollout, `setup_state` reconstructs the `nx.DiGraph` from `state["info"]`, renders it as a PNG using a topological layer layout (sources at top, sinks at bottom), base64-encodes it, and replaces the user message's plain-string content with a `[text, image_url]` multimodal content list. This keeps the HuggingFace dataset lean (only edge lists are stored) while delivering a fresh rendered image to the model at rollout time.

The rendered image is ~30–67 KB (PNG) depending on DAG size, consuming approximately **~590 visual tokens** at the default 800×600 resolution for Qwen3-VL models.

### `load_environment`

Takes no arguments. All parameters are fixed internally:

| Parameter | Value |
|---|---|
| `num_train` | `250` |
| `num_eval` | `100` |
| `min_nodes` | `5` |
| `max_nodes` | `8` |
| `seed` | `42` |

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
