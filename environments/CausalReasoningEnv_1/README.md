# CausalReasoningEnv_1

A causal inference reasoning environment where the model must identify the **minimal adjustment set** for a given DAG.

## Task Description

Each problem presents:
- A randomly generated **Directed Acyclic Graph (DAG)** as a node/edge list with an adjacency summary
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

Default splits: **200 train / 50 eval**, using disjoint samples from the same rejection-sampling pool (seed=42).

## Environment Architecture

**Type**: `vf.ToolEnv` (currently `tools=[]` — behaves as single-turn)

The ToolEnv base is chosen so that graph-exploration tools can be added later without restructuring the reward or prompt logic. Candidate tools (TBD):
- `get_parents(node)` / `get_children(node)` / `get_descendants(node)`
- `find_paths(source, target)`
- `check_d_separation(X, Y, Z)`

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
| `adjustment_set_accuracy` | 1.0 | Exact match → 1.0; partial match → Jaccard similarity; unparseable → 0.0 |
| `format_compliance` | 0.1 | 1.0 if `<answer>{...}</answer>` is correctly present, else 0.0 |

## Installation & Evaluation

```bash
prime env install CausalReasoningEnv_1
prime eval run CausalReasoningEnv_1 -n 20
```
