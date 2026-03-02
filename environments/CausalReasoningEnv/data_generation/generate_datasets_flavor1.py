"""Standalone script: generate and save Flavor 1 train/eval datasets.

Usage:
    uv run generate_datasets.py

The script does NOT import `verifiers`, so it avoids the pyqwest / Rust build
dependency. Run it once to pre-build the datasets on disk; load_flavor1() will
then read them from disk at training / eval time without re-generating.
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "networkx>=3.0",
#   "datasets",
#   "scipy",
#   "pandas",
#   "statsmodels",
# ]
# ///

import pathlib
import sys
from collections import Counter

# Make the package root importable when running as a script
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from data_generation.flavor1_gen import build_dataset, generate_stratified_dag_problems

# ── constants (must match the values in flavor1.py) ──────────────────────────

_NUM_TRAIN = 250
_NUM_EVAL = 100
_MIN_NODES = 8
_MAX_NODES = 12
_SEED = 42

_DATASET_DIR = pathlib.Path(__file__).parent.parent / "datasets" / "flavor1"
_TRAIN_DATASET_PATH = _DATASET_DIR / "train"
_EVAL_DATASET_PATH = _DATASET_DIR / "eval"


# ── format function (must match format_problem in flavor1.py) ─────────────────


def format_problem(
    edges: list,
    nodes: list,
    observed_nodes: list,
    latent_nodes: list,
    X: int,
    Y: int,
) -> str:
    parents: dict[int, list[int]] = {n: [] for n in nodes}
    children: dict[int, list[int]] = {n: [] for n in nodes}
    for u, v in edges:
        children[u].append(v)
        parents[v].append(u)

    edge_str = ", ".join(f"{u}->{v}" for u, v in sorted(edges))
    node_str = ", ".join(str(n) for n in sorted(nodes))
    obs_str = ", ".join(str(n) for n in sorted(observed_nodes))
    lat_str = ", ".join(str(n) for n in sorted(latent_nodes)) if latent_nodes else "none"

    adj_lines = []
    for n in sorted(nodes):
        pa = sorted(parents[n])
        ch = sorted(children[n])
        kind = "latent" if n in latent_nodes else "observed"
        adj_lines.append(
            f"  Node {n} ({kind}): parents=[{', '.join(map(str, pa))}], "
            f"children=[{', '.join(map(str, ch))}]"
        )
    adj_str = "\n".join(adj_lines)

    return (
        f"DAG INFORMATION\n"
        f"───────────────\n"
        f"Nodes:    {node_str}\n"
        f"Observed: {obs_str}\n"
        f"Latent:   {lat_str}\n"
        f"Edges:    {edge_str}\n\n"
        f"Adjacency:\n{adj_str}\n\n"
        f"Treatment (X): {X}\n"
        f"Outcome   (Y): {Y}\n\n"
        f"QUESTION\n"
        f"────────\n"
        f"Is ATE = E[Y | do(X=1)] − E[Y | do(X=0)] identifiable from the causal model implied by this DAG? "
        f"If yes, state the smallest required variable set (excluding {X} and {Y})."
        f"If not, respond with not_identifiable. "
        f"Respond according to the response format specified in the system prompt."
    )


# ── main ─────────────────────────────────────────────────────────────────────

print(f"Generating stratified problems (seed={_SEED}, nodes={_MIN_NODES}–{_MAX_NODES})…")
train_problems, eval_problems = generate_stratified_dag_problems(
    n_train=_NUM_TRAIN,
    n_eval=_NUM_EVAL,
    min_nodes=_MIN_NODES,
    max_nodes=_MAX_NODES,
    seed=_SEED,
)
print(f"  Train: {len(train_problems)} problems")
print(f"  Eval:  {len(eval_problems)} problems")

for split_name, probs in [("Train", train_problems), ("Eval", eval_problems)]:
    counts = Counter(p["problem_type"] for p in probs)
    print(f"  {split_name} type distribution: {dict(counts)}")

train_dataset = build_dataset(train_problems, format_problem)
eval_dataset = build_dataset(eval_problems, format_problem)

_DATASET_DIR.mkdir(parents=True, exist_ok=True)
train_dataset.save_to_disk(str(_TRAIN_DATASET_PATH))
eval_dataset.save_to_disk(str(_EVAL_DATASET_PATH))
print(f"\nTrain dataset ({len(train_dataset)} rows) → {_TRAIN_DATASET_PATH}")
print(f"Eval  dataset ({len(eval_dataset)} rows) → {_EVAL_DATASET_PATH}")
