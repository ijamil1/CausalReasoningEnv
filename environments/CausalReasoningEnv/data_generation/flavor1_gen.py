"""Flavor 1 data generation — Minimal Adjustment Set Identification.

Generates stratified DAG problems where the model must identify the minimal
adjustment set blocking all backdoor paths from treatment X to outcome Y.

Problem types (controlled stratification):
  - "standard" : |min_set| >= |parents(X)|  (all parents needed)
  - "ancestor" : ratio < 1; redundancy via ancestor absorption
  - "collider"  : ratio < 1; redundancy via collider structure
"""

import json
import random

import networkx as nx
from datasets import Dataset
from networkx.algorithms.d_separation import find_minimal_d_separator


# ─────────────────────────────────────────────────────────────────────────────
# DAG generation
# ─────────────────────────────────────────────────────────────────────────────


def _make_dag(n: int, edge_prob: float, rng: random.Random) -> nx.DiGraph:
    """Generate a random DAG by keeping only forward edges from an Erdos-Renyi graph."""
    nodes = list(range(n))
    edges = [
        (u, v)
        for u in nodes
        for v in nodes
        if u < v and rng.random() < edge_prob
    ]
    return nx.DiGraph(edges)


def _try_sample_problem(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
) -> dict | None:
    """Attempt to sample one valid causal adjustment-set problem.

    Each accepted problem satisfies:
      - Y is a descendant of X (a causal path exists).
      - Y is a leaf node (no outgoing edges).
      - At least 4 backdoor paths exist, with at least one of length >= 5 nodes.
      - A minimal d-separator (adjustment set) exists.

    Returns a problem dict (including a temporary "G" key for the nx.DiGraph
    used by _classify_problem) or None if any filter fails.
    """
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, edge_prob, rng)
    nodes_list = list(G.nodes())
    if len(nodes_list) < 2:
        return None

    X, Y = rng.sample(nodes_list, 2)

    if not nx.has_path(G, X, Y):
        return None
    if G.out_degree(Y) > 0:
        return None
    if nx.has_path(G, Y, X):
        return None

    G_bd = G.copy()
    G_bd.remove_edges_from(list(G.out_edges(X)))

    try:
        bd_paths = list(nx.all_simple_paths(G_bd.to_undirected(), X, Y))
        if len(bd_paths) < 4 or not any(len(p) >= 5 for p in bd_paths):
            return None
        min_set = find_minimal_d_separator(G_bd, X, Y)
        if min_set is None:
            return None
    except Exception:
        return None

    return {
        "G": G,  # temporary; removed before dataset serialisation
        "edges": [(int(u), int(v)) for u, v in G.edges()],
        "nodes": [int(nd) for nd in G.nodes()],
        "X": int(X),
        "Y": int(Y),
        "minimal_adjustment_set": sorted(int(nd) for nd in min_set),
        "num_nodes": len(nodes_list),
        "num_backdoor_paths": len(bd_paths),
    }


def _classify_problem(problem: dict) -> str:
    """Classify a problem by the mechanism behind its ratio (|min_set|/|parents(X)|).

    Uses the temporary "G" key placed by _try_sample_problem.

    Classifications:
      "standard" — |min_set| >= |parents(X)|  (all parents needed; ratio >= 1)
      "ancestor" — ratio < 1 AND there exists a dropped parent p such that some
                   z in min_set has a directed path z -> ... -> p in G. Conditioning
                   on z blocks p's backdoor contribution without conditioning on p.
      "collider" — ratio < 1 AND no dropped parent has any ancestor in min_set.
                   The redundancy arises from a collider structure on the backdoor
                   path(s) through that parent, not from ancestor absorption.
    """
    G: nx.DiGraph = problem["G"]
    X = problem["X"]
    parents_X = set(G.predecessors(X))
    min_set = set(problem["minimal_adjustment_set"])

    if len(min_set) >= len(parents_X):
        return "standard"

    dropped_parents = parents_X - min_set
    for p in dropped_parents:
        ancestors_of_p = nx.ancestors(G, p)
        if ancestors_of_p & min_set:
            return "ancestor"

    return "collider"


# ─────────────────────────────────────────────────────────────────────────────
# Stratified problem pool
# ─────────────────────────────────────────────────────────────────────────────


def generate_stratified_dag_problems(
    n_train: int = 250,
    n_eval: int = 100,
    min_nodes: int = 8,
    max_nodes: int = 12,
    edge_prob: float = 0.41,
    seed: int = 42,
    target_ratio_lt1: float = 0.40,
    target_ancestor_fraction: float = 0.50,
    exclude: set[tuple] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Generate stratified train and eval problem pools with controlled difficulty.

    Distribution targets (applied to both train and eval via stratified split):
      - ~target_ratio_lt1 of all problems have |min_set| < |parents(X)|.
      - Within those, ~target_ancestor_fraction are "ancestor" type; the rest
        are "collider" type.

    Args:
        n_train: Number of training problems.
        n_eval: Number of eval problems.
        min_nodes: Minimum number of nodes per DAG.
        max_nodes: Maximum number of nodes per DAG.
        edge_prob: Erdos-Renyi edge probability for DAG generation.
        seed: RNG seed for reproducibility.
        target_ratio_lt1: Fraction of problems where |min_set| < |parents(X)|.
        target_ancestor_fraction: Fraction of ratio-lt-1 problems that are "ancestor" type.
        exclude: Optional set of (frozenset(edges), X, Y) signatures to reject,
                 used to guarantee disjointness from an existing problem pool.

    Returns:
        (train_problems, eval_problems): two lists of problem dicts with keys:
        edges, nodes, X, Y, minimal_adjustment_set, num_nodes,
        num_backdoor_paths, num_parents_X, problem_type.
    """
    rng = random.Random(seed)
    n_total = n_train + n_eval

    n_lt1 = round(target_ratio_lt1 * n_total)
    n_ancestor = round(target_ancestor_fraction * n_lt1)
    n_collider = n_lt1 - n_ancestor
    n_standard = n_total - n_lt1

    targets = {"standard": n_standard, "ancestor": n_ancestor, "collider": n_collider}
    buckets: dict[str, list[dict]] = {"standard": [], "ancestor": [], "collider": []}
    seen: set[tuple] = set(exclude) if exclude else set()

    while any(len(buckets[t]) < targets[t] for t in targets):
        prob = _try_sample_problem(rng, min_nodes, max_nodes, edge_prob)
        if prob is None:
            continue

        sig = (frozenset((int(u), int(v)) for u, v in prob["edges"]), prob["X"], prob["Y"])
        if sig in seen:
            continue

        ptype = _classify_problem(prob)
        if len(buckets[ptype]) >= targets[ptype]:
            continue

        G: nx.DiGraph = prob.pop("G")
        parents_X = set(G.predecessors(prob["X"]))
        prob["num_parents_X"] = len(parents_X)
        prob["problem_type"] = ptype
        buckets[ptype].append(prob)
        seen.add(sig)

    train_frac = n_train / n_total if n_total > 0 else 1.0
    train_problems: list[dict] = []
    eval_problems: list[dict] = []

    for ptype in ("standard", "ancestor", "collider"):
        probs = buckets[ptype]
        rng.shuffle(probs)
        n_to_train = round(len(probs) * train_frac)
        train_problems.extend(probs[:n_to_train])
        eval_problems.extend(probs[n_to_train:])

    rng.shuffle(train_problems)
    rng.shuffle(eval_problems)

    return train_problems, eval_problems


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────


def build_dataset(problems: list[dict], format_fn) -> Dataset:
    """Convert a list of problem dicts into a HuggingFace Dataset.

    Args:
        problems: List of problem dicts from generate_stratified_dag_problems.
        format_fn: Callable(edges, nodes, X, Y) -> str — renders the problem text.
    """
    rows = []
    for p in problems:
        rows.append({
            "question": format_fn(p["edges"], p["nodes"], p["X"], p["Y"]),
            "info": json.dumps({
                "minimal_adjustment_set": p["minimal_adjustment_set"],
                "X": p["X"],
                "Y": p["Y"],
                "edges": p["edges"],
                "nodes": p["nodes"],
                "num_nodes": p["num_nodes"],
                "num_backdoor_paths": p["num_backdoor_paths"],
                "num_parents_X": p.get("num_parents_X"),
                "problem_type": p.get("problem_type"),
            }),
        })
    return Dataset.from_list(rows)
