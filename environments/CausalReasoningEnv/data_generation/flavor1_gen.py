"""Flavor 1 data generation — Minimal Adjustment Set Identification.

Generates stratified DAG problems where the model must:
  (a) determine whether (and how) ATE is identifiable from the observed
      variables, and
  (b) if identifiable via backdoor: produce the minimal valid adjustment set Z.
      if identifiable via front-door: identify the mediator M.

Every problem exposes which nodes are observed vs. latent.  The model is
expected to output one of four identifiability statuses:
  - "identifiable"          — non-empty minimal observed adjustment set exists
  - "empty"                 — adjustment set is empty (X, Y already d-separated
                              in the backdoor graph; no conditioning needed)
  - "identifiable_frontdoor"— no valid backdoor adjustment set exists, but the
                              front-door criterion applies via mediator M
  - "not_identifiable"      — no valid observed adjustment set exists, and
                              front-door is inapplicable (latent confounder
                              with direct X→Y edge prevents full interception)

All problems guarantee a directed causal path from X to Y.  For
identifiable_front_door, this path is indirect (X→M→Y); there is no
direct X→Y edge.

─────────────────────────────────────────────────────────────────────────────
PROBLEM TYPES  (six stratified buckets)
─────────────────────────────────────────────────────────────────────────────

  "identifiable_standard"  (~20% of total)
    Non-empty minimal adjustment set; the min_set is exactly the set of
    observed parents of X  (|min_set| = |observed_parents(X)|).

  "identifiable_ancestor"  (~15%)
    Non-empty set; |min_set| < |observed_parents(X)|; redundancy because
    every dropped observed parent's paths to Y pass through an ancestor
    already in min_set.

  "identifiable_collider"  (~20%)
    Non-empty set; |min_set| < |observed_parents(X)|; redundancy via a
    collider structure on the backdoor path through a dropped parent.

  "identifiable_front_door"  (~10%)
    Backdoor adjustment is blocked by a latent confounder L (L→X, L→Y),
    but ATE is identifiable via the front-door criterion through mediator M.
    Construction:
      - X→M→Y is the only directed X→Y path (no direct X→Y edge).
      - Latent node L appended with edges L→X and L→Y (NOT L→M).
    Front-door conditions satisfied by construction:
      (1) No unblocked backdoor path from X to M — L connects only to X
          and Y, so no L→...→M path exists.
      (2) M intercepts all directed X→Y paths — verified by checking
          nx.has_path(G_without_M, X, Y) is False.
      (3) All backdoor paths from M to Y are blocked by X — the only
          backdoor path M←X←L→Y is blocked by conditioning on X (X is a
          non-collider on that path).

  "empty"  (~15%)
    Empty minimal adjustment set.  X and Y are already d-separated in the
    backdoor graph (all paths blocked by unconditioned colliders), so no
    conditioning is needed.  Problems require ≥1 undirected path in G_bd
    (result is structural, not trivial absence of common causes).

  "not_identifiable"  (~20%)
    No valid observed adjustment set exists, and front-door is inapplicable.
    Construction: X→Y is a direct edge; latent node L appended with edges
    L→X and L→Y.  The direct X→Y edge rules out front-door (condition 2
    requires M to intercept ALL X→Y paths, which the direct edge violates).
    L is a latent fork on X←L→Y that cannot be blocked by any observed set.

─────────────────────────────────────────────────────────────────────────────
SAMPLING DESIGN
─────────────────────────────────────────────────────────────────────────────

Backdoor-identifiable and empty problems  (_try_sample_observed):
  1. Generate a random forward DAG (Erdős–Rényi, 8–12 nodes, p ≈ 0.41).
  2. Sample X, Y such that a directed X→Y path exists, Y is a leaf, and
     Y→X is impossible.
  3. Randomly mark non-X, non-Y nodes as latent (Bernoulli, p=0.15 each).
     X and Y are always observed.
  4. Build the backdoor graph G_bd (remove all outgoing edges from X).
  5. Call find_minimal_d_separator(G_bd, X, Y,
                                   restricted=observed_nodes − {X, Y}).
       None  → no observed adjustment set → discard, try again.
       {}    → empty problem type.
       Z≠{} → identifiable type (standard / ancestor / collider classification
               determined by _classify_identifiable_problem).
  6. Complexity filters:
       identifiable: ≥3 undirected backdoor paths in G_bd, ≥1 of length ≥5.
       empty: ≥1 undirected path in G_bd (non-trivial collider structure).

Front-door problems  (_try_sample_front_door):
  1. Generate a random forward DAG (8–12 nodes).
  2. Find a triple (X, M, Y) where X→M and M→Y are direct edges and Y is
     a leaf.  No direct X→Y edge may exist.
  3. Verify condition (2): remove M from G, check nx.has_path(G_no_M, X, Y)
     is False — ensuring M is the sole interceptor of all X→Y paths.
  4. Add latent node L (index n) with edges L→X and L→Y (not L→M).
     Conditions (1) and (3) are satisfied by this construction (see above).
  5. All original nodes are observed; L is latent.

Not-identifiable problems  (_try_sample_not_identifiable):
  1. Generate a random forward DAG (8–12 nodes).
  2. Find (X, Y) where X→Y is a direct edge and Y is a leaf.
     The direct X→Y edge makes front-door inapplicable regardless of
     any mediator paths that may also exist in the DAG.
  3. Add latent node L (index n) with edges L→X and L→Y.
  4. Verify find_minimal_d_separator(G_bd, X, Y,
                                     restricted=observed − {X, Y}) is None.

─────────────────────────────────────────────────────────────────────────────
DATA FIELDS (per problem dict)
─────────────────────────────────────────────────────────────────────────────

  edges                   list of (u, v) directed edge pairs
  nodes                   all node ids (observed + latent)
  X, Y                    treatment and outcome node ids
  minimal_adjustment_sets all minimum-size adjustment sets (list of sorted
                          lists; every entry has the same cardinality k, the
                          smallest possible), or None for not_identifiable
                          and identifiable_frontdoor
  mediator_node           mediator node id for identifiable_frontdoor, else None
  identifiability_status  "identifiable" | "empty" |
                          "identifiable_frontdoor" | "not_identifiable"
  observed_nodes          sorted list of observed node ids
  latent_nodes            sorted list of latent node ids
  num_nodes               total node count
  num_backdoor_paths      count of undirected paths in G_bd, or None
  num_parents_X           observed parents of X count, or None
  problem_type            one of the six bucket names above

─────────────────────────────────────────────────────────────────────────────
build_dataset  format_fn  signature change
─────────────────────────────────────────────────────────────────────────────

  Old: format_fn(edges, nodes, X, Y) -> str
  New: format_fn(edges, nodes, observed_nodes, latent_nodes, X, Y) -> str
"""

import json
import random
from itertools import combinations

import networkx as nx
from datasets import Dataset
from networkx.algorithms.d_separation import find_minimal_d_separator, is_d_separator


# ─────────────────────────────────────────────────────────────────────────────
# DAG generation
# ─────────────────────────────────────────────────────────────────────────────


def _make_dag(n: int, edge_prob: float, rng: random.Random) -> nx.DiGraph:
    """Generate a random DAG by keeping only forward edges from an Erdős–Rényi graph."""
    nodes = list(range(n))
    edges = [
        (u, v)
        for u in nodes
        for v in nodes
        if u < v and rng.random() < edge_prob
    ]
    return nx.DiGraph(edges)


def _make_backdoor_graph(G: nx.DiGraph, X: int) -> nx.DiGraph:
    """Return a copy of G with all outgoing edges from X removed."""
    G_bd = G.copy()
    G_bd.remove_edges_from(list(G.out_edges(X)))
    return G_bd


def _add_latent_confounder(G: nx.DiGraph, X: int, Y: int, n: int) -> int:
    """Append latent node L=n to G with edges L→X and L→Y. Returns L."""
    G.add_node(n)
    G.add_edge(n, X)
    G.add_edge(n, Y)
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Adjustment set enumeration
# ─────────────────────────────────────────────────────────────────────────────


def _find_all_minimum_adjustment_sets(
    G_bd: nx.DiGraph,
    X: int,
    Y: int,
    observed_nodes: set[int],
    first_hint: set[int] | None = None,
) -> list[list[int]] | None:
    """Find all minimum-size adjustment sets for (X, Y) in G_bd restricted to observed_nodes − {X, Y}.

    "Minimum-size" means every returned set has the smallest cardinality k of
    any valid observed adjustment set.  All size-k valid sets are automatically
    minimal (no proper subset of size k-1 can separate, since k is minimum), so
    the explicit minimality check from the naive approach is unnecessary.

    Args:
        first_hint: A known valid separator (e.g. from a prior call to
                    find_minimal_d_separator) to avoid recomputing it.

    Returns a list of sorted lists, or None if no observed separator exists.
    Complexity: O(C(|candidates|, k)) d-separation checks where k is the
    minimum separator size — much faster than enumerating all 2^n subsets.
    """
    candidates = sorted(observed_nodes - {X, Y})

    # Establish k: the minimum separator size — reuse caller's result if provided
    first = first_hint if first_hint is not None else find_minimal_d_separator(
        G_bd, X, Y, restricted=set(candidates)
    )
    if first is None:
        return None

    k = len(first)

    # Enumerate all size-k subsets; valid ones are guaranteed minimum-size minimal
    all_minimum = [
        sorted(combo)
        for combo in combinations(candidates, k)
        if is_d_separator(G_bd, {X}, {Y}, set(combo))
    ]

    return all_minimum if all_minimum else [sorted(first)]


# ─────────────────────────────────────────────────────────────────────────────
# Problem samplers
# ─────────────────────────────────────────────────────────────────────────────


def _try_sample_observed(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
    latent_node_prob: float,
) -> dict | None:
    """Attempt to sample a backdoor-identifiable or empty problem.

    Latent nodes are assigned BEFORE the minimal d-separator is computed so
    that the observed/latent split drives which problems are solvable.

    Returns a dict with a temporary "G" key (the nx.DiGraph) that must be
    popped before serialisation; it is consumed by _classify_identifiable_problem.
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

    # Assign latent nodes; X and Y are always observed
    other_nodes = [nd for nd in nodes_list if nd != X and nd != Y]
    latent_nodes = {nd for nd in other_nodes if rng.random() < latent_node_prob}
    observed_nodes = set(nodes_list) - latent_nodes

    # Backdoor graph: remove all outgoing edges from X
    G_bd = _make_backdoor_graph(G, X)

    # Minimal d-separator restricted to observed nodes \ {X, Y}
    try:
        min_set = find_minimal_d_separator(G_bd, X, Y, restricted=observed_nodes - {X, Y})
    except Exception:
        return None

    if min_set is None:
        return None  # No observed adjustment set — discard

    # Enumerate all minimum-size sets; reuse min_set to skip a redundant call
    try:
        all_min_sets = _find_all_minimum_adjustment_sets(G_bd, X, Y, observed_nodes, first_hint=min_set)
    except Exception:
        return None

    # Undirected backdoor paths for complexity filters
    try:
        bd_paths = list(nx.all_simple_paths(G_bd.to_undirected(), X, Y))
    except Exception:
        return None

    if len(min_set) == 0:
        # Empty type: require ≥1 undirected path in G_bd (non-trivial collider structure)
        if not bd_paths:
            return None
        return {
            "G": G,
            "edges": [(int(u), int(v)) for u, v in G.edges()],
            "nodes": [int(nd) for nd in nodes_list],
            "X": int(X),
            "Y": int(Y),
            "minimal_adjustment_sets": all_min_sets,  # [[]] — empty set is the only minimal set
            "mediator_node": None,
            "identifiability_status": "empty",
            "observed_nodes": sorted(int(nd) for nd in observed_nodes),
            "latent_nodes": sorted(int(nd) for nd in latent_nodes),
            "num_nodes": len(nodes_list),
            "num_backdoor_paths": len(bd_paths),
            "num_parents_X": None,
            "problem_type": "empty",
        }

    # Identifiable type: require ≥3 backdoor paths, ≥1 of length ≥5
    if len(bd_paths) < 3 or not any(len(p) >= 5 for p in bd_paths):
        return None

    observed_parents_X = set(G.predecessors(X)) & observed_nodes
    return {
        "G": G,
        "edges": [(int(u), int(v)) for u, v in G.edges()],
        "nodes": [int(nd) for nd in nodes_list],
        "X": int(X),
        "Y": int(Y),
        "minimal_adjustment_sets": all_min_sets,
        "mediator_node": None,
        "identifiability_status": "identifiable",
        "observed_nodes": sorted(int(nd) for nd in observed_nodes),
        "latent_nodes": sorted(int(nd) for nd in latent_nodes),
        "num_nodes": len(nodes_list),
        "num_backdoor_paths": len(bd_paths),
        "num_parents_X": len(observed_parents_X),
        "problem_type": None,  # set by _classify_identifiable_problem
    }


def _try_sample_front_door(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
) -> dict | None:
    """Attempt to sample a front-door identifiable problem.

    Constructs X→M→Y with latent L→X, L→Y.  Verifies M is the sole interceptor
    of all directed X→Y paths (front-door condition 2).  Conditions 1 and 3 are
    guaranteed by construction: L connects only to X and Y, not M.
    """
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, edge_prob, rng)

    # Candidate triples: X→M direct edge, M→Y direct edge, Y is a leaf, X ≠ Y
    candidates = [
        (u, m, v)
        for u, m in G.edges()
        for v in G.successors(m)
        if G.out_degree(v) == 0 and u != v
    ]
    if not candidates:
        return None

    X, M, Y = rng.choice(candidates)

    # No direct X→Y edge allowed (front-door requires indirect path only)
    if G.has_edge(X, Y):
        return None

    # Condition (2): M must intercept ALL directed X→Y paths
    G_no_M = G.copy()
    G_no_M.remove_node(M)
    if nx.has_path(G_no_M, X, Y):
        return None  # Some X→Y path bypasses M

    # Add latent confounder L; do NOT add L→M (preserves condition 1)
    L = _add_latent_confounder(G, X, Y, n)

    nodes_list = list(G.nodes())
    observed_nodes = set(range(n))  # original nodes only
    latent_nodes = {L}

    return {
        "edges": [(int(u), int(v)) for u, v in G.edges()],
        "nodes": [int(nd) for nd in nodes_list],
        "X": int(X),
        "Y": int(Y),
        "minimal_adjustment_sets": None,
        "mediator_node": int(M),
        "identifiability_status": "identifiable_frontdoor",
        "observed_nodes": sorted(int(nd) for nd in observed_nodes),
        "latent_nodes": sorted(int(nd) for nd in latent_nodes),
        "num_nodes": len(nodes_list),
        "num_backdoor_paths": None,
        "num_parents_X": None,
        "problem_type": "identifiable_front_door",
    }


def _try_sample_not_identifiable(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
) -> dict | None:
    """Attempt to sample a not-identifiable problem.

    Constructs X→Y (direct) with latent L→X, L→Y.  The direct X→Y edge rules
    out front-door (no mediator can intercept all X→Y paths).  Verifies the
    d-separator restricted to observed nodes returns None.
    """
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, edge_prob, rng)

    # Candidate (X, Y): X→Y direct edge, Y is a leaf
    candidates = [(u, v) for u, v in G.edges() if G.out_degree(v) == 0]
    if not candidates:
        return None

    X, Y = rng.choice(candidates)

    # Add latent confounder L
    L = _add_latent_confounder(G, X, Y, n)

    nodes_list = list(G.nodes())
    observed_nodes = set(range(n))
    latent_nodes = {L}

    G_bd = _make_backdoor_graph(G, X)

    try:
        check = find_minimal_d_separator(G_bd, X, Y, restricted=observed_nodes - {X, Y})
    except Exception:
        return None

    if check is not None:
        return None  # Construction should always give None; guard against edge cases

    return {
        "edges": [(int(u), int(v)) for u, v in G.edges()],
        "nodes": [int(nd) for nd in nodes_list],
        "X": int(X),
        "Y": int(Y),
        "minimal_adjustment_sets": None,
        "mediator_node": None,
        "identifiability_status": "not_identifiable",
        "observed_nodes": sorted(int(nd) for nd in observed_nodes),
        "latent_nodes": sorted(int(nd) for nd in latent_nodes),
        "num_nodes": len(nodes_list),
        "num_backdoor_paths": None,
        "num_parents_X": None,
        "problem_type": "not_identifiable",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Problem classification
# ─────────────────────────────────────────────────────────────────────────────


def _classify_identifiable_problem(problem: dict, G: nx.DiGraph) -> str:
    """Classify a non-empty identifiable problem as standard / ancestor / collider.

    Uses the first entry in minimal_adjustment_sets for classification.

    Compares |min_set| against the number of *observed* parents of X:
      "identifiable_standard" — |min_set| >= |observed_parents(X)|  (no redundancy)
      "identifiable_ancestor" — |min_set| < |observed_parents(X)|; a dropped
                                observed parent has an ancestor in min_set
      "identifiable_collider" — |min_set| < |observed_parents(X)|; redundancy
                                via collider structure on the backdoor path
    """
    X = problem["X"]
    observed_nodes = set(problem["observed_nodes"])
    parents_X = set(G.predecessors(X))
    observed_parents_X = parents_X & observed_nodes
    min_set = set(problem["minimal_adjustment_sets"][0])

    if len(min_set) >= len(observed_parents_X):
        return "identifiable_standard"

    dropped_parents = observed_parents_X - min_set
    for p in dropped_parents:
        if nx.ancestors(G, p) & min_set:
            return "identifiable_ancestor"

    return "identifiable_collider"


# ─────────────────────────────────────────────────────────────────────────────
# Stratified problem pool
# ─────────────────────────────────────────────────────────────────────────────


def generate_stratified_dag_problems(
    n_train: int = 250,
    n_eval: int = 100,
    min_nodes: int = 8,
    max_nodes: int = 12,
    edge_prob: float = 0.41,
    latent_node_prob: float = 0.15,
    seed: int = 42,
    target_standard_frac: float = 0.20,
    target_ancestor_frac: float = 0.15,
    target_collider_frac: float = 0.20,
    target_front_door_frac: float = 0.10,
    target_empty_frac: float = 0.15,
    # not_identifiable receives the remainder (~0.20)
    exclude: set[tuple] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Generate stratified train and eval problem pools across six problem types.

    Default distribution targets (sum to 1.0):
      identifiable_standard:   20%
      identifiable_ancestor:   15%
      identifiable_collider:   20%
      identifiable_front_door: 10%
      empty:                   15%
      not_identifiable:        20%  (remainder)

    Args:
        n_train: Number of training problems.
        n_eval: Number of eval problems.
        min_nodes: Minimum number of DAG nodes.
        max_nodes: Maximum number of DAG nodes.
        edge_prob: Erdős–Rényi forward-edge probability.
        latent_node_prob: Per-node probability of being marked latent for
                          backdoor-identifiable and empty types.
        seed: RNG seed for reproducibility.
        target_*_frac: Fraction of total problems for each bucket.
        exclude: Optional set of (frozenset(edges), X, Y) signatures to reject,
                 used to guarantee disjointness from an existing problem pool.

    Returns:
        (train_problems, eval_problems): two lists of problem dicts with all
        DATA FIELDS described in the module docstring.
    """
    rng = random.Random(seed)
    n_total = n_train + n_eval

    n_standard = round(target_standard_frac * n_total)
    n_ancestor = round(target_ancestor_frac * n_total)
    n_collider = round(target_collider_frac * n_total)
    n_frontdoor = round(target_front_door_frac * n_total)
    n_empty = round(target_empty_frac * n_total)
    n_not_id = n_total - n_standard - n_ancestor - n_collider - n_frontdoor - n_empty

    targets: dict[str, int] = {
        "identifiable_standard": n_standard,
        "identifiable_ancestor": n_ancestor,
        "identifiable_collider": n_collider,
        "identifiable_front_door": n_frontdoor,
        "empty": n_empty,
        "not_identifiable": n_not_id,
    }
    buckets: dict[str, list[dict]] = {k: [] for k in targets}
    seen: set[tuple] = set(exclude) if exclude else set()

    def _sig(prob: dict) -> tuple:
        return (frozenset((int(u), int(v)) for u, v in prob["edges"]), prob["X"], prob["Y"])

    _observed_types = ("identifiable_standard", "identifiable_ancestor", "identifiable_collider", "empty")

    while any(len(buckets[t]) < targets[t] for t in targets):
        need_not_id = len(buckets["not_identifiable"]) < targets["not_identifiable"]
        need_frontdoor = len(buckets["identifiable_front_door"]) < targets["identifiable_front_door"]
        need_observed = any(len(buckets[t]) < targets[t] for t in _observed_types)

        # Weighted random choice of sampler proportional to remaining need
        options: list[tuple[str, int]] = []
        if need_not_id:
            options.append(("not_id", targets["not_identifiable"] - len(buckets["not_identifiable"])))
        if need_frontdoor:
            options.append(("frontdoor", targets["identifiable_front_door"] - len(buckets["identifiable_front_door"])))
        if need_observed:
            obs_remaining = sum(
                targets[t] - len(buckets[t])
                for t in _observed_types
                if len(buckets[t]) < targets[t]
            )
            options.append(("observed", obs_remaining))

        if not options:
            break

        sampler = rng.choices(
            [name for name, _ in options],
            weights=[w for _, w in options],
        )[0]

        # ── not_identifiable ──────────────────────────────────────────────────
        if sampler == "not_id":
            prob = _try_sample_not_identifiable(rng, min_nodes, max_nodes, edge_prob)
            if prob is None:
                continue
            sig = _sig(prob)
            if sig in seen:
                continue
            if len(buckets["not_identifiable"]) < targets["not_identifiable"]:
                buckets["not_identifiable"].append(prob)
                seen.add(sig)

        # ── front_door ────────────────────────────────────────────────────────
        elif sampler == "frontdoor":
            prob = _try_sample_front_door(rng, min_nodes, max_nodes, edge_prob)
            if prob is None:
                continue
            sig = _sig(prob)
            if sig in seen:
                continue
            if len(buckets["identifiable_front_door"]) < targets["identifiable_front_door"]:
                buckets["identifiable_front_door"].append(prob)
                seen.add(sig)

        # ── backdoor-identifiable / empty ─────────────────────────────────────
        elif sampler == "observed":
            prob = _try_sample_observed(rng, min_nodes, max_nodes, edge_prob, latent_node_prob)
            if prob is None:
                continue
            sig = _sig(prob)
            if sig in seen:
                continue

            if prob["identifiability_status"] == "empty":
                if len(buckets["empty"]) < targets["empty"]:
                    prob.pop("G", None)
                    buckets["empty"].append(prob)
                    seen.add(sig)
            else:
                G: nx.DiGraph = prob.pop("G")
                ptype = _classify_identifiable_problem(prob, G)
                if len(buckets[ptype]) < targets[ptype]:
                    prob["problem_type"] = ptype
                    buckets[ptype].append(prob)
                    seen.add(sig)

    # Stratified train/eval split
    train_frac = n_train / n_total if n_total > 0 else 1.0
    train_problems: list[dict] = []
    eval_problems: list[dict] = []

    for ptype in (
        "identifiable_standard",
        "identifiable_ancestor",
        "identifiable_collider",
        "identifiable_front_door",
        "empty",
        "not_identifiable",
    ):
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
        format_fn: Callable(edges, nodes, observed_nodes, latent_nodes, X, Y) -> str
                   — renders the problem text shown to the model.
    """
    rows = []
    for p in problems:
        rows.append({
            "question": format_fn(
                p["edges"],
                p["nodes"],
                p["observed_nodes"],
                p["latent_nodes"],
                p["X"],
                p["Y"],
            ),
            "info": json.dumps({
                "minimal_adjustment_sets": p["minimal_adjustment_sets"],
                "mediator_node": p.get("mediator_node"),
                "identifiability_status": p["identifiability_status"],
                "X": p["X"],
                "Y": p["Y"],
                "edges": p["edges"],
                "nodes": p["nodes"],
                "observed_nodes": p["observed_nodes"],
                "latent_nodes": p["latent_nodes"],
                "num_nodes": p["num_nodes"],
                "num_backdoor_paths": p.get("num_backdoor_paths"),
                "num_parents_X": p.get("num_parents_X"),
                "problem_type": p.get("problem_type"),
            }),
        })
    return Dataset.from_list(rows)
