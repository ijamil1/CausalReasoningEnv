"""Causal ATE benchmark data generation.

Generates discrete CPT-based problems for the CausalReasoningEnv.
All problems provide a DAG (observed/latent nodes), node domains, and CPTs
(stored for exact probability query tools at inference time).

Problem types:
  backdoor_standard  (~35%): non-empty minimal observed adjustment set.
  backdoor_empty     (~20%): empty adjustment set (X,Y d-separated in backdoor graph).
  frontdoor          (~20%): latent confounder; valid frontdoor mediator set M.
  not_identifiable   (~25%): latent confounder; no valid backdoor or frontdoor.

Graph size: 6–12 total nodes. Node values: binary (X, Y always) or ternary.

Data fields per problem:
  problem_type            str
  edges                   list of [u, v]
  nodes                   list of int
  X, Y                    int — treatment and outcome nodes
  observed_nodes          list of int
  latent_nodes            list of int
  domains                 dict str(node_id) → list of int values
  cpts                    dict str(node_id) → CPT with "|"-joined str keys
  topo_order              list of int
  parents_map             dict str(node_id) → list of int parent ids
  identifiability_status  "identifiable" | "not_identifiable"
  true_ATE                float | None  (exact, via do-calculus enumeration)
  minimal_set             list[int] | None  (minimal adjustment or mediator set)
  optimal_turns           int  (0/1/2/2 for not_identifiable/backdoor_empty/standard/frontdoor)
"""

import json
import random
from itertools import product as itertools_product

import networkx as nx
from datasets import Dataset
from networkx.algorithms.d_separation import find_minimal_d_separator


# ─────────────────────────────────────────────────────────────────────────────
# DAG utilities
# ─────────────────────────────────────────────────────────────────────────────


def _make_dag(n: int, edge_prob: float, rng: random.Random) -> nx.DiGraph:
    """Random forward-only DAG (Erdős–Rényi, topological ordering)."""
    nodes = list(range(n))
    edges = [(u, v) for u in nodes for v in nodes if u < v and rng.random() < edge_prob]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def _make_backdoor_graph(G: nx.DiGraph, X: int) -> nx.DiGraph:
    """Return copy of G with all outgoing edges from X removed."""
    G_bd = G.copy()
    G_bd.remove_edges_from(list(G.out_edges(X)))
    return G_bd


def _add_latent_confounder(G: nx.DiGraph, X: int, Y: int, n: int) -> int:
    """Append latent node L=n to G with edges L→X and L→Y. Returns L."""
    G.add_node(n)
    G.add_edge(n, X)
    G.add_edge(n, Y)
    return n


def _check_frontdoor_conditions(G: nx.DiGraph, X: int, Y: int, M: set) -> bool:
    """Check all three frontdoor conditions for mediator set M.

    Condition 1: Y not reachable from X in the subgraph with M removed.
    Condition 2: X d-separated from M in the backdoor graph (X's outgoing edges removed) given empty set.
    Condition 3: M d-separated from Y by {X} in the graph with M's outgoing edges removed.
    """
    M = set(M)
    if not M:
        return False

    # Condition 1: Y not reachable from X without going through M
    G_minus_M = G.subgraph(set(G.nodes()) - M)
    try:
        if nx.has_path(G_minus_M, X, Y):
            return False
    except nx.NetworkXError:
        return False

    # Condition 2: X d-separated from M in backdoor graph given ∅
    G_xbar = G.copy()
    G_xbar.remove_edges_from(list(G.out_edges(X)))
    try:
        if not nx.d_separated(G_xbar, {X}, M, set()):
            return False
    except Exception:
        return False

    # Condition 3: M d-separated from Y by X in graph with M's outgoing edges removed
    G_mbar = G.copy()
    for m in M:
        G_mbar.remove_edges_from(list(G.out_edges(m)))
    try:
        if not nx.d_separated(G_mbar, M, {Y}, {X}):
            return False
    except Exception:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# CPT generation and serialization
# ─────────────────────────────────────────────────────────────────────────────


def _gen_cpt(rng: random.Random, n_cat: int, parent_cats: list[int]) -> dict:
    """Generate a CPT.

    Returns dict mapping tuple(parent_values) →
        float P(V=1|pa)       for binary nodes (n_cat=2)
        list [P(V=0), P(V=1), P(V=2)]  for ternary nodes (n_cat=3)
    """
    combos = list(itertools_product(*[range(k) for k in parent_cats])) if parent_cats else [()]
    cpt = {}
    for combo in combos:
        if n_cat == 2:
            cpt[combo] = round(rng.uniform(0.1, 0.9), 3)
        else:
            for _ in range(100):
                raw = [rng.uniform(0.15, 0.7) for _ in range(3)]
                s = sum(raw)
                probs = [round(r / s, 3) for r in raw]
                probs[2] = round(1.0 - probs[0] - probs[1], 3)
                if all(p >= 0.1 for p in probs):
                    break
            cpt[combo] = probs
    return cpt


def _serialize_cpts(cpts: dict) -> dict:
    """Convert CPT dict (tuple keys) to JSON-serializable (str keys).

    Empty tuple () → "" (root nodes with no parents).
    Non-empty tuple (0, 1) → "0|1".
    """
    result = {}
    for nd, cpt in cpts.items():
        result[str(nd)] = {
            "|".join(map(str, k)): v
            for k, v in cpt.items()
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Exact ATE computation via do-calculus enumeration
# ─────────────────────────────────────────────────────────────────────────────


def _compute_true_ate_exact(
    X: int,
    Y: int,
    cpts: dict,
    n_cats: dict,
    topo_order: list,
    parents_map: dict,
) -> float:
    """Compute true ATE exactly via do-calculus enumeration (no simulation)."""
    non_x = [nd for nd in topo_order if nd != X]

    def e_y_do_x(x_val: int) -> float:
        ey = 0.0
        for vals in itertools_product(*[range(n_cats[nd]) for nd in non_x]):
            config = dict(zip(non_x, vals))
            config[X] = x_val
            prob = 1.0
            for nd in non_x:
                pa = parents_map[nd]
                pa_vals = tuple(config[p] for p in pa)
                cpt_entry = cpts[nd][pa_vals]
                v = config[nd]
                if n_cats[nd] == 2:
                    prob *= cpt_entry if v == 1 else (1.0 - cpt_entry)
                else:
                    prob *= cpt_entry[v]
            ey += prob * config[Y]
        return ey

    return round(e_y_do_x(1) - e_y_do_x(0), 6)


# ─────────────────────────────────────────────────────────────────────────────
# Optimal turns computation
# ─────────────────────────────────────────────────────────────────────────────


def _compute_optimal_turns(problem_type: str) -> int:
    """Minimum tool calls needed for the correct identification formula.

    not_identifiable:  0  (model declares from DAG structure alone)
    backdoor_empty:    1  (one conditional call: P(Y|X))
    backdoor_standard: 2  (conditional P(Y|X,Z) + marginal P(Z))
    frontdoor:         2  (marginal P(X,M) + conditional P(Y|X,M))
    """
    if problem_type == "not_identifiable":
        return 0
    elif problem_type == "backdoor_empty":
        return 1
    elif problem_type == "backdoor_standard":
        return 2
    elif problem_type == "frontdoor":
        return 2
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Problem dict builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_problem_dict(
    G: nx.DiGraph,
    X: int,
    Y: int,
    observed_nodes: set,
    latent_nodes: set,
    n_cats: dict,
    cpts: dict,
    topo_order: list,
    parents_map: dict,
    problem_type: str,
    minimal_set: list | None,
) -> dict:
    true_ate = None
    if problem_type != "not_identifiable":
        true_ate = _compute_true_ate_exact(X, Y, cpts, n_cats, topo_order, parents_map)

    optimal_turns = _compute_optimal_turns(problem_type)

    return {
        "problem_type": problem_type,
        "edges": [[int(u), int(v)] for u, v in sorted(G.edges())],
        "nodes": [int(nd) for nd in sorted(G.nodes())],
        "X": int(X),
        "Y": int(Y),
        "observed_nodes": sorted(int(nd) for nd in observed_nodes),
        "latent_nodes": sorted(int(nd) for nd in latent_nodes),
        "domains": {str(nd): list(range(n_cats[nd])) for nd in sorted(G.nodes())},
        "cpts": _serialize_cpts(cpts),
        "topo_order": [int(nd) for nd in topo_order],
        "parents_map": {str(nd): [int(p) for p in parents_map[nd]] for nd in G.nodes()},
        "identifiability_status": "not_identifiable" if problem_type == "not_identifiable" else "identifiable",
        "true_ATE": float(true_ate) if true_ate is not None else None,
        "minimal_set": [int(nd) for nd in minimal_set] if minimal_set is not None else None,
        "optimal_turns": optimal_turns,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Problem samplers
# ─────────────────────────────────────────────────────────────────────────────


def _try_sample_backdoor(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
    latent_prob: float,
    empty: bool = False,
) -> dict | None:
    """Sample a backdoor_standard or backdoor_empty problem.

    Method ambiguity check: reject if a valid frontdoor set also exists.
    """
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, edge_prob, rng)
    nodes_list = sorted(G.nodes())
    if len(nodes_list) < 3:
        return None

    X, Y = rng.sample(nodes_list, 2)
    if not nx.has_path(G, X, Y):
        return None
    if G.out_degree(Y) > 0:
        return None
    if nx.has_path(G, Y, X):
        return None

    other_nodes = [nd for nd in nodes_list if nd != X and nd != Y]
    latent_nodes = {nd for nd in other_nodes if rng.random() < latent_prob}
    observed_nodes = set(nodes_list) - latent_nodes

    G_bd = _make_backdoor_graph(G, X)
    try:
        min_set = find_minimal_d_separator(G_bd, X, Y, restricted=observed_nodes - {X, Y})
    except Exception:
        return None

    if min_set is None:
        return None
    if empty and len(min_set) != 0:
        return None
    if not empty and len(min_set) == 0:
        return None
    if len(min_set) > 2:  # cap for tractable problems
        return None
    if min_set & nx.descendants(G, X):  # no descendants of X in adjustment set
        return None

    # Eliminate method ambiguity: verify no valid frontdoor set exists
    if not G.has_edge(X, Y):
        M_temp = frozenset(v for v in G.successors(X) if nx.has_path(G, v, Y))
        if M_temp and _check_frontdoor_conditions(G, X, Y, M_temp):
            return None  # frontdoor also valid via M_temp, resample

    adjustment_set = sorted(min_set)

    n_cats = {}
    for nd in nodes_list:
        if nd == X or nd == Y or nd in latent_nodes:
            n_cats[nd] = 2
        else:
            n_cats[nd] = 2 if rng.random() < 0.6 else 3

    topo_order = list(nx.topological_sort(G))
    parents_map = {nd: sorted(G.predecessors(nd)) for nd in G.nodes()}
    cpts = {
        nd: _gen_cpt(rng, n_cats[nd], [n_cats[p] for p in parents_map[nd]])
        for nd in topo_order
    }

    ptype = "backdoor_empty" if empty else "backdoor_standard"
    return _build_problem_dict(
        G, X, Y, observed_nodes, latent_nodes,
        n_cats, cpts, topo_order, parents_map,
        ptype, adjustment_set,
    )


def _try_sample_frontdoor(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
) -> dict | None:
    """Sample a frontdoor problem with a multi-node mediator set.

    Uses minimum_node_cut to find the minimal mediator set M_star. Verifies
    all three frontdoor conditions and that no valid backdoor adjustment set exists.
    """
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, edge_prob, rng)
    nodes_list = sorted(G.nodes())
    if len(nodes_list) < 4:
        return None

    # Pick X as a source (no parents in original DAG) and Y as a sink (no children)
    # with a path from X to Y but no direct X→Y edge.
    sources = [nd for nd in nodes_list if G.in_degree(nd) == 0]
    sinks = [nd for nd in nodes_list if G.out_degree(nd) == 0]
    if not sources or not sinks:
        return None

    X = rng.choice(sources)
    valid_sinks = [
        nd for nd in sinks
        if nd != X and not G.has_edge(X, nd) and nx.has_path(G, X, nd)
    ]
    if not valid_sinks:
        return None
    Y = rng.choice(valid_sinks)

    # Add latent confounder L→X, L→Y
    L = _add_latent_confounder(G, X, Y, n)
    observed_nodes = set(range(n))
    latent_nodes = {L}
    nodes_list = sorted(G.nodes())

    # Build descendant subgraph of X to find mediator set
    desc = nx.descendants(G, X) | {X, Y}
    G_desc = G.subgraph(desc).copy()

    # Find minimal mediator set via minimum vertex cut
    try:
        M_star = nx.minimum_node_cut(G_desc, X, Y)
    except Exception:
        return None

    k = len(M_star)
    if k == 0 or k > 3:
        return None

    # All mediators must be observed
    if M_star & latent_nodes:
        return None

    # Verify all three frontdoor conditions for M_star
    if not _check_frontdoor_conditions(G, X, Y, M_star):
        return None

    # Verify no valid backdoor adjustment set exists among observed nodes
    G_bd = _make_backdoor_graph(G, X)
    try:
        Z = find_minimal_d_separator(G_bd, X, Y, restricted=observed_nodes - {X, Y})
    except Exception:
        return None
    if Z is not None:
        return None  # backdoor adjustment set exists, resample

    minimal_set = sorted(M_star)

    n_cats = {}
    for nd in nodes_list:
        if nd == X or nd == Y or nd == L:
            n_cats[nd] = 2
        else:
            n_cats[nd] = 2 if rng.random() < 0.6 else 3

    topo_order = list(nx.topological_sort(G))
    parents_map = {nd: sorted(G.predecessors(nd)) for nd in G.nodes()}
    cpts = {
        nd: _gen_cpt(rng, n_cats[nd], [n_cats[p] for p in parents_map[nd]])
        for nd in topo_order
    }

    return _build_problem_dict(
        G, X, Y, observed_nodes, latent_nodes,
        n_cats, cpts, topo_order, parents_map,
        "frontdoor", minimal_set,
    )


def _try_sample_not_identifiable(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
) -> dict | None:
    """Sample a not_identifiable problem.

    Requires a direct X→Y edge (blocks frontdoor) and a latent confounder
    L→X, L→Y (blocks backdoor). Verifies find_minimal_d_separator returns None.
    """
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, edge_prob, rng)

    # Need a direct X→Y edge (rules out frontdoor criterion)
    candidates = [(u, v) for u, v in G.edges() if G.out_degree(v) == 0]
    if not candidates:
        return None

    X, Y = rng.choice(candidates)
    L = _add_latent_confounder(G, X, Y, n)
    nodes_list = sorted(G.nodes())
    observed_nodes = set(range(n))
    latent_nodes = {L}

    G_bd = _make_backdoor_graph(G, X)
    try:
        check = find_minimal_d_separator(G_bd, X, Y, restricted=observed_nodes - {X, Y})
    except Exception:
        return None
    if check is not None:
        return None

    n_cats = {nd: 2 for nd in nodes_list}
    topo_order = list(nx.topological_sort(G))
    parents_map = {nd: sorted(G.predecessors(nd)) for nd in G.nodes()}
    cpts = {
        nd: _gen_cpt(rng, n_cats[nd], [n_cats[p] for p in parents_map[nd]])
        for nd in topo_order
    }

    return _build_problem_dict(
        G, X, Y, observed_nodes, latent_nodes,
        n_cats, cpts, topo_order, parents_map,
        "not_identifiable", None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────


def format_problem(p: dict) -> str:
    """Render a problem as a user-facing text prompt."""
    nodes = p["nodes"]
    observed = p["observed_nodes"]
    latent = p["latent_nodes"]
    X = p["X"]
    Y = p["Y"]
    domains = p["domains"]  # str keys
    edges = p["edges"]

    has_latent = bool(latent)
    obs_str = ", ".join(str(nd) for nd in observed)
    lat_str = ", ".join(str(nd) for nd in latent) if latent else "none"
    edge_str = ", ".join(f"{u}→{v}" for u, v in edges)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    adj_lines = []
    for nd in nodes:
        pa = sorted(G.predecessors(nd))
        ch = sorted(G.successors(nd))
        kind = "latent" if nd in latent else "observed"
        label = f" ({kind})" if has_latent else ""
        adj_lines.append(
            f"  Node {nd}{label}: parents=[{', '.join(map(str, pa))}], "
            f"children=[{', '.join(map(str, ch))}]"
        )

    domain_lines = []
    for nd in nodes:
        tag = ""
        if nd == X:
            tag = " (X, treatment)"
        elif nd == Y:
            tag = " (Y, outcome)"
        if nd in latent:
            tag += " [latent — not queryable]"
        d = domains[str(nd)]
        domain_lines.append(f"  Node {nd}{tag}: {{{', '.join(map(str, d))}}}")

    dag_section = (
        f"DAG INFORMATION\n"
        f"───────────────\n"
        f"Nodes:    {', '.join(str(nd) for nd in nodes)}\n"
        f"Observed: {obs_str}\n"
        f"Latent:   {lat_str}\n"
        f"Edges:    {edge_str}\n\n"
        f"Adjacency:\n" + "\n".join(adj_lines)
    )

    domain_section = (
        f"\n\nNODE DOMAINS\n"
        f"────────────\n" + "\n".join(domain_lines)
    )

    footer = (
        f"\n\nTreatment (X): Node {X}\n"
        f"Outcome   (Y): Node {Y}"
    )

    return dag_section + domain_section + footer


def build_dataset(problems: list[dict]) -> Dataset:
    """Convert a list of problem dicts into a HuggingFace Dataset."""
    rows = []
    for p in problems:
        true_ate = p["true_ATE"]
        if p["identifiability_status"] == "identifiable" and true_ate is not None:
            answer = f"ATE={round(true_ate, 4)}"
        else:
            answer = "not_identifiable"

        rows.append({
            "question": format_problem(p),
            "answer": answer,
            "info": json.dumps({
                "problem_type": p["problem_type"],
                "identifiability_status": p["identifiability_status"],
                "true_ATE": p["true_ATE"],
                "optimal_turns": p["optimal_turns"],
                "minimal_set": p["minimal_set"],
                "X": p["X"],
                "Y": p["Y"],
                "nodes": p["nodes"],
                "edges": p["edges"],
                "observed_nodes": p["observed_nodes"],
                "latent_nodes": p["latent_nodes"],
                "domains": p["domains"],
                "cpts": p["cpts"],
                "topo_order": p["topo_order"],
                "parents_map": p["parents_map"],
            }),
        })
    return Dataset.from_list(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────


def generate_problems(
    n: int = 1000,
    seed: int = 42,
    min_nodes: int = 6,
    max_nodes: int = 12,
) -> list[dict]:
    """Generate a problem set. Returns a list of problem dicts."""
    rng = random.Random(seed)

    fracs = {
        "backdoor_standard": 0.35,
        "backdoor_empty": 0.20,
        "frontdoor": 0.20,
        "not_identifiable": 0.25,
    }

    def _sample_bucket(ptype: str, n_target: int) -> list[dict]:
        problems = []
        max_attempts = n_target * 500
        for _ in range(max_attempts):
            if len(problems) >= n_target:
                break
            if ptype == "backdoor_standard":
                p = _try_sample_backdoor(rng, min_nodes, max_nodes, 0.4, 0.2, empty=False)
            elif ptype == "backdoor_empty":
                p = _try_sample_backdoor(rng, min_nodes, max_nodes, 0.4, 0.0, empty=True)
            elif ptype == "frontdoor":
                p = _try_sample_frontdoor(rng, min_nodes, max_nodes, 0.4)
            elif ptype == "not_identifiable":
                p = _try_sample_not_identifiable(rng, min_nodes, max_nodes, 0.4)
            else:
                p = None
            if p is not None:
                problems.append(p)
        return problems

    problems = []
    for ptype, frac in fracs.items():
        n_target = max(1, round(n * frac))
        problems.extend(_sample_bucket(ptype, n_target))
    rng.shuffle(problems)
    return problems
