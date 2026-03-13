"""Causal ATE/LATE benchmark data generation.

Generates discrete CPT-based problems for the CausalReasoningEnv.
All problems provide a DAG (observed/latent nodes), node domains, and CPTs
(stored for exact probability query tools at inference time).

Problem types (mutually exclusive — exactly one method applies per problem):
  backdoor_standard  (~35%): non-empty minimal observed adjustment set.
  backdoor_empty     (~15%): empty adjustment set (X,Y d-separated in backdoor graph).
  frontdoor          (~40%): latent confounder; valid frontdoor mediator set M.
  iv                 (~10%): latent confounder; no backdoor/frontdoor; IV instrument Z.

Variable domains:
  X: always {0, 1}
  Y: always {0, 1, 2, 3, 4}  (ATE ∈ [-4, 4])
  Non-{X,Y} observed: binary {k, k+1} or ternary {k, k+1, k+2} with random shift k
  Latent nodes: always {0, 1}

Y is NOT required to be a leaf node.

Data fields per problem:
  problem_type            str  ("backdoor_standard"|"backdoor_empty"|"frontdoor"|"iv")
  identification_methods  list[str]  (single-element, maps problem_type to method)
  edges                   list of [u, v]
  nodes                   list of int
  X, Y                    int — treatment and outcome nodes
  observed_nodes          list of int
  latent_nodes            list of int
  domains                 dict str(node_id) → list of int values
  cpts                    dict str(node_id) → CPT with "|"-joined str keys
  topo_order              list of int
  parents_map             dict str(node_id) → list of int parent ids
  true_ATE                float | None  (None for iv problems)
  true_LATE               float | None  (None for non-iv problems; Wald estimator)
  minimal_set             list[int] | None
  iv_instrument           int | None  (non-None for iv problems)
"""

import json
import random
from itertools import product as itertools_product

import networkx as nx
from datasets import Dataset
from networkx.algorithms.d_separation import is_d_separator, find_minimal_d_separator


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


def _check_frontdoor_conditions(G: nx.DiGraph, X: int, Y: int, M: set) -> bool | str:
    """Check all three frontdoor conditions for mediator set M.

    Condition 1: Y not reachable from X in the subgraph with M removed.
    Condition 2: X d-separated from M in the backdoor graph (X's outgoing edges removed) given empty set.
    Condition 3: M d-separated from Y by {X} in the graph with M's outgoing edges removed.

    Returns True if all conditions pass, or a string naming the failing condition.
    """
    M = set(M)
    if not M:
        return "cond0_empty_M"

    # Condition 1: Y not reachable from X without going through M
    G_minus_M = G.subgraph(set(G.nodes()) - M)
    try:
        if nx.has_path(G_minus_M, X, Y):
            return "cond1_X_reaches_Y_without_M"
    except nx.NetworkXError:
        return "cond1_networkx_error"

    # Condition 2: X d-separated from M in backdoor graph given ∅
    G_xbar = G.copy()
    G_xbar.remove_edges_from(list(G.out_edges(X)))
    try:
        if not is_d_separator(G_xbar, {X}, M, set()):
            return "cond2_X_not_dsep_from_M"
    except Exception as e:
        return f"cond2_exception:{type(e).__name__}:{e}"

    # Condition 3: M d-separated from Y by X in graph with M's outgoing edges removed
    G_mbar = G.copy()
    for m in M:
        G_mbar.remove_edges_from(list(G.out_edges(m)))
    try:
        if not is_d_separator(G_mbar, M, {Y}, {X}):
            return "cond3_M_not_dsep_from_Y"
    except Exception:
        return "cond3_exception"

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Verifier functions (exported — also used by env.py for reward computation)
# ─────────────────────────────────────────────────────────────────────────────


def is_valid_backdoor_set(
    G: nx.DiGraph, X: int, Y: int, observed_nodes: set, Z: list | set
) -> bool:
    """Return True iff Z is a valid backdoor adjustment set for X→Y.

    Z must:
    - Be a subset of observed_nodes (excluding X and Y)
    - Not contain any descendants of X
    - d-separate X and Y in the backdoor graph (G with X's outgoing edges removed)
    """
    Z = set(Z)
    if not Z.issubset(observed_nodes):
        return False
    if X in Z or Y in Z:
        return False
    if Z & nx.descendants(G, X):
        return False
    G_bd = _make_backdoor_graph(G, X)
    try:
        return is_d_separator(G_bd, {X}, {Y}, Z)
    except Exception:
        return False


def is_valid_frontdoor_set(
    G: nx.DiGraph, X: int, Y: int, observed_nodes: set, M: list | set
) -> bool:
    """Return True iff M is a valid frontdoor mediator set for X→Y."""
    M = set(M)
    if not M:
        return False
    if not M.issubset(observed_nodes):
        return False
    return _check_frontdoor_conditions(G, X, Y, M) is True


def is_valid_iv(
    G: nx.DiGraph, X: int, Y: int, observed_nodes: set, Z_iv: int
) -> bool:
    """Return True iff Z_iv is a valid instrumental variable for X→Y.

    Conditions:
    1. Z_iv is observed, not X, not Y
    2. Z_iv can reach X (directed path Z_iv→...→X)
    3. Exclusion restriction: Z_iv cannot reach Y in G without X
       (i.e., Z_iv and Y are d-separated when X is removed)
    4. Exogeneity: Z_iv has no latent ancestors
    """
    if Z_iv not in observed_nodes:
        return False
    if Z_iv == X or Z_iv == Y:
        return False
    if not nx.has_path(G, Z_iv, X):
        return False
    # Exclusion restriction: remove X and check if Z_iv can still reach Y
    G_no_X = G.copy()
    G_no_X.remove_node(X)
    try:
        if nx.has_path(G_no_X, Z_iv, Y):
            return False
    except nx.NodeNotFound:
        pass
    # Exogeneity: no latent ancestors of Z_iv
    latent_nodes = set(G.nodes()) - set(observed_nodes)
    if nx.ancestors(G, Z_iv) & latent_nodes:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Domain assignment
# ─────────────────────────────────────────────────────────────────────────────


def _sample_domain(cardinality: int, rng: random.Random) -> list[int]:
    """Return a domain list of the given cardinality with a random integer shift.

    Binary:  {k, k+1}   where k ∈ [-4, 4]
    Ternary: {k, k+1, k+2} where k ∈ [-4, 3]
    """
    if cardinality == 2:
        k = rng.randint(-4, 4)
        return [k, k + 1]
    elif cardinality == 3:
        k = rng.randint(-4, 3)
        return [k, k + 1, k + 2]
    else:
        raise ValueError(f"Unsupported cardinality: {cardinality}")


# ─────────────────────────────────────────────────────────────────────────────
# CPT generation and serialization
# ─────────────────────────────────────────────────────────────────────────────


def _gen_cpt(
    rng: random.Random,
    domain: list[int],
    parent_domains: list[list[int]],
) -> dict:
    """Generate a CPT using actual domain values (not 0-indexed).

    Returns dict mapping tuple(actual_parent_values) →
        float P(V=domain[1]|pa)                    for binary nodes (|domain|=2)
        list [P(V=domain[0]|pa), ..., P(V=domain[-1]|pa)]  for multi-valued nodes

    Y (|domain|=5) uses min_prob=0.05 to allow wider ATE range.
    Other multi-valued nodes use min_prob=0.10.
    """
    combos = list(itertools_product(*parent_domains)) if parent_domains else [()]
    n_cat = len(domain)
    cpt = {}

    for combo in combos:
        if n_cat == 2:
            cpt[combo] = round(rng.uniform(0.1, 0.9), 3)

        elif n_cat == 5:
            # Y's 5-valued domain; lower min_prob for wider ATE spread
            min_p = 0.05
            probs_r = None
            for _ in range(200):
                raw = [rng.uniform(0.05, 1.0) for _ in range(5)]
                s = sum(raw)
                probs = [r / s for r in raw]
                if all(p >= min_p for p in probs):
                    pr = [round(p, 4) for p in probs]
                    pr[-1] = round(1.0 - sum(pr[:-1]), 4)
                    if all(p >= min_p for p in pr):
                        probs_r = pr
                        break
            cpt[combo] = probs_r if probs_r is not None else [0.2, 0.2, 0.2, 0.2, 0.2]

        else:
            # Ternary or other multi-valued: min_prob = 0.10
            min_p = 0.1
            probs_r = None
            for _ in range(100):
                raw = [rng.uniform(0.15, 0.7) for _ in range(n_cat)]
                s = sum(raw)
                probs = [r / s for r in raw]
                pr = [round(p, 3) for p in probs]
                pr[-1] = round(1.0 - sum(pr[:-1]), 3)
                if all(p >= min_p for p in pr):
                    probs_r = pr
                    break
            if probs_r is None:
                probs_r = [round(1.0 / n_cat, 3)] * n_cat
            cpt[combo] = probs_r

    return cpt


def _serialize_cpts(cpts: dict) -> dict:
    """Convert CPT dict (tuple keys of actual values) to JSON-serializable (str keys).

    Empty tuple () → "" (root nodes with no parents).
    Non-empty tuple (e.g. (-1, 2)) → "-1|2".
    Negative values are handled correctly since we split on "|" not "-".
    """
    result = {}
    for nd, cpt in cpts.items():
        result[str(nd)] = {
            "|".join(map(str, k)): v
            for k, v in cpt.items()
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Exact ATE and LATE computation
# ─────────────────────────────────────────────────────────────────────────────


def _compute_true_ate_exact(
    X: int,
    Y: int,
    cpts: dict,
    domains: dict,
    topo_order: list,
    parents_map: dict,
) -> float:
    """Compute true ATE exactly via do-calculus enumeration (no simulation).

    Enumerates ALL non-X variables (including Y and Y's children if any),
    so this is correct even when Y is not a leaf node.

    ATE = E[Y | do(X=1)] - E[Y | do(X=0)]

    domains: dict node_id → list of actual int values
    cpts: dict node_id → {tuple(actual_parent_values): prob_or_list}
    """
    non_x = [nd for nd in topo_order if nd != X]

    def e_y_do_x(x_val: int) -> float:
        ey = 0.0
        for vals in itertools_product(*[domains[nd] for nd in non_x]):
            config = dict(zip(non_x, vals))
            config[X] = x_val
            prob = 1.0
            for nd in non_x:
                pa_vals = tuple(config[p] for p in parents_map[nd])
                cpt_entry = cpts[nd][pa_vals]
                v = config[nd]
                dom = domains[nd]
                if len(dom) == 2:
                    prob *= cpt_entry if v == dom[1] else (1.0 - cpt_entry)
                else:
                    prob *= cpt_entry[dom.index(v)]
            ey += config[Y] * prob
        return ey

    return round(e_y_do_x(1) - e_y_do_x(0), 6)


def _compute_true_late(
    Z: int,
    X: int,
    Y: int,
    cpts: dict,
    domains: dict,
    topo_order: list,
    parents_map: dict,
) -> float | None:
    """Compute true LATE via Wald estimator on the exact observational distribution.

    LATE = (E[Y|Z=z1] - E[Y|Z=z0]) / (E[X|Z=z1] - E[X|Z=z0])

    Since Z is exogenous (no parents), E[Y|Z=z] = E[Y|do(Z=z)], computed by
    fixing Z and enumerating all other variables. Latent variables are
    marginalized over automatically.

    Returns None if the IV is degenerate (denominator ≈ 0).
    """
    non_z = [nd for nd in topo_order if nd != Z]
    z_domain = domains[Z]  # binary IV: [z0, z1]

    def expectations_given_z(z_val: int):
        ey = ex = 0.0
        for vals in itertools_product(*[domains[nd] for nd in non_z]):
            config = dict(zip(non_z, vals))
            config[Z] = z_val
            prob = 1.0
            for nd in non_z:
                pa_vals = tuple(config[p] for p in parents_map[nd])
                cpt_entry = cpts[nd][pa_vals]
                v = config[nd]
                dom = domains[nd]
                if len(dom) == 2:
                    prob *= cpt_entry if v == dom[1] else (1.0 - cpt_entry)
                else:
                    prob *= cpt_entry[dom.index(v)]
            ey += config[Y] * prob  # Y ∈ {0,1,2,3,4}
            ex += config[X] * prob  # X ∈ {0,1}; E[X|Z=z] = P(X=1|Z=z)
        return ey, ex

    ey1, ex1 = expectations_given_z(z_domain[1])
    ey0, ex0 = expectations_given_z(z_domain[0])
    denom = ex1 - ex0
    if abs(denom) < 1e-10:
        return None  # degenerate IV: Z has no effect on X
    return round((ey1 - ey0) / denom, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Problem dict builder
# ─────────────────────────────────────────────────────────────────────────────

_PTYPE_TO_METHOD = {
    "backdoor_standard": "backdoor",
    "backdoor_empty": "backdoor",
    "frontdoor": "frontdoor",
    "iv": "iv",
}


def _build_problem_dict(
    G: nx.DiGraph,
    X: int,
    Y: int,
    observed_nodes: set,
    latent_nodes: set,
    domains: dict,
    cpts: dict,
    topo_order: list,
    parents_map: dict,
    problem_type: str,
    minimal_set: list | None,
    iv_instrument: int | None = None,
) -> dict:
    """Build the canonical problem dict.

    For non-iv problems: computes true_ATE via exact enumeration.
    For iv problems: computes true_LATE via Wald estimator.
    """
    true_ate = None
    true_late = None

    if problem_type == "iv":
        assert iv_instrument is not None
        true_late = _compute_true_late(iv_instrument, X, Y, cpts, domains, topo_order, parents_map)
    else:
        true_ate = _compute_true_ate_exact(X, Y, cpts, domains, topo_order, parents_map)

    return {
        "problem_type": problem_type,
        "identification_methods": [_PTYPE_TO_METHOD[problem_type]],
        "edges": [[int(u), int(v)] for u, v in sorted(G.edges())],
        "nodes": [int(nd) for nd in sorted(G.nodes())],
        "X": int(X),
        "Y": int(Y),
        "observed_nodes": sorted(int(nd) for nd in observed_nodes),
        "latent_nodes": sorted(int(nd) for nd in latent_nodes),
        "domains": {str(nd): list(domains[nd]) for nd in sorted(G.nodes())},
        "cpts": _serialize_cpts(cpts),
        "topo_order": [int(nd) for nd in topo_order],
        "parents_map": {str(nd): [int(p) for p in parents_map[nd]] for nd in G.nodes()},
        "true_ATE": float(true_ate) if true_ate is not None else None,
        "true_LATE": float(true_late) if true_late is not None else None,
        "minimal_set": [int(nd) for nd in minimal_set] if minimal_set is not None else None,
        "iv_instrument": int(iv_instrument) if iv_instrument is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Domain assignment helper (shared across all samplers)
# ─────────────────────────────────────────────────────────────────────────────


def _assign_domains(
    nodes_list: list,
    X: int,
    Y: int,
    latent_nodes: set,
    rng: random.Random,
) -> dict:
    """Assign domains to all nodes.

    X → [0, 1]
    Y → [0, 1, 2, 3, 4]
    Latent → [0, 1]
    Other observed: binary {k,k+1} (60%) or ternary {k,k+1,k+2} (40%) with random k
    """
    domains = {}
    for nd in nodes_list:
        if nd == X:
            domains[nd] = [0, 1]
        elif nd == Y:
            domains[nd] = [0, 1, 2, 3, 4]
        elif nd in latent_nodes:
            domains[nd] = [0, 1]
        else:
            cardinality = 2 if rng.random() < 0.6 else 3
            domains[nd] = _sample_domain(cardinality, rng)
    return domains


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

    Y may be a non-leaf (no out_degree constraint).
    Rejects if a valid frontdoor set also exists (backdoor/frontdoor are mutually exclusive).
    IV ambiguity is not checked — the system prompt directs models to prefer ATE methods.
    """
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, edge_prob, rng)
    nodes_list = sorted(G.nodes())
    if len(nodes_list) < 3:
        return None

    X, Y = rng.sample(nodes_list, 2)
    if not nx.has_path(G, X, Y):
        return None
    # Removed: if G.out_degree(Y) > 0: return None  (Y non-leaf is now allowed)
    if nx.has_path(G, Y, X):
        return None

    other_nodes = [nd for nd in nodes_list if nd != X and nd != Y]
    latent_nodes = {nd for nd in other_nodes if rng.random() < latent_prob}

    # For non-empty backdoor problems, ensure at least one parent of X is latent
    # so the model cannot trivially return pa(X) as the adjustment set.
    if not empty:
        parents_of_x = [nd for nd in G.predecessors(X) if nd != Y]
        if parents_of_x and not (set(parents_of_x) & latent_nodes):
            latent_nodes.add(rng.choice(parents_of_x))

    observed_nodes = set(nodes_list) - latent_nodes

    G_bd = _make_backdoor_graph(G, X)

    if empty:
        # X must have parents (non-trivial structure)
        if G.in_degree(X) == 0:
            return None
        # Must have an undirected path in G_bd between X and Y;
        # combined with d-separation given {}, this means paths exist but are collider-blocked
        if not nx.has_path(G_bd.to_undirected(), X, Y):
            return None

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
    if not empty and min_set.issubset(set(G.predecessors(X))):  # reject trivial pa(X) adjustment
        return None

    # Eliminate frontdoor ambiguity: verify no valid frontdoor set exists
    if not G.has_edge(X, Y):
        M_temp = frozenset(
            v for v in G.successors(X)
            if v in observed_nodes and nx.has_path(G, v, Y)
        )
        if M_temp and _check_frontdoor_conditions(G, X, Y, M_temp) is True:
            return None  # frontdoor also valid, resample

    adjustment_set = sorted(min_set)

    domains = _assign_domains(nodes_list, X, Y, latent_nodes, rng)
    topo_order = list(nx.topological_sort(G))
    parents_map = {nd: sorted(G.predecessors(nd)) for nd in G.nodes()}
    cpts = {
        nd: _gen_cpt(rng, domains[nd], [domains[p] for p in parents_map[nd]])
        for nd in topo_order
    }

    ptype = "backdoor_empty" if empty else "backdoor_standard"
    return _build_problem_dict(
        G, X, Y, observed_nodes, latent_nodes,
        domains, cpts, topo_order, parents_map,
        ptype, adjustment_set,
    )


def _try_sample_frontdoor(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
) -> dict | None:
    """Sample a frontdoor problem.

    Y may be a non-leaf (no out_degree constraint).
    X→Y direct edge is still forbidden (frontdoor condition 1 would fail).
    Uses minimum node cut to find M_star; removes external incoming edges to
    ensure all three frontdoor conditions hold. Verifies no backdoor set exists
    (backdoor/frontdoor are mutually exclusive).
    IV ambiguity is not checked — the system prompt directs models to prefer ATE methods.
    """
    n = rng.randint(min_nodes, max_nodes)
    i = 0
    proceed_flag = False
    while i < 50:
        G = _make_dag(n, edge_prob, rng)
        nodes_list = sorted(G.nodes())
        X, Y = rng.sample(nodes_list, 2)
        if not nx.has_path(G, X, Y):
            i += 1
            continue
        # Removed: if G.out_degree(Y) > 0: continue  (Y non-leaf allowed)
        if G.has_edge(X, Y):
            i += 1
            continue
        proceed_flag = True
        break

    if not proceed_flag:
        return None

    if len(nodes_list) < 4:
        return None

    # Add latent confounder L→X, L→Y
    L = _add_latent_confounder(G, X, Y, n)
    observed_nodes = set(range(n))
    latent_nodes = {L}
    nodes_list = sorted(G.nodes())

    # Build descendant subgraph of X and find the minimum vertex cut (M_star).
    desc_of_X = nx.descendants(G, X)
    G_desc = G.subgraph(desc_of_X | {X, Y}).copy()
    try:
        M_star = nx.minimum_node_cut(G_desc, X, Y)
    except Exception:
        return None

    k = len(M_star)
    if k == 0:
        return None
    if k > 3:
        return None
    if M_star & latent_nodes:
        return None

    # Remove external (non-X, non-descendant-of-X) incoming edges to M_star nodes
    # and to all X-descendant ancestors of M_star (the full X→M_star pathway).
    anc_of_M_star = set()
    for m in M_star:
        anc_of_M_star |= nx.ancestors(G, m)
    pathway_nodes = (desc_of_X & anc_of_M_star) - M_star

    for node in M_star | pathway_nodes:
        for parent in list(G.predecessors(node)):
            if parent != X and parent not in desc_of_X and parent not in M_star:
                G.remove_edge(parent, node)

    # Verify all frontdoor conditions hold after edge removals
    _fd_result = _check_frontdoor_conditions(G, X, Y, M_star)
    if _fd_result is not True:
        return None

    # Verify M_star is still a minimum node cut
    G_desc_post = G.subgraph(desc_of_X | {X, Y}).copy()
    try:
        post_cut = nx.minimum_node_cut(G_desc_post, X, Y)
    except Exception:
        return None
    if len(post_cut) != len(M_star):
        return None

    # Mutual exclusivity: no valid backdoor adjustment set
    G_bd = _make_backdoor_graph(G, X)
    try:
        Z = find_minimal_d_separator(G_bd, X, Y, restricted=observed_nodes - {X, Y})
    except Exception:
        return None
    if Z is not None:
        return None

    minimal_set = sorted(M_star)

    domains = _assign_domains(nodes_list, X, Y, latent_nodes, rng)
    topo_order = list(nx.topological_sort(G))
    parents_map = {nd: sorted(G.predecessors(nd)) for nd in G.nodes()}
    cpts = {
        nd: _gen_cpt(rng, domains[nd], [domains[p] for p in parents_map[nd]])
        for nd in topo_order
    }

    return _build_problem_dict(
        G, X, Y, observed_nodes, latent_nodes,
        domains, cpts, topo_order, parents_map,
        "frontdoor", minimal_set,
    )


def _try_sample_iv(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
) -> dict | None:
    """Sample an IV problem.

    Adds a fresh exogenous IV node Z (Z→X only) and a latent confounder L (L→X, L→Y).
    Ensures:
    - Z is the unique valid IV among X's observed direct parents
    - No valid backdoor adjustment set exists
    - No valid frontdoor mediator set exists
    Y may be a non-leaf.
    """
    n = rng.randint(min_nodes, max_nodes)
    G = _make_dag(n, edge_prob, rng)
    nodes_list = sorted(G.nodes())
    if len(nodes_list) < 3:
        return None

    X, Y = rng.sample(nodes_list, 2)
    if not nx.has_path(G, X, Y):
        return None
    if nx.has_path(G, Y, X):
        return None

    # Add fresh exogenous IV node Z (no parents, only edge Z→X)
    Z_iv = n
    G.add_node(Z_iv)
    G.add_edge(Z_iv, X)

    # Add latent confounder L→X, L→Y
    L = n + 1
    G.add_node(L)
    G.add_edge(L, X)
    G.add_edge(L, Y)

    all_nodes = sorted(G.nodes())
    observed_nodes = set(range(n)) | {Z_iv}  # original nodes + IV; L is latent
    latent_nodes = {L}

    # Verify Z_iv satisfies IV conditions
    if not is_valid_iv(G, X, Y, observed_nodes, Z_iv):
        return None

    # Uniqueness: among X's observed direct parents, only Z_iv may be a valid IV
    for parent in list(G.predecessors(X)):
        if parent == Z_iv or parent not in observed_nodes:
            continue
        if is_valid_iv(G, X, Y, observed_nodes, parent):
            return None  # another observed parent is also a valid IV, reject

    # Mutual exclusivity: no valid backdoor adjustment set
    G_bd = _make_backdoor_graph(G, X)
    try:
        Z_bd = find_minimal_d_separator(G_bd, X, Y, restricted=observed_nodes - {X, Y})
    except Exception:
        return None
    if Z_bd is not None:
        return None

    # Mutual exclusivity: no valid frontdoor mediator set
    if not G.has_edge(X, Y):
        desc_of_X = nx.descendants(G, X)
        M_temp = frozenset(
            v for v in G.successors(X)
            if v in observed_nodes and v in desc_of_X and nx.has_path(G, v, Y)
        )
        if M_temp and _check_frontdoor_conditions(G, X, Y, M_temp) is True:
            return None

    domains = _assign_domains(all_nodes, X, Y, latent_nodes, rng)
    # IV instrument Z gets domain {0,1}
    domains[Z_iv] = [0, 1]

    topo_order = list(nx.topological_sort(G))
    parents_map = {nd: sorted(G.predecessors(nd)) for nd in G.nodes()}
    cpts = {
        nd: _gen_cpt(rng, domains[nd], [domains[p] for p in parents_map[nd]])
        for nd in topo_order
    }

    # Enforce monotonicity of X's CPT w.r.t. Z_iv so that no defiers exist,
    # making the Wald estimator equal to the true LATE (Angrist & Imbens 1994).
    # For each stratum of X's non-Z parents, sample p0 <= p1 such that
    # P(X=1 | Z=z_hi, others) >= P(X=1 | Z=z_lo, others) for all others.
    x_parents = parents_map[X]
    z_idx = x_parents.index(Z_iv)
    non_z_parents = [p for i, p in enumerate(x_parents) if i != z_idx]
    non_z_domains = [domains[p] for p in non_z_parents]
    non_z_combos = list(itertools_product(*non_z_domains)) if non_z_domains else [()]
    z_lo, z_hi = domains[Z_iv][0], domains[Z_iv][1]
    monotone_x_cpt = {}
    for non_z_vals in non_z_combos:
        p0 = round(rng.uniform(0.1, 0.9), 3)
        p1 = round(rng.uniform(p0, 0.9), 3)
        key_lo = non_z_vals[:z_idx] + (z_lo,) + non_z_vals[z_idx:]
        key_hi = non_z_vals[:z_idx] + (z_hi,) + non_z_vals[z_idx:]
        monotone_x_cpt[key_lo] = p0
        monotone_x_cpt[key_hi] = p1
    cpts[X] = monotone_x_cpt

    return _build_problem_dict(
        G, X, Y, observed_nodes, latent_nodes,
        domains, cpts, topo_order, parents_map,
        "iv", [Z_iv], iv_instrument=Z_iv,
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
    edge_str = ", ".join(f"{u}->{v}" for u, v in edges)

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

    reminder = (
        f"\n\nYour task: compute the causal effect of X (Node {X}) on Y (Node {Y}).\n"
        f"Follow the instructions and response format specified in the system prompt."
    )

    return dag_section + domain_section + footer + reminder


def build_dataset(problems: list[dict]) -> Dataset:
    """Convert a list of problem dicts into a HuggingFace Dataset."""
    rows = []
    for p in problems:
        ptype = p["problem_type"]
        if ptype == "iv":
            true_val = p["true_LATE"]
            answer = f"LATE={round(true_val, 4)}" if true_val is not None else "unknown"
        else:
            true_val = p["true_ATE"]
            answer = f"ATE={round(true_val, 4)}" if true_val is not None else "unknown"

        rows.append({
            "question": format_problem(p),
            "answer": answer,
            "info": json.dumps({
                "problem_type": p["problem_type"],
                "identification_methods": p["identification_methods"],
                "true_ATE": p["true_ATE"],
                "true_LATE": p["true_LATE"],
                "minimal_set": p["minimal_set"],
                "iv_instrument": p["iv_instrument"],
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
    max_nodes: int = 15,
) -> list[dict]:
    """Generate a problem set. Returns a list of problem dicts.

    Stratification:
      backdoor_standard: 35%
      backdoor_empty:    15%
      frontdoor:         40%
      iv:                10%

    ATE/LATE coverage: bin-based rejection to avoid clustering in [-0.5, 0.5].
    Each 0.5-unit bin of [-4, 4] is capped at max_per_bin per problem type.
    """
    rng = random.Random(seed)

    fracs = {
        "backdoor_standard": 0.35,
        "backdoor_empty": 0.15,
        "frontdoor": 0.40,
        "iv": 0.10,
    }

    def _problem_key(p: dict) -> tuple:
        return (
            tuple(tuple(e) for e in p["edges"]),
            tuple(p["nodes"]),
            tuple(p["observed_nodes"]),
        )

    def _ate_bin(val: float) -> int:
        """Map ATE/LATE value in [-4, 4] to bin index 0..15 (width 0.5)."""
        clamped = max(-4.0, min(3.9999, val))
        return int((clamped + 4.0) / 0.5)

    def _sample_bucket(ptype: str, n_target: int) -> list[dict]:
        problems = []
        seen = set()
        ate_bins: dict[int, int] = {}
        max_per_bin = max(1, int(n_target * 0.12))
        max_attempts = n_target * 1000

        for _ in range(max_attempts):
            if len(problems) >= n_target:
                break

            if ptype == "backdoor_standard":
                p = _try_sample_backdoor(rng, min_nodes, max_nodes, 0.4, 0.2, empty=False)
            elif ptype == "backdoor_empty":
                p = _try_sample_backdoor(rng, min_nodes, max_nodes, 0.4, 0.0, empty=True)
            elif ptype == "frontdoor":
                p = _try_sample_frontdoor(rng, min_nodes, max_nodes, 0.4)
            else:
                p = _try_sample_iv(rng, min_nodes, max_nodes, 0.4)
   
            if p is None:
                continue

            key = _problem_key(p)
            if key in seen:
                continue

            # ATE/LATE coverage bin check
            val = p["true_LATE"] if ptype == "iv" else p["true_ATE"]
            if val is None:
                continue
            b = _ate_bin(val)
            #if ate_bins.get(b, 0) >= max_per_bin:
            #    continue  # bin saturated; skip this problem

            seen.add(key)
            problems.append(p)
            if len(problems) % 25 == 0:
                print('generated {} problems for problems of type {}'.format(len(problems), ptype)) 
            ate_bins[b] = ate_bins.get(b, 0) + 1

        return problems

    problems = []
    for ptype, frac in fracs.items():
        n_target = max(1, round(n * frac))
        bucket = _sample_bucket(ptype, n_target)
        problems.extend(bucket)
        print(f"generated {len(bucket)}/{n_target} {ptype} problems")

    rng.shuffle(problems)
    return problems
